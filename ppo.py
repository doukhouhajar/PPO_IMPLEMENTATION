import argparse
import os
from distutils.util import strtobool
import time
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical 
from torch.utils.tensorboard import SummaryWriter

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id,render_mode="rgb_array") # if want to record add: render_mode="rgb_array"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0)
            #env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='name of the experiement')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='gym env id')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='use determinitic PyTorch when possible') # implement with torch.use_deterministic_algorithms
    parser.add_argument('--mps', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='enable MPS when available') 
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='enable CUDA when available')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False,
                         nargs='?', const=True, help='if toggled, this expirement will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='cleanRL', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='team of wandb project')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?', const=True, help='capture video of the agent performance')
    # Algorithm arguments
    parser.add_argument('--num-envs', type=int, default=4, help='number of parallel game enviroments')
    parser.add_argument('--num-steps', type=int, default=128, help='number of steps to run in each env per policy rollout') # rollouts data = 4*128
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', help='learning rate annealing for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='use General Advantage Estimation for advantange computation') 
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda for gae')
    parser.add_argument('--num-minibatches', type=int, default=4, help ='number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=4, help='the k epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='toggles advantages normalization')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='whether or not use a clipped loss for the value funtion, as per the paper')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='coefficient of the entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='the max norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None, help='the target KL divergence threshold') # for early stopping, also in OpenAI Spinning default=0.015
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__=="__main__":
    args = parse_args()
    #print(args) 
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"   
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda
                          else "mps" if torch.backends.mps.is_available() and args.mps
                          else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    #print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    #print("envs.single_action_space.n", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    #print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)+ envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()   
    next_obs, info = envs.reset(seed=args.seed) #envs.reset return a tuple : (obs, info)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device) # Tensor vs as_tensor
    num_updates = args.total_timesteps // args.batch_size
    #print(num_updates)
    #print("next_obs.shape", next_obs.shape)
    #print("agent.get_value(next_obs)", agent.get_value(next_obs))
    #rint("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    #print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))

    for update in range (1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)  # Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "_episode" in info: # for vector envs, info is a dict of arrays, not a list of dicts
                for i, finished in enumerate(info["_episode"]):
                    if finished:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r'][i]}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"][i], global_step)

            
        # bootstrap reward if not done 
        with torch.no_grad():
            next_value =  agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value 
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # debug variable
                with torch.no_grad():
                    old_approc_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()] # how often the clipped objective is actually triggered


                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_sdvantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss 
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # we max negatives, tandis que in the paper they min positives
                
                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean() # entropy = chaos, max entropy = more exploration
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef # min policy loss and values loss, and max entropy loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y    # if value function is  a good indicator of the returns   

        # record rewards 
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("loss/value_loss", v_loss.item(), global_step)    
        writer.add_scalar("loss/policy_loss", pg_loss.item(), global_step) 
        writer.add_scalar("loss/entropy", entropy_loss.item(), global_step) 
        writer.add_scalar("loss/approx_kl", approx_kl.item(), global_step) 
        writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step) 
        writer.add_scalar("loss/explained_variance", explained_var, global_step)  
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()





    '''
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda t: t % 100 == 0) #, record_video_trigger=lambda t: t % 100 == 0
    observation, info = env.reset()
    #episodic_return = 0
    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info1 = env.step(action)
        #episodic_return += reward
        if terminated or truncated :
            observation, info = env.reset()
            print(f"episodic return: {info1['episode']['r']}")
            #print(f"episodic return: {episodic_return}")
            #episodic_return = 0
    env.close()
    '''
    '''
    def make_env(gym_id):
        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda t: t % 100 == 0)
            return env
        return thunk
    
    envs =  gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    observation, info = envs.reset()
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
        done = terminated or truncated
        if "_episode" in info:
            for i, done in enumerate(info["_episode"]):
                if done:
                    print(f"episodic return {info['episode']['r'][i]}") 
    '''



  
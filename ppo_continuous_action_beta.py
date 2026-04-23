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
from torch.distributions.beta import Beta
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0: # record video for the first environment
            env = gym.make(gym_id,render_mode="rgb_array") 
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0)
        else : 
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),  env.observation_space) # the transformation we do (clipping) does not change the obs space so we use the same obs space, otherways we must provide the new obs space after transformation
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10)) # be careful this can make the task harder for the critic
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed) # to seed the env 
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            )
        # two heads: alpha and beta. Also init with small std so initial params are near 1 
        self.actor_alpha = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(64, act_dim), std=0.01)

        # action space bounds for rescaling 
        self.register_buffer("act_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("act_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        self.register_buffer("act_scale", self.act_high - self.act_low)
        self.register_buffer("log_act_scale", torch.log(self.act_scale))

    def get_dist(self, x):
        features = self.actor_backbone(x)
        alpha = F.softplus(self.actor_alpha(features)) + 1.0 # to assure >1 so that distribution is smoother and unimodal 
        beta = F.softplus(self.actor_beta(features)) + 1.0
        return Beta(alpha, beta)

    def get_value(self, x):
        return self.critic(x)
    '''
    def get_action_and_value(self, x, action=None):
        dist = self.get_dist(x)
        if action is None:
            action_01 = dist.sample()  # in (0, 1)
        else:
            action_01 = (action - self.act_low) / (self.act_high - self.act_low) 
            action_01 = action_01.clamp(1e-6, 1 - 1e-6)  # for safety
        
        # then we rescale to actual action space for env interaction 
        action_env = action_01 * (self.act_high - self.act_low) + self.act_low
        return (action_env, dist.log_prob(action_01).sum(1), dist.entropy().sum(1), self.critic(x),)'''
        # Include the Jacobian correction in the returned log-prob and correct entropy too NOT REALLY IMPORTANT ION OUR CASE
    def get_action_and_value(self, x, action=None):
        dist = self.get_dist(x)

        if action is None:
            action_01 = dist.sample()
            action_01 = action_01.clamp(1e-6, 1 - 1e-6)
        else:
            action_01 = (action - self.act_low) / self.act_scale
            action_01 = action_01.clamp(1e-6, 1 - 1e-6)

        action_env = self.act_low + self.act_scale * action_01

        # log prob in normalized space
        logprob_01 = dist.log_prob(action_01).sum(dim=1)

        # affine transform correction: log |det da/du| = sum log(scale), so log p(a) = log p(u) - sum log(scale)
        logprob_env = logprob_01 - self.log_act_scale.sum()

        # differential entropy under affine transform:
        entropy_env = dist.entropy().sum(dim=1) + self.log_act_scale.sum()

        value = self.critic(x)
        return action_env, logprob_env, entropy_env, value

# to recompute Advantages every data pass, and not every rollout 
def recompute_gae(agent, obs, next_obses, rewards, dones, terminateds, gamma, gae_lambda):
    num_steps, num_envs = rewards.shape
    obs_shape = obs.shape[2:]

    with torch.no_grad():
        flat_obs = obs.reshape((-1,) + obs_shape)
        flat_next_obs = next_obses.reshape((-1,) + obs_shape)

        values = agent.get_value(flat_obs).view(num_steps, num_envs)
        next_values = agent.get_value(flat_next_obs).view(num_steps, num_envs)

        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)

        for t in reversed(range(num_steps)):
            # stop GAE recursion across episode boundary
            nextnonterminal = 1.0 - dones[t]

            # but only suppress bootstrap on true termination
            nextnotterminated = 1.0 - terminateds[t]

            delta = rewards[t] + gamma * next_values[t] * nextnotterminated - values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

    return advantages, returns, values

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='name of the experiement')
    parser.add_argument('--gym-id', type=str, default="HalfCheetah-v5", help='gym env id')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000, help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='use determinitic PyTorch when possible') # implement with torch.use_deterministic_algorithms
    parser.add_argument('--mps', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='enable MPS when available') 
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='enable CUDA when available')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False,
                         nargs='?', const=True, help='if toggled, this expirement will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='PPO', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='team of wandb project')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?', const=True, help='capture video of the agent performance')
    # Algorithm arguments
    parser.add_argument('--num-envs', type=int, default=4, help='number of parallel game enviroments')#4
    parser.add_argument('--num-steps', type=int, default=2048, help='number of steps to run in each env per policy rollout') # 2048 # rollouts data = 4*128
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='learning rate annealing for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='use General Advantage Estimation for advantange computation') 
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda for gae')
    parser.add_argument('--num-minibatches', type=int, default=32, help ='number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=10, help='the k epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?',
                        const=True, help='toggles advantages normalization')
    parser.add_argument('--clip-coef', type=float, default=0.25, help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=False, nargs='?',
                        const=True, help='whether or not use a clipped loss for the value funtion, as per the paper')
    parser.add_argument('--ent-coef', type=float, default=0.001, help='coefficient of the entropy')  # try 0.01 afterwards, or try annealing : 0.01 * (1.0 - (update - 1.0) / num_updates)
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='the max norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=0.015, help='the target KL divergence threshold') # for early stopping, also in OpenAI Spinning default=0.015
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__=="__main__":
    args = parse_args()
    assert args.batch_size % args.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
    #print(args) 
    run_name = f"{args.gym_id}_{args.seed}_{int(time.time())}"   
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=f"ppo-{args.gym_id}",
            monitor_gym=False, # for W&B and Gymnasium record video systems
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
    torch.use_deterministic_algorithms(args.torch_deterministic) # because GPU uses parallel algorithms so it might produce slightly diff results

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda
                          else "mps" if torch.backends.mps.is_available() and args.mps
                          else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

   
    agent = Agent(envs).to(device)
    #print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device) # HalfCheetah-v5 has a 6-dimensional continuous action space
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminateds = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)

    # Start the game
    global_step = 0
    start_time = time.time()   
    ######## I am seeding the env in thunk - à confirmer !
    next_obs, info = envs.reset() #envs.reset(seed=args.seed) #envs.reset return a tuple : (obs, info) 
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device) # Tensor vs as_tensor
    num_updates = args.total_timesteps // args.batch_size
   
    for update in range (1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates # goes from 1 to 0
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs

            # action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs) # action coming out of get_action_and_value has shape (num_envs, 6) = (1, 6)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data
            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)  # Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead
            dones[step] = torch.as_tensor(done, dtype=torch.float32, device=device).view(-1)
            terminateds[step] = torch.as_tensor(terminated, dtype=torch.float32, device=device).view(-1)

            # for bootstrapping use the true next observation of this transition
            bootstrap_obs_np = next_obs_np.copy()

            if "final_observation" in info:
                for i in range(args.num_envs):
                    if done[i] and info["final_observation"][i] is not None:
                        bootstrap_obs_np[i] = info["final_observation"][i]

            bootstrap_obs = torch.as_tensor(bootstrap_obs_np, dtype=torch.float32, device=device)
            next_obses[step] = bootstrap_obs

            with torch.no_grad():
                next_values_buf[step] = agent.get_value(bootstrap_obs).flatten()

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            if "_episode" in info: # for vector envs, info is a dict of arrays, not a list of dicts
                for i, finished in enumerate(info["_episode"]):
                    if finished:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r'][i]}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"][i], global_step)

            
        # bootstrap reward if not done 
        with torch.no_grad():
            if args.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

                for t in reversed(range(args.num_steps)):
                    # stop GAE recursion when episode ended (terminated OR truncated)
                    nextnonterminal = 1.0 - dones[t]

                    # but only remove bootstrap on true termination
                    nextnotterminated = 1.0 - terminateds[t]

                    delta = rewards[t] + args.gamma * next_values_buf[t] * nextnotterminated - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam

                returns = advantages + values

            else:
                returns = torch.zeros_like(rewards, device=device)

                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_return = next_values_buf[t]
                    else:
                        # if step t ended the episode do not continue with returns[t+1] because returns[t+1] belongs to the next episode
                        next_return = torch.where(
                            dones[t].bool(),
                            next_values_buf[t],
                            returns[t + 1],
                        )

                    # only suppress bootstrap on true termination
                    returns[t] = rewards[t] + args.gamma * (1.0 - terminateds[t]) * next_return

                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # to flatten only tensors that do not change across PPO epochs
        #b_advantages = advantages.reshape(-1)
        #b_returns = returns.reshape(-1)
        #b_values = values.reshape(-1)

        # optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # recompute advantages/returns/values using the current critic
            advantages, returns, values_epoch = recompute_gae(agent, obs, next_obses, rewards, dones, terminateds, args.gamma, args.gae_lambda)
            # noe flatten
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values_old = values.reshape(-1)
            
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # debug variable
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [(((ratio - 1.0).abs() > args.clip_coef).float().mean()).item()] # how often the clipped objective is actually triggered
                    # .item() because this is just a logging scalar and afterwards we will use np.mean, so we convert each tensor to a py scalar

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss 
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # we max negatives, tandis que in the paper they min positives
                
                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values_old[mb_inds] + torch.clamp( 
                        newvalue - b_values_old[mb_inds],
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
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # global l2 does not exceed 0.5
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        #y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        with torch.no_grad():
            y_pred = agent.get_value(b_obs).view(-1).cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y    # if value function is a good indicator of the returns   

        # record rewards 
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("loss/value_loss", v_loss.item(), global_step)    
        writer.add_scalar("loss/policy_loss", pg_loss.item(), global_step) 
        writer.add_scalar("loss/entropy", entropy_loss.item(), global_step) 
        writer.add_scalar("loss/approx_kl", approx_kl.item(), global_step) 
        writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step) 
        writer.add_scalar("loss/explained_variance", explained_var, global_step)  
        # debug critic 
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/values_std", torch.as_tensor(y_pred).std().item(), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/var_y", float(var_y), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step) # Steps Per Second

    envs.close()
    writer.close()



  
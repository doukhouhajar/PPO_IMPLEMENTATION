import argparse
import os
from distutils.util import strtobool
import time
import random
import gym.wrappers.record_episode_statistics
import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observatiom_space.seed(seed)
        return env
    return thunk

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
    args = parser.parse_args()
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



  
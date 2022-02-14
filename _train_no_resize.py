# %%
from pathlib import Path

import gym
import navstack_gym
import torch

import numpy as np
import matplotlib.pyplot as plt
import h5py

from gym.wrappers import TimeLimit, FrameStack
from algo.wrapper import ResizeObservation, TensorObservation

from pprint import pprint as p
from algo import utils
from algo.agent import make_DrQ_agent

def start_train():

    ################# Task Setting ####################
    # reference ./conf/ 
    TARGET = 'treasure_hunt_with_fixed_room'
    EXPERIMENT_NAME = 'maintask_no_resize'
    ###################################################

    ################# Fine-Tuning #####################
    enable_load = False
    load_agent = 'subtask_map_fixed_room'
    load_agent_path = utils.get_workspace_path(load_agent) / 'agent'
    ###################################################

    cfg = utils.load_param('conf', 'train_config.yaml', [f'task={TARGET}', f'experiment_name={EXPERIMENT_NAME}'])

    utils.create_workspace(cfg.experiment_name)
    workspace_path = utils.get_workspace_path(cfg.experiment_name)

    p(cfg)
    env_mode = cfg.env_config.full
    room_mode = cfg.room_config.hodoyoi

    gpu = torch.cuda.current_device() if torch.cuda.is_available() else -1

    env = gym.make(cfg.env_id, **env_mode)
    env = FrameStack(env, cfg.stack_num)
    env = TensorObservation(env)
    env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    print(f'timelimit: \t{env.spec.max_episode_steps}')
    print(f'obs_space: \t{env.observation_space.shape} \naction_space: \t{env.action_space.shape}')
    print(f'obs_size: \t{env.observation_space.low.size}')
    print(f'action_size: \t{env.action_space.low.size}')
    print(f'obs_stack: \t{cfg.stack_num}')

    if cfg.set_seed:
        np.random.seed(cfg.env_seed)
        
    obs = env.reset(**room_mode)

    agent = make_DrQ_agent(
        experiment_name=cfg.experiment_name, 
        obs_space=env.observation_space, 
        action_space=env.action_space,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        lr=cfg.lr,
        image_pad=cfg.image_pad,
        gamma=cfg.gamma,
        replay_start_size=cfg.replay_start_size,
        capacity=cfg.capacity,
        gpu=gpu,
        batch_size=cfg.batch_size,
        update_interval=cfg.update_interval,
        is_persistent_buffer=True)

    if enable_load:
        agent.load(str(load_agent_path))

    result_filename = f'{str(workspace_path)}/result.hdf5'

    with h5py.File(result_filename, 'a') as f:
        agent_statics_group = 'agent_statics'
        agent_statics_labels = [s[0] for s in agent.get_statistics()]
        for l in agent_statics_labels:
            f.create_dataset(f'{agent_statics_group}/{l}', shape=(cfg.n_episodes,))

        episode_rewards_group = 'episode_rewards'
        episode_rewards_labels = ['episode', 'total_reward']
        for l in episode_rewards_labels:
            f.create_dataset(f'{episode_rewards_group}/{l}', shape=(cfg.n_episodes,))

    import datetime
    start = datetime.datetime.now()
    print(f'\n====================\nStart Training: {start}\n====================\n')

    for i in range(cfg.n_episodes):
        obs = env.reset(is_generate_room=cfg.change_room, is_generate_pose=cfg.change_pose, **room_mode)
        R = 0  # return (sum of rewards)
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            R += reward
            
            agent.observe(obs, reward, done, done)
            if done:
                break

        print('=============')
        print(f'episode: {i}, reward: {R}')
        print(f'statics: {agent.get_statistics()}')
        
        ## record result
        with h5py.File(result_filename, 'a') as f:
            f[episode_rewards_group]['episode'][i] = i+1
            f[episode_rewards_group]['total_reward'][i] = R
            for statics in agent.get_statistics():
                f[agent_statics_group][statics[0]][i] = statics[1]

    print('Finished')
    print(f'\n====================\nTraining Time: {datetime.datetime.now() - start}\n====================\n')

    agent.save(f'{str(workspace_path)}/agent')

# @hydra.main(config_path="conf", config_name="train_config")
# def main(cfg):
#     start_train(cfg)

if __name__ == '__main__':
    start_train()
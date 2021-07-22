#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/7/21 4:59 下午
# @File  : ppo2.py
# @Author: johnson
# @Desc  :

import os
import base64
import pickle
import zlib
import gym
import numpy as np
import pandas as pd
import torch as th
from torch import nn, tensor
from collections import deque
from gym.spaces import Box, Discrete
from gfootball.env import create_environment, observation_preprocessing
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback


class FootballGym(gym.Env):
    spec = None
    metadata = None
    def __init__(self, config=None, render=False):
        """
        重新封装下环境
        Args:
            config ():
        """
        super(FootballGym, self).__init__()
        #默认中等难度
        env_name = "11_vs_11_easy_stochastic"
        rewards = "scoring,checkpoints"
        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)
        self.env = create_environment(
            env_name=env_name,
            stacked=False,
            representation="raw",
            rewards=rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=render,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0)
        # 19种动作
        self.action_space = Discrete(19)
        # 观察空间
        self.observation_space = Box(low=0, high=255, shape=(72, 96, 16), dtype=np.uint8)
        #奖励范围
        self.reward_range = (-1, 1)
        self.obs_stack = deque([], maxlen=4)

    def transform_obs(self, raw_obs):
        obs = raw_obs[0]
        obs = observation_preprocessing.generate_smm([obs])
        if not self.obs_stack:
            self.obs_stack.extend([obs] * 4)
        else:
            self.obs_stack.append(obs)
        obs = np.concatenate(list(self.obs_stack), axis=-1)
        obs = np.squeeze(obs)
        return obs

    def reset(self):
        self.obs_stack.clear()
        obs = self.env.reset()
        obs = self.transform_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        obs = self.transform_obs(obs)
        return obs, float(reward), done, info


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels, stride)

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class FootballCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]  # channels x height x width
        self.cnn = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=32),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=52640, out_features=features_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))

def make_env(config=None, rank=0, render=False):
    def _init():
        env = FootballGym(config, render=render)
        log_file = os.path.join(".", str(rank))
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init

def do_train():
    check_env(env=FootballGym(), warn=True)
    scenario_name = "11_vs_11_easy_stochastic"
    n_envs = 1
    train_env = DummyVecEnv([make_env({"env_name":scenario_name})])
    eval_env = VecTransposeImage(DummyVecEnv([make_env({"env_name":scenario_name, "rewards":"scoring"})]))

    n_steps = 512
    policy_kwargs = dict(features_extractor_class=FootballCNN,
                         features_extractor_kwargs=dict(features_dim=256))
    model = PPO(CnnPolicy, train_env,
                 policy_kwargs=policy_kwargs,
                 learning_rate=0.000343,
                 n_steps=n_steps,
                 batch_size=8,
                 n_epochs=2,
                 gamma=0.993,
                 gae_lambda=0.95,
                 clip_range=0.08,
                 ent_coef=0.003,
                 vf_coef=0.5,
                 max_grad_norm=0.64,
                 verbose=1)

    eval_freq=3001*10
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path='models/',
                                 log_path='logs/', eval_freq=eval_freq, n_eval_episodes = 1,
                                 deterministic=True, render=False)


    total_timesteps = 3001*450
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

def do_eval():
    trained_checkpoint = "models/best_model.zip"
    scenario_name = "11_vs_11_easy_stochastic"
    eval_env = VecTransposeImage(DummyVecEnv([make_env({"env_name":scenario_name, "rewards":"scoring"}, render=True)]))
    #模型的配置
    n_steps = 512
    policy_kwargs = dict(features_extractor_class=FootballCNN,
                         features_extractor_kwargs=dict(features_dim=256))
    model = PPO(CnnPolicy, eval_env,
                policy_kwargs=policy_kwargs,
                learning_rate=0.000343,
                n_steps=n_steps,
                batch_size=8,
                n_epochs=2,
                gamma=0.993,
                gae_lambda=0.95,
                clip_range=0.08,
                ent_coef=0.003,
                vf_coef=0.5,
                max_grad_norm=0.64,
                verbose=1)
    model.load(trained_checkpoint)
    #环境重置，方便测试模型
    obs = eval_env.reset()
    # 测试模型
    print(f"开始测试模型效果：")
    step = 0
    for i in range(1000):
        step += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        print(f"循环第{i}次，第{step}个step操作{action}，奖励{reward}")
        # eval_env.render()
        if done:
            print(f"这一个episode足球结束，开始下一个step测试")
            step = 0
            obs = eval_env.reset()
    eval_env.close()

if __name__ == '__main__':
    # do_train()
    do_eval()
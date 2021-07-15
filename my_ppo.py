#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/7/15 2:47 下午
# @File  : my_ppo.py
# @Author: johnson
# @Desc  : 使用stable-baseline3 的PPO算法

import gfootball.env as football_env
import argparse
from stable_baselines3 import PPO

def model_config(parser):
    parser.add_argument('--level', default='academy_empty_goal_close', type=str, help='定义要解决的问题，要使用的游戏场景，一共11种')
    parser.add_argument('--state', default='extracted_stacked', type=str, help='extracted 或者extracted_stacked')
    parser.add_argument('--reward_experiment', default='scoring', type=str, help='奖励的方式，"scoring" 或者 "scoring,checkpoints"')
    parser.add_argument('--num_timesteps', default=20000, type=int, help='训练的时间步数')
    parser.add_argument('--nsteps', default=128, type=int, help='batch size 是 nsteps')
    return parser

def data_config(parser):
    parser.add_argument('--log_dir', default='logs', help='日志目录')
    parser.add_argument('--tensorboard', action='store_true')
    return parser

def train_config(parser):
    parser.add_argument('--noptepochs', default=4, type=int, help='每个epoch更新')
    parser.add_argument('--dump_scores', action='store_true', default=True, help="打印分数")
    parser.add_argument('--dump_full_episodes', action='store_true', default=True, help="每个epoch打印")
    parser.add_argument('--render', action='store_true',default=False, help="是否显示动画")
    parser.add_argument('--debug', action='store_true', help="print debug info")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    env = football_env.create_environment(
        env_name=args.level, stacked=('stacked' in args.state),
        rewards=args.reward_experiment,
        logdir=args.log_dir,
        write_goal_dumps=args.dump_scores,
        write_full_episode_dumps=args.dump_full_episodes,
        render=args.render,
        dump_frequency=50)
    #模型的配置
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.num_timesteps)
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

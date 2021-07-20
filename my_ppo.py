#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/7/15 2:47 下午
# @File  : my_ppo.py
# @Author: johnson
# @Desc  : 使用stable-baseline3 的PPO算法
import os
import sys
import time
import gfootball.env as football_env
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

#所有可用的训练模式
levels = ['11_vs_11_competition','11_vs_11_easy_stochastic','11_vs_11_hard_stochastic','11_vs_11_kaggle','11_vs_11_stochastic','1_vs_1_easy',
          '5_vs_5','academy_3_vs_1_with_keeper','academy_corner','academy_counterattack_easy','academy_counterattack_hard','academy_empty_goal',
          'academy_empty_goal_close','academy_pass_and_shoot_with_keeper','academy_run_pass_and_shoot_with_keeper','academy_run_to_score',
          'academy_run_to_score_with_keeper','academy_single_goal_versus_lazy']

def model_config(parser):
    parser.add_argument('--level', default='5_vs_5', type=str, choices=levels, help='定义要解决的问题，要使用的游戏场景，一共11种')
    parser.add_argument('--state', default='extracted_stacked', type=str, help='extracted 或者extracted_stacked')
    parser.add_argument('--reward_experiment', default='scoring,checkpoints', type=str, help='奖励的方式，"scoring" 或者 "scoring,checkpoints, 注意奖励方式，如果踢全场，最好用2种结合"')
    parser.add_argument('--num_timesteps', default=20000000, type=int, help='训练的时间步数，一般可以200万个step')
    parser.add_argument('--nsteps', default=128, type=int, help='batch size 是 nsteps')
    parser.add_argument('--output_path', default='output', type=str, help='模型保存的路径,模型名称根据时间自动命名,默认为output')
    parser.add_argument('--model_save_prefix', default='ppo_model', type=str, help='模型保存的名称的前缀')
    parser.add_argument('--model_save_frequency', default=100000, type=int, help='每所少个step保存一次模型，默认为100000')
    return parser

def data_config(parser):
    parser.add_argument('--log_dir', default='logs', help='日志目录')
    parser.add_argument('--tensorboard', action='store_true')
    return parser

def train_config(parser):
    parser.add_argument('--do_train', action='store_true', help="训练并测试模型")
    parser.add_argument('--do_eval', action='store_true', help="只测试模型，需要给出要加载的模型checkpoint")
    parser.add_argument('--load_checkpoint', default='output/ppo_model_20000000_steps.zip', type=str, help="只测试模型，需要给出要加载的模型checkpoint")
    parser.add_argument('--initial_checkpoint', default='', type=str, help="训练时，使用哪个模型继续训练，默认为空")
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
    if args.do_eval:
        # 那么测试时用真实的时间，那么足球动画就不会加速，能看清
        other_config_options = {'real_time':True}
    else:
        other_config_options = {}
    env = football_env.create_environment(
        env_name=args.level, stacked=('stacked' in args.state),
        rewards=args.reward_experiment,
        logdir=args.log_dir,
        write_goal_dumps=args.dump_scores,
        write_full_episode_dumps=args.dump_full_episodes,
        render=args.render,
        dump_frequency=50,
        other_config_options=other_config_options,)
    #模型的配置
    model = PPO("MlpPolicy", env, verbose=1)
    if args.initial_checkpoint:
        model.load(args.initial_checkpoint)
    if args.do_train:
        print(f"开始训练，会耗时较长, 即将训练{args.num_timesteps}个step,模型保存频率为{args.model_save_frequency}")
        checkpoint_callback = CheckpointCallback(save_freq=args.model_save_frequency, save_path=args.output_path,
                                                 name_prefix=args.model_save_prefix)
        model.learn(total_timesteps=args.num_timesteps, callback=checkpoint_callback)
        #保存最后一次训练好的训练好的模型
        # 模型保存的位置/output/0714095907.zip
        save_path = os.path.join(args.output_path, args.model_save_prefix + '_final.zip')
        model.save(save_path)
    elif args.do_eval:
        print(f"评估模式，直接加载模型")
        model.load(args.load_checkpoint)
    else:
        print(f"请选择需要训练还是测试评估, --do_train,  --do_eval")
        sys.exit(0)
    #环境重置，方便测试模型
    obs = env.reset()
    # 测试模型
    print(f"开始测试模型效果：")
    step = 0
    for i in range(1000):
        step += 1
        print(f"循环第{i}次，开始进行第{step}个step操作")
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(f"这一个episode足球结束，开始下一个step测试")
            step = 0
            obs = env.reset()
    env.close()

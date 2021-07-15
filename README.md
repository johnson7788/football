# Google Research Football

这个资源库包含一个基于开源游戏Gameplay Football的RL环境。它是由谷歌大脑团队为研究目的创建的。

有用的链接：
* __(NEW!)__ [GRF Kaggle competition](https://www.kaggle.com/c/google-football) - 参加比赛，与他人玩游戏，赢得奖品，成为GRF英雄
* [Run in Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb) - 快速开始训练教程
* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)   论文
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)  博客
* [Google Research Football on Cloud](https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e)
* [Mailing List](https://groups.google.com/forum/#!forum/google-research-football) - please use it for communication with us (comments / suggestions / feature ideas)


For non-public matters that you'd like to discuss directly with the GRF team,
please use google-research-football@google.com.

We'd like to thank Bastiaan Konings Schuiling, who authored and open-sourced the original version of this game.


## Quick Start

### Colab快速教程

详见 [Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb),

这种方法不支持游戏在屏幕上的渲染--如果你想看到游戏的运行，请使用下面的方法。

### Docker方法

This is the recommended way to avoid incompatible package versions.
Instructions are available [here](gfootball/doc/docker.md).

### 本地方法

#### 1. 依赖包
#### Linux
```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip

python3 -m pip install --upgrade pip setuptools psutil
```

#### Mac OS X
First install [brew](https://brew.sh/). 安装brew
Next install required packages:

```
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3
```

#### 2a. 使用pip安装
```
python3 -m pip install gfootball
```

#### 2b. 使用源码安装

```
git clone https://github.com/google-research/football.git
cd football
```

Optionally you can use [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```
python3 -m venv football-env
source football-env/bin/activate
```

安装，会自动编译C++环境，耗时几分钟

```
python3 -m pip install .
```


#### 3. 游戏时间
```
python3 -m gfootball.play_game --action_set=full
```
确保已经键盘映射 [keyboard mappings](#keyboard-mappings).
To quit the game press Ctrl+C in the terminal.

# Contents #

* [Running training](#training-agents-to-play-GRF)
* [Playing the game](#playing-the-game)
    * [Keyboard mappings](#keyboard-mappings)
    * [Play vs built-in AI](#play-vs-built-in-AI)
    * [Play vs pre-trained agent](#play-vs-pre-trained-agent)
    * [Trained checkpoints](#trained-checkpoints)
* [Environment API](gfootball/doc/api.md)
* [Observations & Actions](gfootball/doc/observation.md)
* [Scenarios](gfootball/doc/scenarios.md)
* [Multi-agent support](gfootball/doc/multi_agent.md)
* [Running in docker](gfootball/doc/docker.md)
* [Saving replays, logs, traces](gfootball/doc/saving_replays.md)
* [Imitation Learning](gfootball/doc/imitation.md)

## Training agents to play GRF

### Run training
如果使用TF TensorFlow训练，需要额外配置

- Update PIP, so that tensorflow 1.15 is available: `python3 -m pip install --upgrade pip setuptools`
- TensorFlow: `python3 -m pip install tensorflow==1.15.*` or
  `python3 -m pip install tensorflow-gpu==1.15.*`, depending on whether you want CPU or
  GPU version;
- Sonnet: `python3 -m pip install dm-sonnet==1.*`;
- OpenAI Baselines:
  `python3 -m pip install git+https://github.com/openai/baselines.git@master`.

Then:

- 运行PPO实验在场景`academy_empty_goal` 下, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_empty_goal_close`
- To run on `academy_pass_and_shoot_with_keeper` scenario, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_pass_and_shoot_with_keeper`

为了训练被保存的replay，运行
`python3 -m gfootball.examples.run_ppo2 --dump_full_episodes=True --render=True`

为了再现论文中的PPO结果，请参考。

- gfootball/examples/repro_checkpoint_easy.sh
- gfootball/examples/repro_scoring_easy.sh

## Playing the game
请注意，玩游戏是通过环境实现的，所以人类控制的玩家使用与agent相同的界面。一个重要的含义是，每100毫秒有一个动作报告给环境，这可能会导致游戏时的滞后效应。

### Keyboard mappings
游戏定义了以下键盘映射（对于“键盘”player类型）：

* `ARROW UP` - 向上跑
* `ARROW DOWN` - 向下跑
* `ARROW LEFT` - 向左跑
* `ARROW RIGHT` - 向右跑
* `S` - 在进攻模式下短传，防御模式下用于施压。
* `A` - 在进攻模式下高传，防御模式下用于滑铲。
* `D` - 在进攻模式下射门，防御模式下用于组队施压。
* `W` - 在进攻模式下长传，防御模式下用于守门员施压。
* `Q` - 防守模式用于切换激活的队员
* `C` - 在进攻模式下盘球
* `E` - 冲刺.

### Play vs built-in AI
运行`python3 -m gfootball.play_game --action_set=full`。默认情况下，它启动基本场景，左侧球员由键盘控制。
支持不同类型的球员（游戏手柄、外部机器人、agent...）。对于可能的选项，运行`python3 -m gfootball.play_game -helpfull`。

### Play vs pre-trained agent

特别是，人们可以用以下命令与用`run_ppo2`脚本训练的agent进行比赛（注意没有action_set标志，因为PPOagent使用默认动作集）。
`python3 -m gfootball.play_game --players "keyboard:left_players=1;ppo2_cnn:right_players=1,checkpoint=$YOUR_PATH"`

### Trained checkpoints
我们为以下方案提供了受过训练的PPOcheckpoint：
  - [11_vs_11_easy_stochastic](https://storage.googleapis.com/gfootball/11_vs_11_easy_stochastic_v2),
  - [academy_run_to_score_with_keeper](https://storage.googleapis.com/gfootball/academy_run_to_score_with_keeper_v2).

为了看到checkpoint玩游戏，运行
`python3 -m gfootball.play_game --players "ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT" --level=$LEVEL`,
其中`$CHECKPOINT`是下载的checkpoint的路径。请注意，这些checkpoint是用Tensorflow 1.15版本训练的。
使用不同的Tensorflow版本可能会导致错误。运行这些checkpoint的最简单方法是通过提供的`Dockerfile_examples`图像。

See [running in docker](gfootball/doc/docker.md) for details (just override the default Docker definition with `-f Dockerfile_examples` parameter).

为了针对checkpoint进行训练，你可以向create_environment函数传递'extra_players'参数。
例如，extra_players='ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT'。
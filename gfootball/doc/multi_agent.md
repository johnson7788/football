# Multiagent support #

使用play_game脚本（详见 "自己玩游戏 "部分），可以设置多个agent之间的游戏。
`players`命令行参数是一个以逗号分隔的两队队员名单。
例如，要用gamepad自己玩，你的团队中有两个懒惰的机器人，对三个机器人，你可以运行
`python3 -m gfootball.play_game --players=gamepad:left_players=1;lazy:left_players=2;bot:right_players=1;bot:right_players=1;bot:right_players=1`.

注意到懒人玩家使用了`left_players=2`，机器人玩家不支持它。

你可以通过在env/players目录下添加其实现来实现你自己的控制多个玩家的player（不需要其他改动）。
请看一下现有的player代码，以获得一个实现的例子。

要训练一个控制多个球员的策略，必须做以下工作。
- 将 "number_of_players_agent_controls "传递给 "create_environment"，定义你要控制的球员数量。
- 不要用一个动作调用'.step'函数，而是用一个动作数组调用，每个玩家一个动作

这取决于调用者是否以理想的方式解包/后处理它。
在examples/run_multiagent_rllib.py中可以找到一个训练多Agent的简单例子。

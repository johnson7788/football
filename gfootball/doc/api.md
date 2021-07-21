# Environment API #
谷歌研究足球环境遵循gym API设计:

* `reset()` - 将环境重置为初始状态.
* `observation, reward, done, info = step(action)` - 执行单一步骤.
* `observation = observation()` - 返回当前观察.
* `render(mode=human|rgb_array)` - 可以随时调用以启用渲染.
  与GYM API的主要区别是，调用`render`可以连续渲染episode（不需要在每一步都调用该方法）。
  调用`render`可以使像素在观察中可用。
  注意 - rendering 会大大减慢`step`方法的速度。
* `disable_render()` - 禁用以前启用的渲染“render”。
* `close()` - 关闭.

在标准API的基础上，我们提供了一些额外的方法:

* `state = get_state()` - 
  提供一个当前环境的状态，包含所有影响环境行为的值（随机数生成器的状态，当前玩家的心理模型，物理学等）。
  返回的状态是一个不透明的对象，可以被`set_state(state)` API方法使用，以恢复环境过去的状态。  
* `set_state(state)` - 
  使用`get_state()`将环境的状态恢复到之前的快照状态。
  这个方法可以用来检查从一个固定状态开始执行不同动作序列的结果。
* `write_dump(name)` -  将当前场景的写到磁盘上。包含episode中每一步的观测快照，可以用来离线分析episode的轨迹。
  
For example API usage have a look at [play_game.py](../play_game.py).

# Saving replays, logs, traces #

GFootball环境支持对场景进行记录，以便日后观看或分析。每个跟踪dump包括一个pickled的episode跟踪（观察、奖励、额外的调试信息）和一个带有渲染episode的AVI文件。
Pickled的episode跟踪可以在以后使用`replay.py`脚本进行回放。
默认情况下，为了不占用磁盘空间，跟踪转储被禁用。它们由以下一组参数控制。

-  `dump_full_episodes` - 记录每个整个episode的轨迹。
-  `dump_scores` - 保存记录分数的样本轨迹。 
-  `tracesdir` - 保存轨迹的目录。
-  `write_video` - 视频与轨迹一起记录。 如果渲染被禁用（`render`配置标志），视频包含一个简单的episode动画。

提供以下脚本以在轨迹转储上运行：

-  `dump_to_txt.py` - 将轨迹转换为人类可读形式。
-  `dump_to_video.py` - 将轨迹转换为2D表示视频。
-  `replay.py` - 使用环境replay给定的轨迹转储。

## Environment logs
环境使用`absl.logging`模块进行日志记录。
你可以通过设置--verbosity标志为下列数值之一来改变日志级别。

- `-1` - 警告，当遇到问题时只记录警告及以上。
- `0` - info (默认), 每个episode的统计数据和类似的信息也被记录下来。
- `1` - debug，包括额外的调试信息。
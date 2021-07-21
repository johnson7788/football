# Scenarios #
我们提供两组场景/级别：

* Football Benchmarks
   * __11_vs_11_stochastic__ 完整的90分钟足球比赛（中等困难）
   * __11_vs_11_easy_stochastic__ 完整90分钟足球比赛（简单）
   * __11_vs_11_hard_stochastic__ 完整的90分钟足球比赛（困难）

* Football Academy - 共有11个场景
   * __academy_empty_goal_close__ - 我们的球员在禁区内开始带球，需要面对一个空门得分。
   * __academy_empty_goal__ - 我们的球员带球从场地中央开始，需要对着一个空门得分。
   * __academy_run_to_score__ - 我们的球员带着球从场地中间开始，需要对着一个空门得分。五个对手球员从后面追赶我们的球员。
   * __academy_run_to_score_with_keeper__ - 我们的球员带球从场地中间开始，需要面对守门员得分。五个对手球员从后面追赶我们的球员。
   * __academy_pass_and_shoot_with_keeper__ - 我们的两名球员试图从禁区边缘进球，一名在有球的一侧，旁边有一名后卫。另一个在中间，unmarked，面对对手的守门员。
   * __academy_run_pass_and_shoot_with_keeper__ -  我们的两名球员试图从禁区边缘进球，一个在有球的一侧，unmarked。另一个在中间，挨着后卫，面对对手的守门员。
   * __academy_3_vs_1_with_keeper__ - 我们的三名球员试图从禁区边缘进球，两边各一名，另一名在中间。最初，中间的球员有球，并面对后卫。有一个对手的守门员。
   * __academy_corner__ - 标准的角球情况，只是角球手可以从角球处带球跑。
   * __academy_counterattack_easy__ - 4对1的反击，有守门员；两队所有剩余的球员都向后跑去。
   * __academy_counterattack_hard__ - 4对2的反击，有守门员；两队的所有剩余球员都向后跑去。
   * __academy_single_goal_versus_lazy__ - 完全的11对11的比赛，对手不能移动，但他们只能在球离他们足够近的时候拦截。我们的中后卫一开始就有球。

你可以通过在`gfootball/scenarios/`目录下添加一个新文件来增加你自己的方案。请看一下现有的方案，例如。

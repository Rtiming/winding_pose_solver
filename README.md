# winding_pose_solver

这个项目用于：

1. 根据中心线数据构建工艺局部坐标系。
2. 求解工具在 `Frame 2` 下的位姿。
3. 基于位姿结果在 RoboDK 中生成程序。

## 运行方式

当前默认入口已经是非交互模式，直接运行：

```bash
python main.py
```

程序会自动执行以下流程：

1. 读取 `data/validation_centerline.csv`
2. 构建局部坐标系
3. 求解名义工具位姿并刷新 `data/tool_poses_frame2.csv`
4. 连接 RoboDK，收集整条路径的多组 IK 候选
5. 在整条路径上做全局优化
6. 只把最终选中的关节序列写入 RoboDK 并生成程序

如果只是临时检查中心线、边界或局部坐标系方向，可运行：

```bash
python main.py --visualize
```

这个入口只做可视化，不会生成 RoboDK 程序。

## 运行环境

推荐使用 Conda 环境：

```bash
conda activate winding_pose_solver
```

常规依赖：

- `numpy`
- `pandas`
- `matplotlib`

如需生成 RoboDK 程序，还需要：

- 可用的 `robodk` Python API
- RoboDK 正在运行，或可被 API 自动拉起
- RoboDK 站点中存在与配置一致的机器人和参考坐标系

## 当前 RoboDK 生成逻辑

`src/robodk_program.py` 当前不是逐点接受 RoboDK 的默认单一 IK 解，而是采用下面的流程：

1. 对每个目标位姿同时调用 `SolveIK_All` 和 `SolveIK + 多组 seed`
2. 收集该点的多组候选关节解
3. 先用硬约束过滤候选
4. 再用动态规划在整条路径上选择一条全局代价最小的关节序列
5. 最后只把这一条最终结果写成 RoboDK target 和 program

全局优化目标主要包括：

- 尽量减小相邻点之间的关节变化
- 尽量减少 branch switching、wrist flip、J6 大幅旋转
- 惩罚大跳变
- 惩罚接近机器人关节限位的解
- 惩罚接近奇异位形的解

## 当前硬约束

RoboDK 程序生成阶段会强制过滤不满足以下条件的 IK 解：

- `A1` 必须在 `[-150, 30]` 度之间
- `A2` 必须小于 `115` 度

这些参数目前在 [main.py](/C:/Users/22290/Desktop/机械臂/仿真/winding_pose_solver/main.py) 顶部维护：

- `A1_MIN_DEG`
- `A1_MAX_DEG`
- `A2_MAX_DEG`

## 连续性硬约束

除了上面的单点关节约束，当前还增加了“点与点之间”的连续性硬约束：

- 相邻路径点之间，如果任一关节变化量超过设定阈值，这条转移会被直接判定为不可用
- 动态规划只会在剩余可行转移里寻找整条路径的全局最优解

当前默认阈值在 [main.py](/C:/Users/22290/Desktop/机械臂/仿真/winding_pose_solver/main.py) 顶部维护：

- `ENABLE_JOINT_CONTINUITY_CONSTRAINT = True`
- `MAX_JOINT_STEP_DEG = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)`

阈值顺序对应：

- `A1, A2, A3, A4, A5, A6`

也就是说，默认要求相邻两点之间：

- `A1` 变化不超过 `5` 度
- `A2` 变化不超过 `5` 度
- `A3` 变化不超过 `5` 度
- `A4` 变化不超过 `180` 度
- `A5` 变化不超过 `100` 度
- `A6` 变化不超过 `180` 度

如果后续需要改连续性约束，只需要改 `MAX_JOINT_STEP_DEG` 即可。

## 姿态重构桥接

即使有了连续性约束，某些路径段仍然可能不可避免地发生支路切换或大幅腕部调整。
针对这种情况，当前版本额外加入了“姿态重构桥接”机制：

1. 先识别坏段：
   - 相邻两点的最大关节跳变超过阈值
   - 或者两点的配置标志发生变化
2. 在这两个原始点之间自动插入若干桥接点
3. 桥接点不是简单复制原始点，而是：
   - 先在关节空间做细分插值
   - 再围绕原始目标位姿附近做小范围位姿搜索
   - 优先选择“法兰位姿偏差更小、同时关节过渡更平滑”的解

这样做的目的不是完全消除姿态调整，而是把姿态调整摊开，并尽量让法兰位姿在调整过程中保持接近原始目标位姿。

当前相关参数也在 [main.py](/C:/Users/22290/Desktop/机械臂/仿真/winding_pose_solver/main.py) 顶部：

- `ENABLE_POSE_BRIDGE`
- `BRIDGE_TRIGGER_JOINT_DELTA_DEG`
- `BRIDGE_STEP_DEG`
- `BRIDGE_TRANSLATION_SEARCH_MM`
- `BRIDGE_ROTATION_SEARCH_DEG`

当前版本的桥接逻辑已经不是“每个桥接点局部贪心选一个 IK”。
现在的实际流程是：

1. 先识别需要做姿态重构的坏段。
2. 按桥接层为每一层收集多组候选关节解。
3. 候选解会同时围绕坏段前后两个原始法兰位姿，以及关节线性插值对应的 FK 位姿做局部搜索。
4. 再在整段桥接层上做一次局部动态规划，统一选择整段最优桥接序列。

桥接层优化的目标是：

- 让桥接期间的法兰位姿尽量贴近前后原始位姿，避免姿态调整时 TCP 明显漂移。
- 让相邻桥接点之间的法兰位姿变化也尽量小，把姿态重构摊开，而不是集中在一两个点突然翻转。
- 同时继续约束相邻桥接点之间的关节变化、配置切换、腕部翻转、近奇异和近限位风险。

当前版本还额外增加了一层“严格固定法兰位置”的尝试逻辑：

1. 对检测到的坏段，程序会先尝试生成“姿态变化时法兰位置固定”的原地 `MoveL` 桥接段。
2. 如果 RoboDK 判定这类桥接在线性运动上无解，程序会明确打印告警，并自动回退到原有的 best-effort 桥接模式。

这不是代码偷懒，而是 6 轴非冗余机器人在某些分支切换段上，本来就可能不存在严格的“固定法兰位置同时完成姿态重构”的可行线性路径。

默认含义：

- 当相邻点最大关节跳变超过 `20` 度时，触发桥接
- 桥接插值的目标步长为 `A1~A6 = (2, 2, 2, 20, 10, 20)` 度
- 每个桥接点会在原始位姿附近做小范围搜索：
  - 平移扰动：`0 / 1 / 2 mm`
  - 旋转扰动：`0 / 0.5 / 1.0 deg`

## 仓库结构

```text
winding_pose_solver/
├── data/
│   ├── validation_centerline.csv
│   ├── tool_poses_frame2.csv
│   └── tool_poses_frame2_optimized.csv
├── src/
│   ├── frame_math.py
│   ├── pose_solver.py
│   ├── robodk_program.py
│   └── visualization.py
├── main.py
└── README.md
```

## 主要模块说明

### `main.py`

统一维护默认路径、求解参数、RoboDK 参数，以及当前默认执行流程。

### `src/frame_math.py`

读取中心线 CSV，校验字段并构建局部坐标系。

### `src/pose_solver.py`

根据局部坐标系求解工具位姿，并输出 `data/tool_poses_frame2.csv`。

### `src/visualization.py`

提供中心线、边界和局部坐标系的可视化能力。

### `src/robodk_program.py`

读取位姿 CSV，在 RoboDK 中：

1. 收集多组 IK 候选
2. 执行全局路径优化
3. 应用单点硬约束和点间连续性硬约束
4. 创建目标点
5. 生成程序

## 默认配置

当前主要默认值集中在 `main.py` 顶部：

- 输入 CSV：`data/validation_centerline.csv`
- 输出位姿 CSV：`data/tool_poses_frame2.csv`
- 优化后 exact-pose CSV：`data/tool_poses_frame2_optimized.csv`
- 目标原点：`(1126.0, -400.0, 1200.0)`，位于 `Frame 2`
- 机器人名称：`KUKA`
- 参考坐标系名称：`Frame 2`
- 程序名称：`Path_From_CSV`
- 单点关节硬约束：`A1 ∈ [-150, 30]`，`A2 < 115`
- 点间连续性约束：`MAX_JOINT_STEP_DEG = (5, 5, 5, 180, 100, 180)`

## 输入输出约定

### 输入

默认输入文件是 `data/validation_centerline.csv`。

位姿求解至少需要以下列：

- `x, y, z`
- `tx, ty, tz`
- `nx, ny, nz`

可视化还需要边界列：

- `left_x, left_y, left_z`
- `right_x, right_y, right_z`

### 输出

位姿求解默认输出到 `data/tool_poses_frame2.csv`。
如果 RoboDK 选路阶段接受了全局或局部补偿，最终实际用于 exact-pose 选路和建程的结果会额外写到 `data/tool_poses_frame2_optimized.csv`。
正常情况下优先是 `Y/Z` 位置补偿；如果严格 `Y/Z only` 在当前站点下完全无解，程序会自动回退到围绕名义目标原点的刚体姿态搜索，因此优化 sidecar 也可能包含姿态旋转后的 exact pose。

无效行不会直接删除，而是保留 `valid=False`，对应位姿字段写入 `NaN`，方便后续排查。

## 维护建议

如果后续还需要恢复菜单式入口，建议保留当前“默认直接跑功能 2”的行为，只把其他功能作为显式参数入口。这样更符合目前项目的实际使用方式。
## 2026-03 当前新增的位姿 / 位形优化

这次版本除了原来的多解 IK + 全局 DP 之外，又额外加了两层“尽量从源头避免姿态重构”的优化：

1. 整路径统一 `Y/Z` 位置补偿搜索  
   程序不再默认做整条路径的 `XYZ` 姿态旋转搜索，而是只允许把整条法兰路径的目标原点在世界坐标系 `Y/Z` 方向做小范围平移。  
   工具坐标系三个轴方向保持不变，只微调 `目标工艺坐标系 A 在 Frame 2 中的原点位置`，从源头上尽量减少关节跳变和构型切换。

2. 位形走廊优先的 exact-pose 动态规划  
   在 exact-pose 候选层上，除了原有关节变化、近奇异、近限位惩罚之外，现在还会识别“能长距离保持同一位形族且相邻关节变化较小”的候选走廊，并对这些走廊给奖励。  
   目的不是让单点局部最优，而是尽量让整条路径更早进入稳定分支。

3. config-family 预规划骨架  
   代码里还增加了 `config_flags` 层面的预规划逻辑。后续如果继续做“在多解更丰富的层主动切族”，会以这层逻辑为基础扩展。

4. 解丰富窗口内的局部 `Y/Z` 位置重分配
   当全局 `Y/Z` 位置补偿后仍然存在残留坏段时，程序会在多解更丰富的窗口里，对后续整段路径施加一个平滑的 `Y/Z` 原点平移过渡，然后重新跑整条 exact-pose IK + 全局 DP。
   这一步同样不会改工具坐标系方向，只会平滑调整目标原点位置，并且整体偏移始终受 `Y/Z` 预算限制。
   目标是把 unavoidable 的位形切换摊到更宽的窗口里，并尽量放在解空间更丰富的位置发生，而不是把所有腕部翻转都挤在一个点上。

5. `Y/Z only` 无解时的姿态 fallback
   如果连全局 `Y/Z` 平移都无法让整条路径形成任何可行 IK 走廊，程序会自动回退到旧版的“整路径统一刚体姿态搜索 + 局部姿态重分配”。
   这一步只在严格 `Y/Z` 策略完全无解时触发，目标是保证当前 RoboDK 站点仍然能够生成可运行程序。

6. 固定位置桥接的腕部补偿搜索
   对仍然需要“法兰位置锁定”桥接的坏段，当前版本不再只依赖简单的单轴姿态扰动和默认 IK seed。  
   现在会额外加入：
   - `A4/A6` 相位补偿 seed
   - 更大范围的 wrist phase shift seed
   - 局部双轴组合姿态候选  
   这样做的目的，是尽量在固定位置前提下挖出更多腕部候选分解，让“先在驻留点开始调姿态，再继续走路径”更容易成功。

### 相关配置

这些开关都在 `app_settings.py` 里：

- `ENABLE_GLOBAL_TARGET_ORIGIN_YZ_SEARCH`
- `GLOBAL_TARGET_ORIGIN_YZ_MAX_OFFSET_MM`
- `GLOBAL_TARGET_ORIGIN_YZ_STEP_SCHEDULE_MM`
- `ENABLE_LOCAL_TARGET_ORIGIN_YZ_REDISTRIBUTION`
- `LOCAL_TARGET_ORIGIN_YZ_WINDOW_RADIUS`
- `LOCAL_TARGET_ORIGIN_YZ_OFFSET_MM`
- `ENABLE_ORIENTATION_FALLBACK_WHEN_YZ_INFEASIBLE`
- `ORIENTATION_FALLBACK_MAX_OFFSET_DEG`
- `ORIENTATION_FALLBACK_STEP_SCHEDULE_DEG`
- `ENABLE_LOCAL_ORIENTATION_FALLBACK_REDISTRIBUTION`
- `LOCAL_ORIENTATION_FALLBACK_WINDOW_RADIUS`
- `LOCAL_ORIENTATION_FALLBACK_OFFSET_DEG`

对应的 RoboDK 生成配置会传入 `RoboDKMotionSettings`：

- `enable_global_target_origin_yz_search`
- `global_target_origin_yz_max_offset_mm`
- `global_target_origin_yz_step_schedule_mm`
- `enable_local_target_origin_yz_redistribution`
- `local_target_origin_yz_window_radius`
- `local_target_origin_yz_offset_mm`
- `enable_orientation_fallback_when_yz_infeasible`
- `orientation_fallback_origin_mm`
- `orientation_fallback_max_offset_deg`
- `orientation_fallback_step_schedule_deg`
- `enable_local_orientation_fallback_redistribution`
- `local_orientation_fallback_window_radius`
- `local_orientation_fallback_offset_deg`

### 当前策略顺序

当前 RoboDK 程序生成的优先级顺序是：

1. 先做整路径统一 `Y/Z` 位置补偿搜索；
2. 在每个 exact pose 上收集多组 IK 候选；
3. 用全局 DP 选出整条最优关节序列；
4. 如果严格 `Y/Z only` 完全无解，则自动回退到整路径统一刚体姿态搜索；
5. 如果残留坏段仍然存在，优先尝试与当前全局策略对应的局部重分配；
6. 只有当 exact path 仍然留下明显坏段时，才进入桥接层做兜底。

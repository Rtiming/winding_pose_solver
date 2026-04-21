# winding-motion-module 接入指南

本文档面向 TypeScript 适配层（`winding-motion-module`）：

- 该模块是门面与编排层；
- 不能重写求解算法；
- 必须调用 `winding_pose_solver` 正式入口。

## 1. 推荐服务入口

推荐启动：

```powershell
python -m src.runtime.http_service --host 127.0.0.1 --port 8898
```

兼容启动（旧命令保留）：

```powershell
python scripts/model_demo_solver_api.py --host 127.0.0.1 --port 8898
```

默认端口：`8898`。  
若要改端口，由上层配置注入（例如 `WPS_API_PORT`）并统一在适配层 base URL 管理。

## 2. 推荐调用顺序

### 2.1 启动时

1. `GET /health`（或 `/api/health`）
2. 若 `configured=false`，调用 `POST /api/configure`
3. 记录返回的 `model_id`、`kinematics_source`、`kinematics_hash`，用于确认当前显示资产与求解模型一致。

### 2.2 交互/预览（单点）

1. `POST /api/fk`（可选）
2. `POST /api/ik`

说明：`ik` 仅做单步解算与候选选择，不是整条路径连续性求解。

### 2.4 模型显示与碰撞资产

- 显示层只加载资产和应用姿态，不实现 FK。
- 姿态应来自 `/api/fk`、`/api/ik` 或 `/api/path/solve` 返回结果。
- 若使用 RoboDK 导出的模型资产，优先校验 `kinematics_hash` 与资产清单中的 hash 是否一致。
- 后续碰撞检测模型选择应由 `winding_pose_solver` 暴露模型/资产清单，`winding-motion-module` 只选择模式和展示结果。

### 2.3 完整路径求解

1. 构造 `ProfileEvaluationRequest`
2. `POST /api/path/solve`（单请求）或 `/api/path/solve-batch`（批量）
3. 使用返回的 `result + quality` 做状态判断与 UI 展示

## 3. TypeScript 适配层最小字段

路径请求最小字段：

- `request_id`
- `robot_name`
- `frame_name`
- `motion_settings.ik_backend`（推荐 `six_axis_ik`）
- `reference_pose_rows`
- `frame_a_origin_yz_profile_mm`
- `row_labels`
- `inserted_flags`
- `create_program`（HTTP 场景建议 `false`）

上层播放状态机建议最少维护：

- `request_id`
- `status`
- `semantic_status`
- `strictly_valid`
- `gate_tier`
- `block_reasons`

## 4. 诊断字段与 UI 展示建议

可以直接展示给 UI：

- `ik_empty_row_count`
- `config_switches`
- `bridge_like_segments`
- `big_circle_step_count`
- `worst_joint_step_deg`
- `mean_joint_step_deg`
- `gate_tier`
- `block_reasons`

不要作为硬约束直接阻断（默认语义）：

- `config_switches`（诊断信号，不是单独硬门禁）

应作为硬失败或阻断依据：

- `invalid_row_count > 0`
- `ik_empty_row_count > 0`
- `selected_path` 缺失/不完整
- `bridge_like_segments > 0`
- `big_circle_step_count > 0`
- `worst_joint_step_deg` 超阈值
- `block_reasons` 非空且 `gate_tier != official`

## 5. 异常/失败处理建议

接口错误（HTTP 非 2xx 或 `ok=false`）：

- 进入失败态，记录 `error.code/message/details`
- 可用 `error_legacy` 兼容旧日志/旧提示

业务不可达：

- `semantic_status=invalid` 或 `strictly_valid=false`
- 用 `block_reasons` 精确提示原因
- 保留诊断结果用于重试策略（例如调整起点、参数或批量候选）

路径不连续/质量门禁失败：

- 不要在适配层自行做修复算法
- 应重新构造请求并交由 `winding_pose_solver` 的 `search/runtime` 能力处理

## 6. 严格边界（适配层禁止实现）

`winding-motion-module` 禁止实现以下逻辑：

- 新 IK/FK 算法
- IK 候选搜索与 DP 选路
- 连续性优化与 bridge 修复
- Frame-A Y/Z repair 与 sweep
- RoboDK 导入/程序生成求解逻辑

以上能力必须由 `winding_pose_solver` 提供并维护。

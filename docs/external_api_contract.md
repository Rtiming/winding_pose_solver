# winding_pose_solver External API Contract

本文档定义 `winding_pose_solver` 对外稳定调用契约。  
核心原则：**本仓库是唯一求解内核**，外部项目不得重写 IK/FK、候选搜索、路径连续性优化、Frame-A Y/Z 修复、RoboDK 导入逻辑。

## 1. 正式入口与边界

正式入口：

- HTTP（推荐）：`python -m src.runtime.http_service --host 127.0.0.1 --port 8898`
- Python API：`src/runtime/external_api.py`

兼容入口：

- `python scripts/model_demo_solver_api.py --host 127.0.0.1 --port 8898`  
  该脚本仅为薄包装，内部启动 `src.runtime.http_service`。

非正式入口（内部/诊断）：

- `src/search/*`、`src/core/*`、`src/six_axis_ik/*` 内部函数
- `scripts/diagnose_*.py`、`scripts/validate_*.py` 等诊断脚本

外部项目不应直接依赖上述内部模块和诊断脚本作为长期 API。

## 2. HTTP API

默认返回 JSON。除健康检查外，失败格式统一如下：

```json
{
  "ok": false,
  "error": {
    "code": "path_solve_failed",
    "message": "RuntimeError: ...",
    "details": {}
  },
  "error_legacy": "RuntimeError: ..."
}
```

- `error`：结构化错误（新契约）
- `error_legacy`：兼容旧调用方的顶层字符串

### 2.1 `GET /health` 与 `GET /api/health`

用途：健康检查与会话配置状态。

返回：

```json
{
  "ok": true,
  "configured": false,
  "kinematics_source": "unconfigured",
  "kinematics_hash": null,
  "model_id": null
}
```

### 2.2 `POST /api/configure`

用途：配置 IK/FK 会话（机器人模型、Tool、Frame、基坐标、关节限位）。

请求（最小）：

```json
{
  "robot": {}
}
```

说明：

- 未提供 `kinematics_inferred` 时回退到 `src.six_axis_ik.config` 默认模型。
- 已提供 `kinematics_inferred` 但字段或维度不合法时返回 `configure_failed`，不会静默回退默认模型。
- `strict_kinematics=true` 时必须提供合法 `kinematics_inferred`。
- 返回 `kinematics_hash` 与 `model_id`，用于外部显示资产和求解模型的一致性检查。
- 未提供关节限位时回退默认关节限位。

### 2.3 `POST /api/fk`

用途：单点 FK。

请求：

```json
{
  "q_deg": [0, 0, 0, 0, 0, 0]
}
```

返回关键字段：

- `tcp_frame_pose`：TCP 在参考坐标系下的 4x4
- `tcp_world_pose`：TCP 在世界坐标系下的 4x4
- `joint_frames_world`：各关节层级位姿

### 2.4 `POST /api/ik`

用途：单点 IK（交互式单步）。

请求关键字段：

- `target_frame_pose`：目标 4x4
- `seed_q_deg`（可选）
- `previous_q_deg`（可选）
- `include_orientation`（默认 `true`）

返回关键字段：

- `q_deg`
- `selected_by`（`ik_all_continuity` / `ik_single_fallback` / `numeric_refine` 等）
- `candidate_count`、`shortlisted_count`
- `joint_distance_to_previous_deg`
- `discontinuous_warning`
- `position_error_mm`、`orientation_error_deg`

重要语义：

- 单点 IK 内的 `previous_q_deg` 仅用于**单步候选选择**，不代表完整路径连续性求解。
- 完整路径连续性、候选选路与修复必须走 `/api/path/solve` 或 `/api/path/solve-batch`。

### 2.5 `POST /api/path/solve`

用途：完整路径求解（单请求）。复用 `evaluate_single_request`。

请求体可直接传 `ProfileEvaluationRequest`，或包一层：

```json
{
  "request": {
    "request_id": "demo_001",
    "robot_name": "KUKA",
    "frame_name": "Frame 2",
    "motion_settings": {
      "ik_backend": "six_axis_ik"
    },
    "reference_pose_rows": [],
    "frame_a_origin_yz_profile_mm": [],
    "row_labels": [],
    "inserted_flags": [],
    "create_program": false
  }
}
```

返回关键字段：

- `result`：`ProfileEvaluationResult` 原始结构
- `quality`：`result_quality_summary` 摘要
- `strictly_valid`
- `continuity_warnings`
- `semantic_status`

### 2.6 `POST /api/path/solve-batch`

用途：批量路径求解。复用 `evaluate_batch_request`。

支持三种输入：

- 标准批量：`{ "evaluations": [ ... ] }`
- RemoteSearchRequest：`{ "base_request": ... }`
- 单请求：`{ "request": ... }` 或直接 `ProfileEvaluationRequest`

返回关键字段：

- `result`：`EvaluationBatchResult`
- `quality_summaries`：每个结果的质量摘要
- `best_request_id`
- `best_quality`

### 2.7 Reserved RoboDK-Parity Runtime Endpoints

These endpoints reserve the future contract for RoboDK-class operation. They
must stay solver-owned. Adapter packages may call them, but must not implement
fallback behavior locally.

- `GET /api/capabilities`: returns implemented and reserved runtime
  capabilities.
- `POST /api/simulation/control`: reserved for play/pause/stop/seek/step,
  run-from-instruction, speed-ratio changes, and future online/robot execution
  control.
- `POST /api/program/plan`: reserved for building a controller-neutral program
  instruction plan from selected solver results.
- `POST /api/program/export`: reserved for RoboDK station and KUKA C5 ecosystem
  export targets.

Current behavior:

- `GET /api/capabilities` is implemented.
- The reserved POST endpoints return structured `501` errors until their
  owning runtime layer is implemented.

KUKA C5 direction:

- First concrete export target: `kuka_krl_src_dat` for KR C5 / KSS 8.7 style
  KRL output.
- Reserved planning targets: `kuka_mxautomation_plan` and `kuka_srci_plan`.
- KUKA export must consume selected paths and program IR; it must not choose IK
  candidates or repair continuity.

## 3. 质量字段语义（硬门禁 vs 诊断）

门禁定义来自 `src/runtime/delivery.py`（`STRICT_DELIVERY_GATE`）。

官方交付（`official_delivery_allowed=true`）需要同时满足：

- `invalid_row_count == 0`
- `ik_empty_row_count == 0`
- `selected_path` 完整非空
- 闭合绕线时 terminal hard rule 通过
- `bridge_like_segments == 0`
- `big_circle_step_count == 0`
- `worst_joint_step_deg <= official_worst_joint_step_deg_limit`

诊断字段：

- `config_switches`：诊断信号，默认不是单独硬阻断
- `bridge_like_segments` / `big_circle_step_count` / `worst_joint_step_deg`：参与官方门禁
- `gate_tier`：`official` / `debug` / `diagnostic`
- `block_reasons`：明确阻断原因代码与消息

## 4. online coordinator/server/receiver 场景

- `server`：纯计算（six_axis_ik，含候选、选路、重试修复），不依赖 RoboDK。
- `receiver`：本地 RoboDK 最终导入/程序落地。
- `coordinator`：Windows 侧编排 server + receiver。

HTTP 服务适合上层系统在线调用求解能力；  
online role 流程适合跨机、Slurm、RoboDK 终态交付链路。

## 5. Python API（`src/runtime/external_api.py`）

稳定函数：

- `configure_robot(payload, session=None)`
- `solve_fk(payload, session=None)`
- `solve_ik(payload, session=None)`
- `solve_path_request(payload)`
- `solve_path_batch(payload)`

示例：

```python
from src.runtime.external_api import (
    configure_robot,
    solve_fk,
    solve_ik,
    solve_path_request,
)

configure_robot({"robot": {}})
fk = solve_fk({"q_deg": [0, 0, 0, 0, 0, 0]})
ik = solve_ik({"target_frame_pose": fk["tcp_frame_pose"]})

# 路径请求按 ProfileEvaluationRequest 结构传入
result = solve_path_request({...})
```

# unitree_rl_lab_egg

> IsaacLab 强化学习 + 纯 MuJoCo 传统步态控制，一个面向 **Unitree Go2** 的学习型仓库。
> A learning-oriented fork of [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) with Go2 URDF support, a standalone classical gait playground, and full Chinese documentation.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-%E2%89%A53.1-blue)](https://mujoco.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](./LICENCE)

---

## 这个仓库有什么

| 模块 | 内容 | 入口 |
|---|---|---|
| **强化学习训练** | 基于 IsaacLab + rsl-rl PPO 的四足速度跟踪环境（Go2 / H1 / G1 等） | `source/unitree_rl_lab/`、`scripts/rsl_rl/` |
| **传统步态控制** | 不依赖 Isaac/ROS2，纯 `python + mujoco`：站立 → IK → 摆线 → trot → Raibert → 平衡转向 | [`classical_gait/`](./classical_gait/) |
| **部署运行时** | C++ FSM + ONNX Runtime + unitree_sdk2（DDS），可跑 sim2sim 与真机 | `deploy/` |
| **中文文档** | 安装跑通、训练指标判读、传统控制原理与整定手册 | [`docs/`](./docs/) |

---

## 文档导航

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [`docs/GO2_GUIDE.md`](./docs/GO2_GUIDE.md) | 环境安装、模型文件、训练/回放/导出 ONNX、sim2sim、sim2real 全流程 | 想跑通 RL 训练的人 |
| [`docs/GO2_TRAINING_METRICS.md`](./docs/GO2_TRAINING_METRICS.md) | 任务是什么、每个指标的含义、曲线判读、异常排查、下一步学习路线 | 训练中看不懂曲线的人 |
| [`docs/GO2_CLASSICAL_GAIT.md`](./docs/GO2_CLASSICAL_GAIT.md) | 单腿 FK/IK 推导、摆线轨迹、trot 相位、Raibert 落脚点、姿态/偏航反馈 | 想打控制基础的人 |
| [`docs/UPSTREAM_README.md`](./docs/UPSTREAM_README.md) | 上游仓库原始 README（存档） | 查阅上游信息 |

**建议阅读顺序**：`GO2_CLASSICAL_GAIT.md` §0 → §0.5 → 跑 `classical_gait/` → 再回头看 §3 数学 → 想训 RL 时读 `GO2_TRAINING_METRICS.md` → 部署前读 `GO2_GUIDE.md` Step 6/7。

---

## 快速开始

### 0. 前置条件

- NVIDIA GPU + 驱动（Isaac Sim 5.1 要求）
- CUDA 环境下的 Isaac Lab 2.3（安装见 [官方文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)）
- conda 环境（示例：`env_isaaclab`）

### 1. 获取代码并安装

```bash
git clone https://github.com/chenbinzhiegg/unitree_rl_lab_egg.git ~/unitree_rl_lab_egg
cd ~/unitree_rl_lab_egg
conda activate env_isaaclab
./unitree_rl_lab.sh -i          # 可编辑模式安装 + 写入 conda 环境变量
# 重新打开 shell 使环境变量生效
```

> 上面把仓库克隆到 `~/unitree_rl_lab_egg`；你当然可以放在任意目录（克隆时省略第二个参数即使用仓库名）。
> 下文命令里的路径请按你的实际目录调整。

### 2. 准备机器人描述文件（二选一）

**方式 A：URDF（推荐，Isaac Sim ≥ 5.0）**

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git ~/unitree_ros
```

**方式 B：USD**

```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model ~/unitree_model
```

路径默认读 `~/unitree_ros` / `~/unitree_model`，也可用环境变量覆盖：

```bash
export UNITREE_ROS_DIR=/path/to/unitree_ros
export UNITREE_MODEL_DIR=/path/to/unitree_model
```

> Go2 默认已配置为 URDF（见 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`）。
> 切回 USD 只需把 `UNITREE_GO2_CFG` 里的 `spawn` 换回 `UnitreeUsdFileCfg`。

### 3. 验证与训练

```bash
./unitree_rl_lab.sh -l | grep -i go2        # 确认任务已注册
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --num_envs 1024
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity --num_envs 32   # 回放 + 导出 ONNX
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity
```

> 8 GB 显存建议 `--num_envs 512~1024`（默认 4096 会爆显存）。

### 4. 传统步态（完全不需要 Isaac）

```bash
cd classical_gait
pip install -r requirements.txt
python 01_stand.py
python 02_leg_ik.py
python 03_swing_trajectory.py
python 04_trot_openloop.py
python 05_raibert_walk.py
python 06_balance_turn.py --vx 0.3 --wz 0.4
```

需要图形界面时加 `--viewer --realtime`。详见 [`classical_gait/README.md`](./classical_gait/README.md)。

### 5. 部署（可选）

```bash
cd deploy/robots/go2
mkdir -p build && cd build && cmake .. && make -j
./go2_ctrl            # 真机：./go2_ctrl --network eth0
```

依赖 `unitree_sdk2`、`yaml-cpp`、`boost`、`eigen`、`spdlog`、`fmt`，详见 `docs/GO2_GUIDE.md`。

---

## 目录结构

```
unitree_rl_lab_egg/
├── README.md
├── LICENCE
├── unitree_rl_lab.sh              # 统一入口：-i 安装 / -l 列任务 / -t 训练 / -p 回放
├── docs/                          # 中文文档（本仓库新增）
├── classical_gait/                # 纯 MuJoCo 传统步态控制（本仓库新增）
├── source/unitree_rl_lab/         # IsaacLab 环境、机器人配置、MDP
├── scripts/                       # 训练 / 回放 / 数据转换脚本
├── deploy/                        # C++ 部署运行时（FSM + ONNX + DDS）
├── docker/  doc/                  # 容器与文档素材
├── logs/  outputs/                # 本地运行产物（已 gitignore）
└── thirdparty/                    # ONNX Runtime（deploy 使用）
```

---

## 相对上游的改动

1. **Go2 默认使用 URDF**（`go2_description.urdf`），无需下载体积巨大的 USD 数据集；
2. **模型路径支持环境变量** `UNITREE_ROS_DIR` / `UNITREE_MODEL_DIR`，默认 `~/unitree_ros`、`~/unitree_model`；
3. **新增 `classical_gait/`**：6 个渐进式脚本 + FK/IK/步态控制器，可直接跑通站立到转向；
4. **新增 `docs/`**：3 份中文文档（RL 全流程、训练指标判读、传统步态原理）；
5. `-i` 安装时，`activate-global-python-argcomplete` 缺失不再中断安装。

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `No conda environment activated` | 先 `conda activate env_isaaclab`，脚本依赖 `CONDA_PREFIX` |
| 训练 OOM / 卡死 | 降低 `--num_envs`（4096 默认值不适配 8 GB 显存） |
| URDF 转 USD 时出现 `Head_* visuals Unresolved reference` | 仅显示层警告，可忽略，物理与训练不受影响 |
| 曲线怎么读、什么时候该停 | 见 [`docs/GO2_TRAINING_METRICS.md`](./docs/GO2_TRAINING_METRICS.md) |
| 想先学传统控制 | 见 [`docs/GO2_CLASSICAL_GAIT.md`](./docs/GO2_CLASSICAL_GAIT.md) |

---

## 致谢

本项目基于以下开源工作：

- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)（本仓库的上游）
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：训练与仿真基础
- [rsl-rl](https://github.com/leggedrobotics/rsl_rl)：PPO 实现
- [mujoco](https://github.com/google-deepmind/mujoco) 与 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)：传统步态与 sim2sim
- [unitree_ros](https://github.com/unitreerobotics/unitree_ros)：URDF 模型描述
- [robot_lab](https://github.com/fan-ziqi/robot_lab)、[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)：工程结构参考

## License

[Apache-2.0](./LICENCE)。本仓库是上游 `unitree_rl_lab` 的衍生作品，原始版权归 Unitree Robotics 与 The Isaac Lab Project Developers 所有；新增部分同样以 Apache-2.0 发布。

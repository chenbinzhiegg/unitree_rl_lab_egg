# classical_gait —— 纯 MuJoCo 的 Go2 传统步态控制

不依赖 Isaac Sim / ROS2 / 强化学习，只用 `python + mujoco` 把四足控制的基本功走一遍：

```
站立 → 单腿逆运动学 → 摆动轨迹 → 开环 trot → 速度跟踪 → 平衡与转向
```

完整的原理推导、参数整定手册与实测数据见 **[../docs/GO2_CLASSICAL_GAIT.md](../docs/GO2_CLASSICAL_GAIT.md)**。

## 依赖

```bash
pip install -r requirements.txt     # mujoco>=3.1, numpy
```

模型来自 `unitree_mujoco` 的官方 MJCF，默认路径：

```
~/unitree_mujoco/unitree_robots/go2/scene.xml
```

如果不在这个位置，复制 `go2.xml` + `assets/` 到 `~/unitree_mujoco/unitree_robots/go2/`，或修改
`go2_model.py` 里的 `DEFAULT_SCENE`。

## 运行

```bash
python 01_stand.py                  # PD 站立（验收：高度 0.28 m，不摔）
python 02_leg_ik.py                 # FK/IK 数值验证（验收：误差 < 1e-6 rad / 1 mm）
python 03_swing_trajectory.py       # 吊住机身，单腿摆动（验收：抬腿 ≈ 6 cm）
python 04_trot_openloop.py          # 开环 trot（验收：6 s 不摔、能前进）
python 05_raibert_walk.py           # Raibert 速度跟踪（验收：误差 < 0.15 m/s）
python 06_balance_turn.py --wz 0.4  # 姿态反馈 + 转向（验收：倾角 < 0.15 rad）
```

常用参数：

```bash
python 06_balance_turn.py --vx 0.3 --wz 0.4 --viewer --realtime
python 04_trot_openloop.py --vx 0.2 --period 0.6 --duty 0.65
```

`--viewer` 需要图形界面；无显示器时直接运行即可（纯计算，500 Hz 仿真很快）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `go2_model.py` | 模型常量、正/逆运动学、PD 力矩控制、状态读取、摔倒判定 |
| `gait.py` | trot 相位调度、摆线摆动轨迹、Raibert 落脚点、姿态与偏航反馈 |
| `01_stand.py` … `06_balance_turn.py` | 六个渐进阶段，每个都可独立运行并打印验收结果 |

## 本机参考结果

| 阶段 | 结果 |
|---|---|
| 站立 | 高度 0.2787 m（理论 0.2868，PD 静态下垂 8 mm），倾角 0.003 rad |
| IK | 往返误差 1.3e-15 rad，笛卡尔跟踪 0.0000 mm |
| 单腿摆动 | 抬腿 5.8 cm / 指令 6.0 cm |
| 开环 trot | 6 s 前进 1.162 m，0.202 m/s（指令 0.3） |
| Raibert | 0.318 m/s（指令 0.4） |
| 平衡 + 转向 | 倾角 0.053 rad，偏航 0.306 rad/s（指令 0.4） |

> 结果与机器性能无关（纯 CPU 计算），但 PD 增益与模型版本相关，换机器人/模型后需重新整定。

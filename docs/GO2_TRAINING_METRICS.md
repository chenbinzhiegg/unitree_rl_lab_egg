# Go2 训练补充文档：任务说明与效果判读

> 配套文档：[GO2_GUIDE.md](./GO2_GUIDE.md)（安装、跑通、部署）
> 本文档只回答三个问题：**这个任务在学什么？怎么看它学得好不好？训练结束后数据还在吗？**
> 想先学传统控制：见 [GO2_CLASSICAL_GAIT.md](./GO2_CLASSICAL_GAIT.md)

---

## 0. 零基础先读：3 分钟建立框架

RL 训练里所有东西都可以归到 **MDP 五要素**，本仓库的实现就在 `velocity_env_cfg.py` 里面对应五个 `@configclass`：

| MDP 要素 | 含义 | Go2 里是什么 | 代码位置 |
|---|---|---|---|
| **状态 State** | 机器人当前“看到/知道”什么 | 45 维观测（角速度、重力方向、指令、关节角/速、上帧动作） | `ObservationsCfg` |
| **动作 Action** | 策略能改什么 | 12 个关节目标角（不是力矩） | `ActionsCfg` |
| **奖励 Reward** | 什么算做得好 | 跟踪速度 + 平滑/省电/安全惩罚 | `RewardsCfg` |
| **转移 Transition** | 动作之后世界怎么变 | Isaac Sim (PhysX) 物理仿真 | `RobotSceneCfg` + `sim.dt` |
| **终止 Termination** | 什么时候结束一轮 | 超时 / 躯干碰地 / 翻车 | `TerminationsCfg` |

再记住三句话：

1. **训练 = 反复试错 + 把“奖励高的动作”概率调大**（PPO）；
2. **曲线看的是统计量，不是单次表现**：4096 个机器人同时跑，打印的是平均值；
3. **收敛 ≠ 能用**：策略还得经过回放（看行为）→ sim2sim（看差异）→ 真机（看安全）。

> 小贴士：`--num_envs` 是“同时训练多少个机器人”，只影响样本量与显存，**不是** batch size。

---

## 1. `Unitree-Go2-Velocity` 是什么任务

四足机器人最经典的入门任务：**速度跟踪（velocity tracking）**。

### 1.1 目标

让 Go2 在仿真里学会**按任意给定速度指令平稳行走**。

| 要素 | 内容 |
|---|---|
| 指令 | 每个 episode 内固定的 $(v_x, v_y, \omega_z)$，每 10 s 重新采样 |
| 策略输入 | 45 维观测：机体角速度(3) + 重力投影(3) + 速度指令(3) + 关节角(12) + 关节速度(12) + 上一帧动作(12) |
| 策略输出 | 12 个关节的目标角度（`scale=0.25`，叠加在默认站姿上） |
| 控制频率 | 50 Hz（`sim.dt=0.005` × `decimation=4`） |
| episode 长度 | 20 s = 1000 个策略步 |
| 终止条件 | 超时 / 躯干接触地面 / 姿态倾斜 > 0.8 rad |

### 1.2 奖励构成（决定它“学成什么样”）

| 类别 | 项 | 权重 | 作用 |
|---|---|---|---|
| **任务奖励** | `track_lin_vel_xy` | +1.5 | 线速度跟得越准越高（指数核） |
| | `track_ang_vel_z` | +0.75 | 角速度跟得越准越高 |
| 平滑/省力 | `joint_vel` / `joint_acc` / `joint_torques` / `energy` | 负 | 抑制抖动、猛打关节、耗电 |
| 姿态 | `flat_orientation_l2` / `base_linear_velocity` / `base_angular_velocity` | 负 | 保持躯干水平、不上下窜 |
| 步态 | `feet_air_time` / `air_time_variance` / `feet_slide` | ± | 鼓励抬腿、步态匀称、脚不打滑 |
| 安全 | `undesired_contacts` / `dof_pos_limits` | 负 | 头部/髋/大腿/小腿不许磕地，关节不许超限 |

> 一句话：**“跟上指令”是主奖励，其余都是让动作好看、省电、安全的正则项。**

### 1.3 难度是怎么加上去的

- **课程学习 1**：指令范围从 ±0.1 m/s 起步，随训练逐步放大到 ±1.0 m/s（曲线名 `Curriculum/lin_vel_cmd_levels`）。
- **课程学习 2**：地形难度 `Curriculum/terrain_levels`（当前只开了平地，所以基本不涨）。
- **域随机化**（`EventCfg`）：摩擦/弹性 0.3~1.2、base 质量 +(-1, 3) kg、每 5~10 s 随机推一把 —— 这些**不会让训练变容易，但决定能不能上真机**。

---

## 2. 怎么看训练效果（四个层次）

### 层次 1 · 终端实时打印（最快，判断“有没有在学”）

每次 iteration 结束会打印一组指标。名称与含义：

| 指标（已在本机实际输出中确认） | 含义 | 期望趋势 |
|---|---|---|
| `Episode_Reward/track_lin_vel_xy` | 线速度跟踪奖励 | ↑ 后收敛 |
| `Episode_Reward/track_ang_vel_z` | 角速度跟踪奖励 | ↑ 后收敛 |
| `Episode_Reward/feet_air_time` | 抬腿奖励 | 从 0 升到稳定值 |
| `Episode_Reward/joint_vel` 等 | 惩罚项（负值） | 绝对值变大（说明开始动了），随后趋于稳定 |
| `Metrics/base_velocity/error_vel_xy` | 实际 vs 指令的线速度误差 | ↓（< 0.3 m/s 算不错） |
| `Metrics/base_velocity/error_vel_yaw` | 角速度误差 | ↓ |
| `Episode_Termination/base_contact` | 躯干着地比例 | ↓ 接近 0 |
| `Episode_Termination/bad_orientation` | 翻车比例 | ↓ 接近 0 |
| `Episode_Termination/time_out` | 正常跑满 20 s 的比例 | ↑ 接近 1 |
| `Curriculum/lin_vel_cmd_levels` | 指令难度等级 | ↑（说明课程在推进） |
| `Curriculum/terrain_levels` | 地形难度等级 | 平地任务停在 1 附近 |

### 层次 2 · TensorBoard 曲线（主要手段）

```bash
cd ~/unitree_rl_lab_egg
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity --port 6006
# 浏览器打开 http://localhost:6006
```

除了上面的环境指标，rsl-rl 还会写入算法侧标量（以下名称来自本机安装的 rsl-rl 源码）：

| 指标 | 含义 | 备注 |
|---|---|---|
| `Train/mean_reward` | 每轮平均回报 | 整体上升即“在学好” |
| `Train/mean_episode_length` | 平均存活步数 | 上限 1000（20 s / 0.02 s），越接近越好 |
| `Policy/mean_noise_std` | 探索噪声标准差 | 通常缓慢下降 |
| `Loss/learning_rate` | 自适应 KL 调出来的学习率 | 剧烈跳动说明 batch 太小或奖励尺度太猛 |
| `Perf/total_fps` | 每秒仿真步数 | 用来估算总训练时间 |
| `Perf/learning_time` | 每次迭代的训练耗时 | 与 `Perf/total_fps` 对照看瓶颈在仿真还是网络 |

#### 判读标准（经验值）

| 训练阶段 | 应看到什么 | 若不符合说明什么 |
|---|---|---|
| 0~50 iter | 各项都在乱跳，`mean_reward` 很低甚至为 0 | 正常，属于“热身”，**不要在此判断成败** |
| 50~300 iter | 机器人学会站起来/迈步，`feet_air_time` 从 0 变为非零，`base_contact` 明显下降 | 一直站着不动 → 检查奖励权重、动作 scale、初始站姿 |
| 300~3000 iter | `track_lin_vel_xy` 陡升后走平，`error_vel_xy` 平滑下降，`time_out` 上升 | 奖励不涨 / 上下震荡 → 见第 4 节排查表 |
| 收敛后 | 曲线平台期，回放时能稳定走、转向、侧移 | 指标好但画圈/歪走 → 奖励设计偏向，需回到第 3 节 |

#### 多组实验对比

- 每次训练自动建新时间戳目录，TensorBoard 会把所有 run **并排画在同一张图**；
- 建议加 `--run_name` 给实验命名，便于区分：

```bash
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless \
    --num_envs 1024 --run_name reward_v2
```

目录名会变成 `2026-10-22_22-00-12_reward_v2`。

#### 导出数据

TensorBoard 右上角可 **Download as CSV**，方便用 pandas/Excel 做对比；也可以只筛某个 tag：

```bash
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity --port 6006 \
            --samples_per_plugin scalars=0
```

### 层次 3 · 回放（最直观，必须做）

```bash
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity --num_envs 32
```

观察清单：

- [ ] 能原地站稳、不抖
- [ ] 直走不画圈、不侧滑
- [ ] 转向跟指令方向一致
- [ ] 停下时能保持平衡
- [ ] 被推一下能恢复（域随机化生效）

> 曲线漂亮但回放画圈，通常意味着某个惩罚项压过了跟踪奖励，或者观测里有信息缺失。

### 层次 4 · sim2sim / 真机（最终标准）

仿真里表现好只是必要条件。sim2sim（MuJoCo）会暴露 PD 参数、观测顺序、延迟等问题，流程见 `GO2_GUIDE.md` 第 4 节 Step 6/7。

---

## 3. 训练结束后，曲线还在吗？

**在。** TensorBoard 不是把数据放在内存里，而是实时写入日志目录：

```
logs/rsl_rl/unitree_go2_velocity/2026-10-22_22-00-12/
├── events.out.tfevents.1792677620.Legion.1208580.0   ← 曲线数据（可随时重新读取）
├── model_0.pt                                        ← checkpoint
├── model_19.pt                                       ← 最后一次（按 save_interval=100 保存）
├── git/                                              ← 代码版本快照
└── params/
    ├── env.yaml            # 环境配置
    ├── agent.yaml          # PPO 超参
    ├── deploy.yaml         # 部署用参数
    └── velocity_env_cfg.py # 配置源码快照
```

结论与注意事项：

1. **随时可看**：训练进程退出、电脑重启后，再跑一次 `tensorboard --logdir ...` 就能重新看到全部曲线（已在本机验证存在 `events.out.tfevents.*` 文件）。
2. **不会互相覆盖**：每次训练都是新的时间戳目录。
3. **删了才没有**：只有删除 run 目录才会丢曲线，`logs/` 目录请勿随意清理。
4. **看不到曲线的两种情况**：`--logdir` 指错层级（应指向任务级目录，而不是某个时间戳目录）、或 TensorBoard 端口被占用（换 `--port`）。
5. **可以随时续训**：`--resume --load_run <run名>` 会在原目录继续写曲线，事后能看出“接续点”。

---

## 4. 曲线异常排查表

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `mean_reward` 一直不涨 | 奖励尺度太小 / 课程起点太容易 / 学习率过低 | 先把 `--num_envs` 调大保证样本量；再试提高跟踪奖励权重 |
| 奖励陡升后突然崩 | 学习率过大、KL 自适应失控 | 看 `Loss/learning_rate`；降低 `learning_rate` 或 `desired_kl` |
| `error_vel_xy` 掉到 0.1 以下但回放画圈 | 跟踪奖励过大、缺少朝向/侧移约束 | 提高 `flat_orientation_l2` 权重或检查指令采样 |
| `base_contact` 始终很高 | 初始站姿太高/太低、PD 太软、episode 一开始就摔 | 检查 `UNITREE_GO2_CFG.init_state.pos=(0,0,0.4)` 与 actuator 的 kp/kd |
| `mean_episode_length` 上不去 | 终止条件太严（0.8 rad 姿态限制） | 初期可放松 `bad_orientation` 的 `limit_angle` |
| 曲线剧烈震荡 | `num_envs` 太少 / `num_steps_per_env` 太小 | 增加并行环境数，或增大 `num_steps_per_env` |
| 训练很慢（`total_fps` 低） | 环境数太少、开了渲染、`--headless` 未生效、地形网格太重 | 加 `--num_envs`；确保无 GUI；简化地形 |
| TensorBoard 空白 | logdir 层级不对 / 事件文件未生成 | 指向 `logs/rsl_rl/unitree_go2_velocity`，确认目录内有 `events.out.tfevents.*` |

---

## 5. 在 8 GB 显存上的训练预算（估算方法）

不要背固定数字，用两条公式自己算：

```
每 iteration 样本数 = num_envs × num_steps_per_env(24)
达到目标样本量的迭代数 ≈ 目标总样本量 / 每 iteration 样本数
总耗时 ≈ 迭代数 × Iteration time（终端里每轮都打印）
```

经验参考：

- 同类四足速度任务通常需要 **1 亿 ~ 2 亿环境步**才收敛；
- `num_envs=1024` 时每 iteration 约 2.4 万步，即需要 **4000~8000 iterations**；
- 若单轮耗时 2 s，则约 **3~5 小时**；
- 默认 `max_iterations=50000` 远远超出所需，实际用不到（想省时可以显式传小一点，或中途 Ctrl+C 后拿最近的 checkpoint）。

> 判断“可以停了”的依据是曲线平台化 + 回放验收，而不是跑满迭代数。

---

## 6. 一分钟自检流程

```bash
# 1. 训练（观察终端指标是否在变好）
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless --num_envs 1024

# 2. 另开一个终端看曲线
#   （训练结束后这一条依然可以用，数据不会消失）
tensorboard --logdir logs/rsl_rl/unitree_go2_velocity --port 6006

# 3. 回放验收
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity --num_envs 32

# 4. 导出 ONNX 后进入 sim2sim
ls logs/rsl_rl/unitree_go2_velocity/*/exported/
```

---

## 7. 新手 FAQ

**Q：训练多久算够？怎么判断可以停？**
A：不看时间，看两条：① `track_lin_vel_xy` 与 `Track/*` 停止上升、`error_vel_xy` 平稳；
② `-p` 回放能稳定走。参考量级：1 亿~2 亿环境步。

**Q：为什么我的指标一开始全是 0？**
A：前几十轮是随机探索期，机器人还没学会站立；同时课程学习从 $\pm0.1$ m/s 的极慢指令起步。属正常。

**Q：`--num_envs` 要设多少？**
A：本机 8 GB 显卡建议 512~1024；太少（<128）反而不稳定且效率低，4096 会爆显存。

**Q：训练中断了怎么办？**
A：`--resume --load_run <run目录名>` 接着训；曲线会追加到同一目录。

**Q：能不能直接用别人训好的模型？**
A：可以，但要注意观测/动作定义必须与你的环境一致，否则 sim2sim 会对不上。

**Q：为什么 sim2sim 表现比 Isaac 里差？**
A：摩擦/质量/延迟/关节阻尼差异。解决办法是加强域随机化重训，而不是改 PD 去“抬”仿真。

**Q：我该先学 RL 还是先学传统控制？**
A：想打基础、看懂“为什么能走”，先看 [GO2_CLASSICAL_GAIT.md](./GO2_CLASSICAL_GAIT.md)；
想快速拿到能跑的策略，直接照着本文档训。两条路最后都会汇到同一个部署接口上。

---

## 8. 下一步学习方向

### 8.1 先把这一套用熟（1~2 周）

1. **奖励设计实验**：按 [GO2_GUIDE.md](./GO2_GUIDE.md) 第 6 节的三个实验做消融，
   每个只改一个变量，用 TensorBoard 同图对比；
2. **域随机化实验**：把摩擦范围拉大（如 0.2~1.5），看 `error_vel_xy` 变差多少、
   回放是否更稳；这直接决定 sim2real 成败；
3. **课程学习**：把 `lin_vel_cmd_levels` 的起始值改小/改大，观察收敛速度；
4. **观测加项**：给 critic 打开 `height_scanner`（代码里已有注释示例），看地形适应性变化。

### 8.2 理解算法本身（2~4 周）

| 主题 | 要搞清楚的三个问题 | 入口 |
|---|---|---|
| PPO | 为什么要 clip？GAE 在算什么？KL 自适应怎么调 lr？ | rsl-rl 的 `rsl_rl/algorithms/ppo.py` |
| 非对称 actor-critic | critic 用特权信息、actor 只用可观测信息，为什么能上真机？ | `ObservationsCfg.PolicyCfg` vs `CriticCfg` |
| 并行采样 | 4096 个环境如何同步/异步采样？向量化怎么实现？ | IsaacLab `RslRlVecEnvWrapper` |
| 策略蒸馏 | 如何把大策略/特权策略蒸饱成可部署的小网络？ | rsl-rl `DistillationRunner` |

### 8.3 把策略变成产品（1~2 月）

1. **sim2sim**：先跑 `unitree_mujoco`，记录与 Isaac 的行为差异（速度、步态、能耗）；
2. **状态估计与延迟**：真机 IMU 有噪声/零偏，控制有 2~4 ms 延迟，需要补偿；
3. **安全层**：倾角/力矩/温度超限自动切 Passive；保留遥控急停；
4. **持续学习**：把真机日志回放成新场景，再做微调（DAgger / 域适应）。

### 8.4 扩展任务（有基础后）

| 方向 | 参考 |
|---|---|
| 复杂地形 / 越野 | 打开 `COBBLESTONE_ROAD_CFG` 里的随机粗糙、台阶、斜坡地形重训 |
| 动作跟踪（mimic） | 本仓库 `tasks/mimic/`（G1 跳舞/风格动作） |
| 多机器人 / 鲁棒性 | 随机推力、质量变化、电机故障注入 |
| 传统控制对照 | [GO2_CLASSICAL_GAIT.md](./GO2_CLASSICAL_GAIT.md) 的 trot 控制器 |

### 8.5 学习资源

- Isaac Lab 文档：<https://isaac-sim.github.io/IsaacLab>
- rsl-rl 源码：<https://github.com/leggedrobotics/rsl_rl>
- PPO 原论文（Schulman et al., 2017）与 GAE（2015）
- 四足 RL 经典：ANYmal 的 *Learning to Walk in Minutes*、*RMA: Rapid Motor Adaptation*
- 综述：*Learning-based legged locomotion: state of the art*

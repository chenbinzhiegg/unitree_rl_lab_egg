#!/usr/bin/env python3
"""阶段 3：单腿摆动轨迹（机身被“钉住”，只看足端运动）。

做法：每一步都把机体的 6 个自由度重置回固定高度，因此机器人相当于被吊起来，
只有一条腿在摆动 —— 用来单独验证摆线轨迹 + IK，不受平衡问题干扰。

运行：
    python 03_swing_trajectory.py
    python 03_swing_trajectory.py --leg FL --cycles 3 --viewer
验收标准：
    * 抬腿峰值 ≈ swing_height（±20%）
    * 落地时足心高度 ≈ 足半径 0.022 m，不穿地（> -5 mm）
"""

from __future__ import annotations

import argparse

import numpy as np

from gait import TrotController, TrotParams
from go2_model import FOOT_RADIUS, HOME_HEIGHT, HOME_LEG, LEGS, Go2, keep_viewer_open, sync_viewer, try_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段3：单腿摆动轨迹")
    parser.add_argument("--leg", default="FL", choices=list(LEGS))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--swing-height", type=float, default=0.06)
    parser.add_argument("--vx", type=float, default=0.0, help="给 0 只做垂直抬腿；>0 时摆动相会前后扫动")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    p = TrotParams(period=args.period, swing_height=args.swing_height, vx=args.vx)
    ctrl = TrotController(p)
    robot = Go2()
    robot.reset()
    viewer = try_viewer(robot.model, robot.data) if args.viewer else None

    dt = robot.model.opt.timestep
    duration = args.cycles * p.period
    n_steps = int(round(duration / dt))
    q_hold = np.tile(HOME_LEG, 4)
    hover = HOME_HEIGHT  # 吊住机身

    zs, xs = [], []
    print(f"吊起机身高度 {hover:.4f} m，仅 {args.leg} 摆动，周期 {p.period}s，抬腿 {p.swing_height}m\n")
    print(f"{'t (s)':>6} | {'phase':>6} | {'足心 z (m)':>10} | {'足心 x (m)':>10} | {'关节角 [hip,thigh,calf]':>28}")

    for k in range(n_steps):
        t = k * dt
        # 钉住机身（模拟“吊带”）
        robot.data.qpos[0:3] = (0.0, 0.0, hover)
        robot.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        robot.data.qvel[0:6] = 0.0

        q_des = q_hold.copy()
        target = ctrl.leg_target(args.leg, t)
        i = LEGS.index(args.leg)
        q_des[i * 3 : i * 3 + 3] = robot.leg_ik_base(args.leg, target)

        robot.apply_pd(q_des)
        robot.step()
        sync_viewer(viewer, dt)

        foot = robot.foot_pos(args.leg)
        zs.append(foot[2])
        xs.append(foot[0])
        if k % int(round(0.05 / dt)) == 0:
            ph = (t / p.period) % 1.0
            print(
                f"{t:6.2f} | {ph:6.2f} | {foot[2]:10.4f} | {foot[0]:10.4f} | "
                f"[{q_des[i * 3]:+.3f}, {q_des[i * 3 + 1]:+.3f}, {q_des[i * 3 + 2]:+.3f}]"
            )

    zs = np.array(zs)
    xs = np.array(xs)
    x_range = float(xs.max() - xs.min())
    peak = float(zs.max() - FOOT_RADIUS)
    min_z = float(zs.min())

    print("\n--- 阶段 3 结果 ---")
    print(f"抬腿峰值        : {peak * 100:.1f} cm（目标 {p.swing_height * 100:.1f} cm）")
    print(f"足心最低高度    : {min_z:.4f} m（足半径 {FOOT_RADIUS:.3f}）")
    print(f"足端前后行程    : {x_range * 100:.1f} cm")

    height_ok = abs(peak - p.swing_height) < 0.2 * p.swing_height
    ground_ok = min_z > FOOT_RADIUS - 0.005
    print(f"验收: {'通过 ✅' if (height_ok and ground_ok) else '未通过 ❌'}")

    if viewer is not None:
        keep_viewer_open(viewer, 0 if (height_ok and ground_ok) else 1)
    return 0 if (height_ok and ground_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

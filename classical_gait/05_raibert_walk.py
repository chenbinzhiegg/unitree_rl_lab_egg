#!/usr/bin/env python3
"""阶段 5：Raibert 落脚点 + 速度跟踪。

在阶段 4 的基础上加一条速度反馈：
    落脚点整体前移量 = k * (v_meas - v_cmd)
当前速度低于指令时把落脚点往后放（负偏移），支撑腿就会把机身往前推。
这就是 Raibert 的“速度伺服”思想，也是四足能跟踪速度指令的最小实现。

运行：
    python 05_raibert_walk.py
    python 05_raibert_walk.py --vx 0.5 --raibert-k 0.12 --duration 8 --viewer
验收标准：
    * 8 s 内不摔
    * 稳态平均速度与指令误差 < 0.15 m/s
"""

from __future__ import annotations

import argparse

import numpy as np

from gait import TrotController, TrotParams
from go2_model import Go2, keep_viewer_open, sync_viewer, try_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段5：Raibert 速度跟踪")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--duty", type=float, default=0.6)
    parser.add_argument("--swing-height", type=float, default=0.06)
    parser.add_argument("--raibert-k", type=float, default=0.4, help="速度反馈增益 [s]（本机整定 0.4）")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    p = TrotParams(
        period=args.period,
        duty=args.duty,
        swing_height=args.swing_height,
        vx=args.vx,
        vy=args.vy,
        raibert_k=args.raibert_k,
    )
    ctrl = TrotController(p)
    robot = Go2()
    robot.reset()
    viewer = try_viewer(robot.model, robot.data) if args.viewer else None

    dt = robot.model.opt.timestep
    n_steps = int(round(args.duration / dt))
    vxs, vys = [], []

    print(f"Raibert trot：指令=({p.vx:+.2f}, {p.vy:+.2f}) m/s, k={p.raibert_k}, 抬腿={p.swing_height}m\n")

    for k in range(n_steps):
        t = k * dt
        v_meas = robot.velocity_in_heading()
        q_des = robot.solve(ctrl.targets(t, roll=0.0, pitch=0.0, vx_meas=v_meas[0]))
        robot.apply_pd(q_des)
        robot.step()
        sync_viewer(viewer, dt)

        if t > 1.5:
            vxs.append(v_meas[0])
            vys.append(v_meas[1])

        if k % int(round(0.5 / dt)) == 0:
            print(f"t={t:5.2f}s  {robot.summary()}")

        if robot.fallen():
            print(f"\n[!] t={t:.2f}s 摔倒，提前结束")
            break

    mean_vx = float(np.mean(vxs)) if vxs else 0.0
    mean_vy = float(np.mean(vys)) if vys else 0.0
    err = abs(mean_vx - p.vx)

    print("\n--- 阶段 5 结果 ---")
    print(f"平均速度     : ({mean_vx:+.3f}, {mean_vy:+.3f}) m/s   指令 ({p.vx:+.3f}, {p.vy:+.3f})")
    print(f"速度误差     : {err:.3f} m/s")
    print(f"是否摔倒     : {'是 ❌' if robot.fallen() else '否 ✅'}")

    ok = (not robot.fallen()) and err < 0.15
    print(f"验收: {'通过 ✅' if ok else '未通过 ❌（调 raibert-k / period / duty）'}")

    if viewer is not None:
        keep_viewer_open(viewer, 0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

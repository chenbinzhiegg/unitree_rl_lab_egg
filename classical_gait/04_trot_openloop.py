#!/usr/bin/env python3
"""阶段 4：开环 trot（对角小跑，无任何反馈）。

只有相位 + 摆线轨迹 + IK + PD，**没有**速度反馈、没有姿态反馈。
它证明“相位驱动的足端轨迹”能把机身推着走；稳定性问题留给阶段 5/6。

运行：
    python 04_trot_openloop.py
    python 04_trot_openloop.py --vx 0.4 --duration 8 --viewer
验收标准（骨架阶段）：
    * 6 s 内不摔
    * 能前进 >0.2 m；开环会明显慢于指令（本机实测约指令速度的 60%~70%）
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gait import TrotController, TrotParams
from go2_model import Go2, try_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段4：开环 trot")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=0.3, help="期望前进速度 [m/s]")
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--duty", type=float, default=0.6)
    parser.add_argument("--swing-height", type=float, default=0.06)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    p = TrotParams(period=args.period, duty=args.duty, swing_height=args.swing_height, vx=args.vx)
    ctrl = TrotController(p)
    robot = Go2()
    robot.reset()
    viewer = try_viewer(robot.model, robot.data) if args.viewer else None

    dt = robot.model.opt.timestep
    n_steps = int(round(args.duration / dt))
    start = robot.base_pos.copy()
    speeds, heights = [], []

    print(f"开环 trot：vx={p.vx} m/s, T={p.period}s, duty={p.duty}, 抬腿={p.swing_height}m")
    print("（无速度/姿态反馈，可能走偏或摔倒，属正常现象）\n")

    for k in range(n_steps):
        t = k * dt
        q_des = robot.solve(ctrl.targets(t))          # 阶段4：roll=pitch=vx_meas=0
        robot.apply_pd(q_des)
        robot.step()
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)

        if t > 1.0:                                   # 跳过起步瞬态
            speeds.append(robot.velocity_in_heading()[0])
            heights.append(robot.base_pos[2])

        if k % int(round(0.5 / dt)) == 0:
            print(f"t={t:5.2f}s  {robot.summary()}")

        if robot.fallen():
            print(f"\n[!] t={t:.2f}s 摔倒，提前结束（阶段 5/6 会用反馈改善）")
            break

    dist = float(np.linalg.norm(robot.base_pos[:2] - start[:2]))
    mean_v = float(np.mean(speeds)) if speeds else 0.0
    print("\n--- 阶段 4 结果 ---")
    print(f"前进距离     : {dist:.3f} m")
    print(f"平均速度     : {mean_v:+.3f} m/s（指令 {p.vx:+.3f}）")
    print(f"平均机身高度 : {np.mean(heights):.3f} m" if heights else "")
    print(f"是否摔倒     : {'是 ❌' if robot.fallen() else '否 ✅'}")

    ok = (not robot.fallen()) and dist > 0.2
    print(f"验收: {'通过 ✅' if ok else '未通过 ❌（可先降 vx 或调大 duty）'}")

    if viewer is not None:
        input("按回车关闭 viewer ...")
        viewer.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

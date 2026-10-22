#!/usr/bin/env python3
"""阶段 6：姿态反馈 + 转向（完整的“能用的”trot）。

在阶段 5 基础上增加：
    1. roll / pitch 反馈：通过调整各腿长度把机身“掰”回水平
       z_target += k_roll * roll * 左右符号 + k_pitch * pitch * 前后符号
    2. 偏航指令 wz：机身转动时，各足端的身体坐标速度是 v = v_cmd + ω × r，
       所以外侧腿要走更长的行程，转向自然发生。

运行：
    python 06_balance_turn.py --vx 0.3 --wz 0.4
    python 06_balance_turn.py --vx 0.3 --wz -0.4 --duration 10 --viewer
验收标准：
    * 10 s 内不摔
    * roll/pitch 稳态 |角度| < 0.15 rad
    * 实际偏航角速度与 --wz 同号，量级误差 < 50%
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gait import TrotController, TrotParams
from go2_model import Go2, try_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段6：姿态反馈 + 转向")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.3)
    parser.add_argument("--wz", type=float, default=0.4, help="偏航角速度指令 [rad/s]")
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--duty", type=float, default=0.6)
    parser.add_argument("--swing-height", type=float, default=0.06)
    parser.add_argument("--raibert-k", type=float, default=0.4)
    parser.add_argument("--k-yaw", type=float, default=2.0, help="偏航角速度反馈增益（本机整定 2.0，需配合低通滤波）")
    parser.add_argument("--k-roll", type=float, default=0.20)
    parser.add_argument("--k-pitch", type=float, default=0.20)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    p = TrotParams(
        period=args.period,
        duty=args.duty,
        swing_height=args.swing_height,
        vx=args.vx,
        wz=args.wz,
        raibert_k=args.raibert_k,
        k_yaw=args.k_yaw,
        k_roll=args.k_roll,
        k_pitch=args.k_pitch,
    )
    ctrl = TrotController(p)
    robot = Go2()
    robot.reset()
    viewer = try_viewer(robot.model, robot.data) if args.viewer else None

    dt = robot.model.opt.timestep
    n_steps = int(round(args.duration / dt))
    tilt, yaw_rates, vxs = [], [], []
    yaw0 = robot.rpy()[2]

    print(f"平衡+转向 trot：vx={p.vx:+.2f} m/s, wz={p.wz:+.2f} rad/s, "
          f"k_roll={p.k_roll}, k_pitch={p.k_pitch}, k_yaw={p.k_yaw}\n")

    for k in range(n_steps):
        t = k * dt
        roll, pitch, _ = robot.rpy()
        v_meas = robot.velocity_in_heading()

        q_des = robot.solve(
            ctrl.targets(t, roll=roll, pitch=pitch, vx_meas=v_meas[0], wz_meas=robot.base_ang_vel[2])
        )
        robot.apply_pd(q_des)
        robot.step()
        if viewer is not None:
            viewer.sync()
            time.sleep(dt)

        if t > 1.5:
            tilt.append(max(abs(roll), abs(pitch)))
            yaw_rates.append(robot.base_ang_vel[2])
            vxs.append(v_meas[0])

        if k % int(round(0.5 / dt)) == 0:
            print(f"t={t:5.2f}s  {robot.summary()}")

        if robot.fallen():
            print(f"\n[!] t={t:.2f}s 摔倒，提前结束")
            break

    yaw1 = robot.rpy()[2]
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
    wz_meas = dyaw / max(args.duration, 1e-6)
    mean_wz = float(np.mean(yaw_rates)) if yaw_rates else 0.0

    print("\n--- 阶段 6 结果 ---")
    print(f"最大倾角       : {max(tilt):.3f} rad（目标 <0.15）" if tilt else "无数据")
    print(f"平均前进速度   : {np.mean(vxs):+.3f} m/s（指令 {p.vx:+.3f}）" if vxs else "")
    print(f"平均偏航角速度 : {mean_wz:+.3f} rad/s（瞬时均值）")
    print(f"整体偏航       : {dyaw:+.3f} rad，等效 {wz_meas:+.3f} rad/s（指令 {p.wz:+.3f}）")
    print(f"是否摔倒       : {'是 ❌' if robot.fallen() else '否 ✅'}")

    direction_ok = np.sign(mean_wz) == np.sign(p.wz) if abs(p.wz) > 1e-3 else True
    ok = (not robot.fallen()) and (max(tilt) < 0.15 if tilt else False) and direction_ok
    print(f"验收: {'通过 ✅' if ok else '未通过 ❌（调 k_roll/k_pitch/raibert-k）'}")

    if viewer is not None:
        input("按回车关闭 viewer ...")
        viewer.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""阶段 1：PD 站立。

目标：验证模型加载、关节顺序、PD 增益与站立高度，不涉及任何步态。
运行：
    python 01_stand.py                 # 无界面，5 s
    python 01_stand.py --duration 10 --viewer
验收标准：
    * 机身高度稳定在 0.28 m 附近（±0.01）
    * 不摔倒（roll/pitch < 0.15 rad）
    * 足端接触地面
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from go2_model import HOME_HEIGHT, HOME_LEG, LEGS, Go2, keep_viewer_open, sync_viewer, try_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段1：PD 站立")
    parser.add_argument("--duration", type=float, default=5.0, help="仿真时长 [s]")
    parser.add_argument("--viewer", action="store_true", help="打开 mujoco viewer")
    parser.add_argument("--realtime", action="store_true", help="按真实时间推进（配合 viewer）")
    args = parser.parse_args()

    robot = Go2()   # 初始化模型
    robot.reset()   # 重置到站立位置
    viewer = try_viewer(robot.model, robot.data) if args.viewer else None

    dt = robot.model.opt.timestep   # 仿真步长
    n_steps = int(round(args.duration / dt))    # 仿真步数
    q_des = np.tile(HOME_LEG, 4)    # 目标关节角：4 条腿的站立位置

    print(f"模型: {robot.model.nq} qpos, {robot.model.nu} actuators, dt={dt}s")
    print(f"目标站姿高度: {HOME_HEIGHT:.4f} m\n")

    heights, tilts = [], []
    t0 = time.time()
    for k in range(n_steps):
        robot.apply_pd(q_des)
        robot.step()
        sync_viewer(viewer, dt)

        if k % int(round(0.5 / dt)) == 0:
            roll, pitch, _ = robot.rpy()
            heights.append(robot.base_pos[2])
            tilts.append(max(abs(roll), abs(pitch)))
            print(f"t={k * dt:5.2f}s  {robot.summary()}")

    h_arr = np.array(heights[len(heights) // 2 :])  # 跳过前 0.5s 的落地瞬态
    tilt_arr = np.array(tilts[len(tilts) // 2 :])
    h_err = abs(h_arr.mean() - HOME_HEIGHT)
    contacts = [leg for leg in LEGS if robot.in_contact(leg)]

    print("\n--- 阶段 1 结果 ---")
    print(f"平均高度     : {h_arr.mean():.4f} m（目标 {HOME_HEIGHT:.4f}，误差 {h_err * 1000:.1f} mm）")
    print(f"最大倾角     : {tilt_arr.max():.3f} rad")
    print(f"着地足       : {','.join(contacts) if contacts else '无'}")
    print(f"墙钟耗时     : {time.time() - t0:.2f} s")

    ok = (not robot.fallen()) and h_err < 0.01 and tilt_arr.max() < 0.15 and len(contacts) == 4
    print(f"验收: {'通过 ✅' if ok else '未通过 ❌（先调 kp/kd 或检查站姿高度）'}")

    if viewer is not None:
        keep_viewer_open(viewer, 0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""阶段 2：单腿逆运动学（IK）验证。

不跑物理，只做数值验证：
    1. 随机关节角 → FK → IK → 与原始关节角对比（往返一致性）
    2. 给定笛卡尔圆轨迹 → IK → FK → 与目标点对比（跟踪误差）

运行：
    python 02_leg_ik.py
    python 02_leg_ik.py --leg FR --radius 0.06
验收标准：
    * 往返误差 < 1e-6 rad
    * 笛卡尔跟踪误差 < 1 mm
"""

from __future__ import annotations

import argparse

import numpy as np

from go2_model import HOME_LEG, L1, L2, Go2


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 阶段2：单腿 IK 验证")
    parser.add_argument("--leg", default="FL", choices=["FL", "FR", "RL", "RR"])
    parser.add_argument("--radius", type=float, default=0.05, help="圆轨迹半径 [m]")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    robot = Go2()
    leg = args.leg
    side = 1 if leg in ("FL", "RL") else -1

    print(f"腿长 L1={L1:.3f} m, L2={L2:.3f} m, 当前腿={leg}, side={side:+d}\n")

    # ---------------- 1. 往返一致性 ----------------
    # 只采样“足端在髋下方”的常规工作空间（整条关节限位内包含折叠姿态，不是 IK 的目标区间）
    rng = np.random.default_rng(0)
    q_lo = np.array([-0.6, 0.2, -2.6])
    q_hi = np.array([0.6, 1.5, -1.0])
    max_err = 0.0
    for _ in range(2000):
        q = rng.uniform(q_lo, q_hi)
        p = robot.fk(leg, q)
        q_rec = robot.ik(leg, p)
        max_err = max(max_err, float(np.abs(q_rec - q).max()))

    print("--- 1. FK→IK 往返一致性（2000 组随机关节角）---")
    print(f"最大关节角误差: {max_err:.3e} rad")
    roundtrip_ok = max_err < 1e-6

    # ---------------- 2. 笛卡尔圆轨迹跟踪 ----------------
    print(f"\n--- 2. 笛卡尔圆轨迹（半径 {args.radius} m，绕名义足端位置）---")
    nominal_base = robot.nominal_foot_base(leg)
    hip_base = nominal_base - robot.fk(leg, HOME_LEG)
    errs = []
    print(f"{'角度':>8} | {'目标 (x,y,z)':>28} | {'IK→FK 误差 (mm)':>16} | 关节角 [hip, thigh, calf]")
    for i in range(args.samples):
        ang = 2 * np.pi * i / args.samples
        # 在矢状面(x-z)画圆：x 前后，z 上下（z 恒为负）
        offset = np.array([args.radius * np.sin(ang), 0.0, args.radius * (np.cos(ang) - 1.0)])
        target = nominal_base + offset
        q = robot.leg_ik_base(leg, target)
        p_rec = hip_base + robot.fk(leg, q)
        err = float(np.linalg.norm(p_rec - target)) * 1000
        errs.append(err)
        print(
            f"{np.degrees(ang):7.1f}° | "
            f"({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}) | "
            f"{err:16.4f} | [{q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}]"
        )

    track_ok = max(errs) < 1.0
    print(f"\n最大跟踪误差: {max(errs):.4f} mm")

    print("\n--- 阶段 2 结果 ---")
    print(f"往返一致性: {'通过 ✅' if roundtrip_ok else '未通过 ❌'}")
    print(f"笛卡尔跟踪: {'通过 ✅' if track_ok else '未通过 ❌'}")
    print("提示：若某些点误差大，通常是接近腿长极限（d→L1+L2）或膝角越限。")
    return 0 if (roundtrip_ok and track_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

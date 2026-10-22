"""步态与足端轨迹：trot 相位调度、摆线摆动相、落脚点规划。

坐标：机体系（x 前 / y 左 / z 上），足端目标以**机体系**给出。
位置控制的本质：支撑相让足端相对机身向后移动，机身就被“推”向前。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go2_model import HIP_X, HIP_Y, LEG_EXT, LEG_FRONT, LEG_SIDE, LEGS, THIGH_Y_OFF


def cycloid(u: float) -> float:
    """摆线：速度在两端为 0，中间最大，适合做摆动相水平位移。"""
    return u - np.sin(2 * np.pi * u) / (2 * np.pi)


def bell(u: float) -> float:
    """0→1→0 的平滑抬腿曲线（sin² 形状）。"""
    return (1.0 - np.cos(2 * np.pi * u)) / 2.0


@dataclass
class TrotParams:
    """trot 步态参数。"""

    period: float = 0.5           # 步态周期 [s]
    duty: float = 0.6             # 支撑相占比（0.5~0.7 常见）
    swing_height: float = 0.06    # 抬腿高度 [m]
    body_height: float = LEG_EXT  # 机身到足心高度 [m]（LEG_EXT≈0.2647）
    vx: float = 0.3               # 指令前进速度 [m/s]
    vy: float = 0.0               # 指令侧移速度 [m/s]
    wz: float = 0.0               # 指令偏航角速度 [rad/s]
    raibert_k: float = 0.0        # Raibert 速度反馈增益（阶段5起 >0）
    k_yaw: float = 0.0            # 偏航角速度反馈增益（阶段6起 >0）
    wz_alpha: float = 0.05        # 角速度反馈的低通系数（0=不滤波）
    wz_limit: float = 2.0         # 有效角速度限幅 [rad/s]
    wz_gate: float = 0.2          # |wz_cmd| 低于此值时不给反馈（避免直行时被噪声拖慢）
    k_roll: float = 0.0           # 横滚补偿增益（阶段6起 >0）
    k_pitch: float = 0.0          # 俯仰补偿增益（阶段6起 >0）


class TrotController:
    """相位驱动的 trot 控制器：给定时刻 → 12 个关节目标角。

    对角腿同相（FL+RR 相位 0，FR+RL 相位 0.5），因此任意时刻都有两条腿支撑。
    """

    #: 对角小跑相位偏移
    PHASE_OFFSET = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}

    def __init__(self, params: TrotParams):
        self.p = params
        self._wz_filt = 0.0

    # ------------------------------------------------------------------ 名义位姿
    @staticmethod
    def nominal(leg: str) -> np.ndarray:
        """机体系下的名义足端位置（站立时足心位置）。"""
        return np.array(
            [
                HIP_X * LEG_FRONT[leg],
                HIP_Y * LEG_SIDE[leg] + THIGH_Y_OFF * LEG_SIDE[leg],
                -LEG_EXT,
            ]
        )

    def body_point_velocity(self, leg: str, wz: float | None = None) -> np.ndarray:
        """刚体运动学：机体以 (vx,vy,wz) 运动时，足端处的 xy 速度。"""
        nom = self.nominal(leg)
        wz = self.p.wz if wz is None else wz
        vx = self.p.vx - wz * nom[1]
        vy = self.p.vy + wz * nom[0]
        return np.array([vx, vy])

    # ------------------------------------------------------------------ 单腿轨迹
    def leg_target(
        self,
        leg: str,
        t: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        vx_meas: float = 0.0,
        wz_meas: float = 0.0,
    ) -> np.ndarray:
        """返回该腿在机体系下的足端目标。"""
        p = self.p
        ph = (t / p.period + self.PHASE_OFFSET[leg]) % 1.0
        nom = self.nominal(leg)
        # 偏航反馈：转得不够快时把有效角速度放大，让左右腿行程差更大。
        # 增益随指令大小渐变（直行时自动关闭），避免把步态抖动放大成横向摆动。
        gain = p.k_yaw * min(1.0, abs(p.wz) / max(p.wz_gate, 1e-6))
        wz_eff = p.wz + gain * (p.wz - wz_meas)
        wz_eff = float(np.clip(wz_eff, -p.wz_limit, p.wz_limit))
        vp = self.body_point_velocity(leg, wz_eff)
        t_stance = p.duty * p.period

        if ph < p.duty:                      # ---- 支撑相：足端相对机身向后匀速移动
            u = ph / p.duty
            xy = nom[:2] - vp * t_stance * (u - 0.5)
            z = nom[2]
        else:                                # ---- 摆动相：从离地点摆到落地点
            u = (ph - p.duty) / (1.0 - p.duty)
            p0 = nom[:2] - vp * t_stance * 0.5
            p1 = nom[:2] + vp * t_stance * 0.5
            xy = p0 + (p1 - p0) * cycloid(u)
            z = nom[2] + p.swing_height * bell(u)

        # Raibert 落脚点反馈：当前速度低于指令时把落脚点往后放，从而加速
        xy = xy.copy()
        xy[0] += p.raibert_k * (vx_meas - p.vx)

        # 姿态补偿（腿相当于支撑杆）：
        #   roll>0 左高右低 → 缩短左腿（z 目标抬高）
        #   pitch>0 低头 → 加长前腿（z 目标降低），从而把机身“撑”回水平
        z += p.k_roll * roll * LEG_SIDE[leg] - p.k_pitch * pitch * LEG_FRONT[leg]

        return np.array([xy[0], xy[1], z])

    def targets(
        self,
        t: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        vx_meas: float = 0.0,
        wz_meas: float = 0.0,
    ) -> dict[str, np.ndarray]:
        # 对角小跑的 yaw 角速度天然带步态频率的抖动，直接反馈会把抖动放大成横向摆动；
        # 所以先做一阶低通（wz_alpha 每控制周期更新一次）。
        if self.p.wz_alpha > 0.0:
            self._wz_filt += self.p.wz_alpha * (wz_meas - self._wz_filt)
            wz_meas = self._wz_filt
        return {leg: self.leg_target(leg, t, roll, pitch, vx_meas, wz_meas) for leg in LEGS}

    def swing_ratio(self, t: float, leg: str) -> float:
        """该腿当前处于摆动相的比例（0=支撑，>0 表示摆动进度）。"""
        ph = (t / self.p.period + self.PHASE_OFFSET[leg]) % 1.0
        return 0.0 if ph < self.p.duty else (ph - self.p.duty) / (1.0 - self.p.duty)

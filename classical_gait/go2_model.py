"""Go2 的 MuJoCo 封装：模型常量、正逆运动学、PD 力矩控制。

模型来源（Unitree 官方 MJCF）：
    ~/unitree_mujoco/unitree_robots/go2/go2.xml

约定
----
* 关节角单位 rad，长度单位 m，力矩 N·m，时间 s。
* 关节顺序统一为 ``LEGS x (hip, thigh, calf)``，即
  FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_calf。
* 机体坐标系：x 向前，y 向左，z 向上。

注意：MJCF 里的 actuator 顺序是 FR/FL/RR/RL，与上面的 LEQS 顺序**不同**，
本模块通过关节名反查 actuator 编号，避免下标错位。
"""

from __future__ import annotations

import os
import sys
import time

import mujoco
import numpy as np

# ----------------------------------------------------------------------------
# 几何常量（数值直接来自 go2.xml，改动模型时请同步）
# ----------------------------------------------------------------------------
DEFAULT_SCENE = os.path.expanduser("~/unitree_mujoco/unitree_robots/go2/scene.xml")

L1 = 0.213          # thigh：髋(侧摆后) → 膝
L2 = 0.213          # calf ：膝 → 足心
HIP_X = 0.1934      # 髋关节在机体系中的 x 偏移
HIP_Y = 0.0465      # 髋关节在机体系中的 y 偏移
THIGH_Y_OFF = 0.0955  # 侧摆电机 → 腿平面 的 y 偏移
FOOT_RADIUS = 0.022   # 足端球半径

HOME_LEG = np.array([0.0, 0.9, -1.8])                     # [hip, thigh, calf]
LEG_EXT = 2.0 * L1 * np.cos(HOME_LEG[1])                  # 站姿髋到足心距离 ≈0.2647
HOME_HEIGHT = LEG_EXT + FOOT_RADIUS                       # 站姿机身高度 ≈0.2867

LEGS = ("FL", "FR", "RL", "RR")
LEG_SIDE = {"FL": 1, "FR": -1, "RL": 1, "RR": -1}         # 左 +1 / 右 -1
LEG_FRONT = {"FL": 1, "FR": 1, "RL": -1, "RR": -1}        # 前 +1 / 后 -1

# 起始增益（待站立测试校核）；比 deploy 的 FixStand 略硬，以减小重力引起的稳态下垂
KP_STAND = np.array([80.0, 120.0, 120.0])
KD_STAND = np.array([5.0, 2.0, 2.0])
TAU_LIMIT = np.array([23.7, 23.7, 45.43])                 # 与 MJCF ctrlrange 一致

JOINT_NAMES = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]


def _expand(x, n: int = 12) -> np.ndarray:
    """把长度 3（单腿）或 12（全部）的数组扩展成 12。"""
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 3:
        return np.tile(a, 4)
    if a.size == n:
        return a
    raise ValueError(f"期望长度 3 或 {n}，得到 {a.size}")


def try_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    """尝试打开交互式 viewer；无显示环境时返回 None（不影响仿真）。"""
    try:
        import mujoco.viewer

        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:  # pragma: no cover - 依赖本机图形环境
        print(f"[warn] 无法打开 viewer（{exc}），改为无界面运行")
        return None


def sync_viewer(viewer, dt: float) -> None:
    """同步一次 viewer，并按 dt 节流到实时速度。

    没有 viewer 时什么都不做。用户关闭窗口时抛出 SystemExit(0)，让脚本正常收尾。
    """
    if viewer is None:
        return
    if not viewer.is_running():
        raise SystemExit(0)
    viewer.sync()
    time.sleep(dt)


def keep_viewer_open(viewer, exit_code: int = 0) -> None:
    """仿真结束后保持窗口，直到用户关闭。

    MuJoCo 的 GLFW viewer 在某些显卡驱动上销毁 GL 上下文时会段错误（exit 139），
    因此先尝试正常 close，随后直接 os._exit，保证返回码干净。
    """
    if viewer is None:
        return
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    try:
        viewer.close()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


class Go2:
    """Go2 机器人封装（一个 model + 一个 data）。"""

    def __init__(self, scene_path: str = DEFAULT_SCENE, flat_ground: bool = True, timestep: float = 0.002):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        if flat_ground:
            self._flatten_world()

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        )
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.act_ids = np.array([self._actuator_of_joint(j) for j in self.joint_ids])

        self.hip_bid = {leg: self._body_id(f"{leg}_hip") for leg in LEGS}
        self.foot_bid = {leg: self._body_id(f"{leg}_foot") for leg in LEGS}
        self.foot_gid = {leg: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS}
        self.base_bid = self._body_id("base_link")

        self.reset()

    # ------------------------------------------------------------------ 初始化
    @property
    def joint_names(self) -> list[str]:
        return JOINT_NAMES

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _actuator_of_joint(self, joint_id: int) -> int:
        for i in range(self.model.nu):
            if int(self.model.actuator_trnid[i, 0]) == int(joint_id):
                return i
        raise RuntimeError(f"找不到关节 {joint_id} 对应的 actuator")

    def _flatten_world(self) -> None:
        """关闭 worldbody 中的障碍物 box（scene.xml 里 x>1m 处的几块砖），保留地面。"""
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] != 0:
                continue
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                self.model.geom_contype[i] = 0
                self.model.geom_conaffinity[i] = 0

    # ------------------------------------------------------------------ 状态
    def reset(self, height: float | None = None, yaw: float = 0.0) -> None:
        """把机器人放到原点、给定高度与朝向，采用 home 站姿。"""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        h = HOME_HEIGHT + 0.002 if height is None else float(height)
        self.data.qpos[0:3] = (0.0, 0.0, h)
        self.data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        self.data.qpos[self.qpos_adr] = np.tile(HOME_LEG, 4)
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def base_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    @property
    def base_lin_vel(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    def rpy(self) -> tuple[float, float, float]:
        """返回机体的 roll / pitch / yaw（rad）。"""
        R = self.data.xmat[self.base_bid].reshape(3, 3)
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return roll, pitch, yaw

    def velocity_in_heading(self) -> np.ndarray:
        """把世界系线速度投影到当前朝向的机体 xy 方向。"""
        _, _, yaw = self.rpy()
        c, s = np.cos(yaw), np.sin(yaw)
        vx_w, vy_w = self.base_lin_vel[0], self.base_lin_vel[1]
        return np.array([c * vx_w + s * vy_w, -s * vx_w + c * vy_w])

    def foot_pos(self, leg: str) -> np.ndarray:
        """足心世界坐标。"""
        return self.data.xpos[self.foot_bid[leg]].copy()

    def foot_pos_base(self, leg: str) -> np.ndarray:
        """足心在机体系下的坐标。"""
        R = self.data.xmat[self.base_bid].reshape(3, 3)
        return R.T @ (self.foot_pos(leg) - self.base_pos)

    def in_contact(self, leg: str) -> bool:
        gid = self.foot_gid[leg]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return True
        return False

    # ------------------------------------------------------------------ 运动学
    def fk(self, leg: str, q: np.ndarray) -> np.ndarray:
        """单腿正运动学：关节角 → 足心在**髋关节坐标系**下的位置。"""
        q1, q2, q3 = q
        side = LEG_SIDE[leg]
        xp = -L1 * np.sin(q2) - L2 * np.sin(q2 + q3)
        zp = -L1 * np.cos(q2) - L2 * np.cos(q2 + q3)
        yp0 = side * THIGH_Y_OFF
        x = xp
        y = np.cos(q1) * yp0 - np.sin(q1) * zp
        z = np.sin(q1) * yp0 + np.cos(q1) * zp
        return np.array([x, y, z])

    def ik(self, leg: str, p_hip: np.ndarray) -> np.ndarray:
        """单腿逆运动学：足心在髋坐标系下的位置 → [hip, thigh, calf]。

        膝角取负解（Go2 的 knee 关节限位就是负区间）。
        """
        x, y, z = float(p_hip[0]), float(p_hip[1]), float(p_hip[2])
        side = LEG_SIDE[leg]

        # 1) 侧摆：先在 yz 平面解 q1
        m = np.hypot(y, z)
        r = np.sqrt(max(m * m - THIGH_Y_OFF**2, 0.0))
        # y + i z = e^{i q1} (yp0 + i zp)，其中 yp0 = side*off, zp = -r
        q1 = np.arctan2(z, y) - np.arctan2(-r, side * THIGH_Y_OFF)

        # 2) 腿平面内两连杆 IK（腿长 L1 + L2）
        d = np.hypot(x, r)
        d = np.clip(d, abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4)
        cos_knee = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        q3 = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
        cos_beta = (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
        delta = np.arctan2(x, r)
        q2 = beta - delta
        return np.array([q1, q2, q3])

    def leg_ik_base(self, leg: str, p_base: np.ndarray) -> np.ndarray:
        """足心在机体系下的目标位置 → 该腿三个关节角。"""
        hip = np.array([HIP_X * LEG_FRONT[leg], HIP_Y * LEG_SIDE[leg], 0.0])
        return self.ik(leg, np.asarray(p_base, dtype=float) - hip)

    def nominal_foot_base(self, leg: str) -> np.ndarray:
        """站立（HOME_LEG）时足心在机体系中的位置。"""
        hip = np.array([HIP_X * LEG_FRONT[leg], HIP_Y * LEG_SIDE[leg], 0.0])
        return hip + self.fk(leg, HOME_LEG)

    def solve(self, targets_base: dict[str, np.ndarray]) -> np.ndarray:
        """多腿目标（机体系）→ 12 维关节角，按 LEGS 顺序。"""
        q = np.zeros((4, 3))
        for i, leg in enumerate(LEGS):
            q[i] = self.leg_ik_base(leg, targets_base[leg])
        return q.reshape(-1)

    # ------------------------------------------------------------------ 控制
    def apply_pd(
        self,
        q_des: np.ndarray,
        kp: np.ndarray = KP_STAND,
        kd: np.ndarray = KD_STAND,
        tau_limit: np.ndarray = TAU_LIMIT,
    ) -> np.ndarray:
        """力矩 PD：tau = kp*(q_des-q) - kd*qd，然后裁剪到电机限幅。"""
        kp = _expand(kp)
        kd = _expand(kd)
        lim = _expand(tau_limit)
        q_err = np.asarray(q_des, dtype=float).reshape(-1) - self.joint_pos
        tau = kp * q_err - kd * self.joint_vel
        tau = np.clip(tau, -lim, lim)
        self.data.ctrl[self.act_ids] = tau
        return tau

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def hold(self, seconds: float, q_des: np.ndarray | None = None) -> None:
        """保持某个关节目标一段时间（按物理步长推进）。"""
        q_des = np.tile(HOME_LEG, 4) if q_des is None else np.asarray(q_des, dtype=float)
        n = int(round(seconds / self.model.opt.timestep))
        for _ in range(n):
            self.apply_pd(q_des)
            self.step()

    # ------------------------------------------------------------------ 诊断
    def summary(self) -> str:
        roll, pitch, yaw = self.rpy()
        vx, vy = self.velocity_in_heading()
        feet = " ".join(f"{leg}:{self.foot_pos(leg)[2]:+.3f}" for leg in LEGS)
        return (
            f"h={self.base_pos[2]:.3f}m  rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f})  "
            f"v=({vx:+.2f},{vy:+.2f})  feet_z[{feet}]"
        )

    def fallen(self, min_height: float = 0.16, max_tilt: float = 0.7) -> bool:
        roll, pitch, _ = self.rpy()
        return self.base_pos[2] < min_height or abs(roll) > max_tilt or abs(pitch) > max_tilt

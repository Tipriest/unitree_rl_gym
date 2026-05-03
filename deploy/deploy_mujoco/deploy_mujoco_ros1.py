import time
import threading

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from typing import List, Tuple




class CmdVelReceiver:
    """ROS1 /cmd_vel receiver.

    Maps geometry_msgs/Twist to cmd = [vx, vy, wz] (policy command order).
    """

    def __init__(self, topic: str = "/cmd_vel") -> None:
        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._last_msg_time = None  # type: float | None

        try:
            import rospy  # type: ignore
            from geometry_msgs.msg import Twist  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ROS1 python packages not available. "
                "Please source your ROS1 environment (e.g. `source /opt/ros/noetic/setup.bash`) "
                "and ensure `rospy` + `geometry_msgs` are installed."
            ) from e

        self._rospy = rospy
        self._Twist = Twist

        self._sub = rospy.Subscriber(topic, Twist, self._cb, queue_size=1)

    def _cb(self, msg) -> None:
        cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)
        with self._lock:
            self._cmd[:] = cmd
            try:
                self._last_msg_time = float(self._rospy.get_time())
            except Exception:
                self._last_msg_time = time.time()

    def get_cmd(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    @property
    def last_msg_time(self):
        with self._lock:
            return self._last_msg_time

# ROS1 subscriber for /cmd_vel
cmd_vel_receiver = None
try:
    import rospy  # type: ignore

    rospy.init_node("deploy_mujoco", anonymous=True, disable_signals=True)
    cmd_vel_receiver = CmdVelReceiver(topic="/cmd_vel")
    print("[deploy_mujoco_ros1] ROS1 subscribed to: /cmd_vel")
except Exception as e:
    print(f"[deploy_mujoco_ros1] ROS1 init/subscribe failed: {e}")
    print("[deploy_mujoco_ros1] Continuing without ROS commands (using cmd_init).")
    cmd_vel_receiver = None

def quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    """_summary_
    求共轭四元数
    前提是该四元数归一化为单位四元数
    在 3D 旋转中, 如果你的四元数是单位四元数(norm=1), 那么:
    四元数的逆, 共轭就是它的逆, 常用于把“从 A 到 B 的旋转”变成“从 B 回到 A 的旋转”
    Args:
        q_wxyz (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    # 把输入 q_wxyz 转成 NumPy 数组，类型是 float32
    # 这样无论你传的是 list、tuple 还是 ndarray，都统一成 float32 的 ndarray
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    return np.array([q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]], dtype=np.float32)

def quat_apply(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """用四元数旋转一个3d向量
    这里假设 q_wxyz 是单位四元数(norm=1), 表示一个 3D 旋转。
    四元数可以写成：
        q=(w,q)=(w,(x,y,z))
    Args:
        q_wxyz: shape (4,)
        v_xyz: shape (3,)

    Returns:
        Rotated vector shape (3,)
    """
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    v_xyz = np.asarray(v_xyz, dtype=np.float32)
    w = q_wxyz[0]
    qvec = q_wxyz[1:]
    t = 2.0 * np.cross(qvec, v_xyz)
    return v_xyz + w * t + np.cross(qvec, t)

def quat_rotate_inverse(q_wxyz: np.ndarray, v_xyz: np.ndarray) -> np.ndarray:
    """用四元数向反方向旋转一个3d向量

    Args:
        q_wxyz (np.ndarray): _description_
        v_xyz (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    return quat_apply(quat_conjugate(q_wxyz), v_xyz)

def get_actuated_joint_names_in_ctrl_order(model: mujoco.MjModel) -> List[str]:
    """按照控制量（actuator）的顺序，返回对应的关节名字列表
    Args:
        model (mujoco.MjModel): _description_

    Returns:
        List[str]: _description_
    """
    names: List[str] = []
    # nu 是 MuJoCo 里控制量（actuators）的个数
    # 也就是 ctrl 向量的长度：data.ctrl.shape == (model.nu,)
    # 遍历所有 actuator 的索引 act_id = 0, 1, ..., nu-1，顺序与 data.ctrl 中的控制量顺序一致
    for act_id in range(model.nu):
        # actuator_trnid 是一个（nu, 2）的数组，每一行存储当前 actuator 作用的目标:
        # [..., 0]：target 的 id（比如 joint id 或 tendon id）
        # [..., 1]：目标类型（joint / tendon 等，对应某种 enum）
        # 在你这个模型配置里，显然 actuator 的 target 是 joint，所以取 [act_id][0] 就是这个 actuator 关联的 joint 的 id。
        # int(...) 保证转成 Python int 类型，方便后面访问。
        joint_id = int(model.actuator_trnid[act_id][0])
        names.append(model.joint(joint_id).name)
    return names

def get_actuated_q_dq_in_ctrl_order(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
    """按控制量（actuator）的顺序，取出对应关节的一维位置 q 和速度 dq，并返回两个对齐的数组。

    Args:
        model (mujoco.MjModel): _description_
        data (mujoco.MjData): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    # 初始化输出数组
    q = np.zeros(model.nu, dtype=np.float32)
    dq = np.zeros(model.nu, dtype=np.float32)
    # model.nu 是 actuator 的数量，也就是 data.ctrl 的长度。
    # q[i] 用来放第 i 个 actuator 控制的关节位置
    # dq[i] 用来放对应的关节速度
    # act_id 是 actuator 的索引（0 到 nu-1），顺序与 data.ctrl 一样。
    for act_id in range(model.nu):
        # model.actuator_trnid[act_id][0] 是这个 actuator 作用的“目标”的 id。
        # data.qpos 是所有自由度“位置”的堆叠（长度 model.nq）
        # data.qvel 是所有自由度“速度”的堆叠（长度 model.nv）
        # 但一个 joint 不一定只占 1 个元素（比如球关节是 4 个 qpos / 3 个 qvel）。
        # jnt_qposadr 和 jnt_dofadr 是“起始地址表”：
        # model.jnt_qposadr[joint_id]
        # 给出该 joint 在 data.qpos 中的起始索引
        # model.jnt_dofadr[joint_id]
        # 给出该 joint 第一维自由度在 data.qvel 中的起始索引
        joint_id = int(model.actuator_trnid[act_id][0])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        q[act_id] = data.qpos[qpos_adr]
        dq[act_id] = data.qvel[dof_adr]
    return q, dq

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)

        policy_dof_names = config.get("policy_dof_names", None)
        action_clip = config.get("action_clip", None)

        actuator_mode = str(config.get("actuator_mode", "torque")).lower()
        if actuator_mode not in ("torque", "position"):
            raise ValueError(f"Unsupported actuator_mode={actuator_mode!r}. Expected 'torque' or 'position'.")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0



    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    if m.nu != num_actions:
        raise ValueError(f"Model nu={m.nu} does not match num_actions={num_actions}")

    mujoco_dof_names = get_actuated_joint_names_in_ctrl_order(m)
    if policy_dof_names is None:
        policy_dof_names = mujoco_dof_names
    if len(policy_dof_names) != num_actions:
        raise ValueError(f"policy_dof_names length {len(policy_dof_names)} != num_actions {num_actions}")

    mujoco_name_to_index = {name: i for i, name in enumerate(mujoco_dof_names)}
    policy_name_to_index = {name: i for i, name in enumerate(policy_dof_names)}
    missing_in_mujoco = [n for n in policy_dof_names if n not in mujoco_name_to_index]
    if missing_in_mujoco:
        raise ValueError(f"policy_dof_names contains names not in Mujoco actuators: {missing_in_mujoco}")

    # Map indices between policy order and Mujoco ctrl order
    mujoco_index_for_policy = np.array([mujoco_name_to_index[n] for n in policy_dof_names], dtype=np.int64)
    policy_index_for_mujoco = np.array([policy_name_to_index[n] for n in mujoco_dof_names], dtype=np.int64)

    # Precompute ctrl limits (if present)
    ctrl_min = None
    ctrl_max = None
    if hasattr(m, "actuator_ctrlrange") and m.actuator_ctrlrange is not None and m.actuator_ctrlrange.size:
        ctrl_min = m.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_max = m.actuator_ctrlrange[:, 1].astype(np.float32)

    print(f"[deploy_mujoco] xml: {xml_path}")
    print(f"[deploy_mujoco] policy: {policy_path}")
    print(f"[deploy_mujoco] actuated joints (ctrl order): {mujoco_dof_names}")
    if policy_dof_names != mujoco_dof_names:
        print(f"[deploy_mujoco] policy_dof_names: {policy_dof_names}")

    # load policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Read q/dq in actuator ctrl order (robust to joint ordering differences).
            q_mj, dq_mj = get_actuated_q_dq_in_ctrl_order(m, d)

            # PD in Mujoco ctrl order.
            target_mj = target_dof_pos[policy_index_for_mujoco]

            if actuator_mode == "position":
                ctrl = target_mj
                if ctrl_min is not None:
                    ctrl = np.clip(ctrl, ctrl_min, ctrl_max)
                d.ctrl[:] = ctrl
            else:
                kps_mj = kps[policy_index_for_mujoco]
                kds_mj = kds[policy_index_for_mujoco]
                tau = pd_control(target_mj, q_mj, kps_mj, np.zeros_like(kds_mj), dq_mj, kds_mj)
                if ctrl_min is not None:
                    tau = np.clip(tau, ctrl_min, ctrl_max)
                d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # Update commands from ROS1 /cmd_vel (vx, vy, wz) if available.
                if cmd_vel_receiver is not None:
                    cmd = cmd_vel_receiver.get_cmd()

                # Create observations in *policy dof order*.
                q_policy = q_mj[mujoco_index_for_policy]
                dq_policy = dq_mj[mujoco_index_for_policy]

                # Base state
                quat_wxyz = d.qpos[3:7].astype(np.float32)
                lin_vel_world = d.qvel[0:3].astype(np.float32)
                ang_vel_world = d.qvel[3:6].astype(np.float32)
                base_lin_vel = quat_rotate_inverse(quat_wxyz, lin_vel_world)
                base_ang_vel = quat_rotate_inverse(quat_wxyz, ang_vel_world)
                projected_gravity = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))

                # Scales
                qj = (q_policy - default_angles) * dof_pos_scale
                dqj = dq_policy * dof_vel_scale
                base_ang_vel = base_ang_vel * ang_vel_scale

                if num_obs == 48:
                    # Matches LeggedRobot.compute_observations
                    lin_vel_scale = config.get("lin_vel_scale", 2.0)
                    obs[0:3] = base_lin_vel * lin_vel_scale
                    obs[3:6] = base_ang_vel
                    obs[6:9] = projected_gravity
                    obs[9:12] = cmd * cmd_scale
                    obs[12 : 12 + num_actions] = qj
                    obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                    obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action
                else:
                    # Matches H1/G1/H1_2 style observations (no lin_vel, with gait phase)
                    period = 0.8
                    count = counter * simulation_dt
                    phase = count % period / period
                    sin_phase = np.sin(2 * np.pi * phase)
                    cos_phase = np.cos(2 * np.pi * phase)

                    obs[:3] = base_ang_vel
                    obs[3:6] = projected_gravity
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase], dtype=np.float32)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    action = policy(obs_tensor).cpu().numpy().squeeze().astype(np.float32)
                if action_clip is not None:
                    action = np.clip(action, -float(action_clip), float(action_clip))

                # Transform action (policy order) -> target dof positions (policy order)
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # time.sleep(10)
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

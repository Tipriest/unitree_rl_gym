import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time
import threading

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


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    commands = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device, requires_grad=False)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        commands[0] = torch.from_numpy(cmd_vel_receiver.get_cmd()) if cmd_vel_receiver is not None else 0.0
        print(f"Received command: {commands[0].cpu().numpy()} (last msg time: {cmd_vel_receiver.last_msg_time if cmd_vel_receiver is not None else 'N/A'})")
        obs[:, 9:12] = commands  # append cmd_vel to the end of the observation
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

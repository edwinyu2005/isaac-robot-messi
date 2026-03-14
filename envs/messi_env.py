"""
Milestone 3: MessiEnv RL Environment Class.
Encapsulates physics stepping, tensor extraction, and reward computation.
"""

import torch
from messi_utils.math_utils import get_relative_pos
from gymnasium.spaces import Box


class MessiEnv:
    def __init__(self, num_envs: int, device: str = "cuda:0"):
        """
        Initializes the vectorized RL environment.
        """
        self.num_envs = num_envs
        self.device = device

        # RL Space Definitions
        # Actions: 12 joint targets for Unitree Go2
        self.num_actions = 12
        # Obs: 12 joint pos, 12 joint vel, 3 base lin_vel, 3 base ang_vel,
        # 3 relative ball pos = 33 dimensions
        self.num_obs = 33

        self.observation_space = Box(low=-float("inf"), high=float("inf"), shape=(self.num_obs,))
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # Allocate PyTorch buffers for RL transitions
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.prims import RigidPrimView

        # Initialize Views (Assuming build_shootout_scene is already called)
        self.robot = ArticulationView(prim_paths_expr="/World/.*RobotMessi", name="robot_view")
        self.ball = RigidPrimView(prim_paths_expr="/World/.*Football", name="ball_view")

        # Target goal position in world frame (example: 10 meters forward)
        self.goal_pos = torch.tensor([[10.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)

    def reset_idx(self, env_ids: torch.Tensor):
        """
        Asynchronously resets specific environments that have terminated.
        """
        if len(env_ids) == 0:
            return

        # 1. Reset Robot to standing pose
        default_joints = torch.tensor([[0.0, 0.9, -1.8] * 4], device=self.device)
        self.robot.set_joint_positions(default_joints.repeat(len(env_ids), 1), indices=env_ids)

        # Reset velocities to zero
        self.robot.set_velocities(torch.zeros((len(env_ids), 6), device=self.device), indices=env_ids)
        self.robot.set_joint_velocities(torch.zeros((len(env_ids), 12), device=self.device), indices=env_ids)

        # 2. Randomize Ball Position (1m to 3m in front of robot)
        r = torch.rand(len(env_ids), device=self.device) * 2.0 + 1.0
        theta = (torch.rand(len(env_ids), device=self.device) - 0.5) * torch.pi

        ball_x = r * torch.cos(theta)
        ball_y = r * torch.sin(theta)
        ball_z = torch.ones_like(ball_x) * 0.5

        ball_pos = torch.stack([ball_x, ball_y, ball_z], dim=-1)
        self.ball.set_world_poses(positions=ball_pos, indices=env_ids)
        self.ball.set_velocities(torch.zeros((len(env_ids), 6), device=self.device), indices=env_ids)

        # Clear reset flags for these environments
        self.reset_buf[env_ids] = 0

    def step(self, actions: torch.Tensor):
        """
        Applies actions, steps physics, and computes new states.
        """
        # Apply PD control targets to the robot's joints
        self.robot.set_joint_position_targets(actions)

        # Physics engine steps forward (usually handled by the main training loop)
        # omni.isaac.core.World().step() is called outside this function

        # Compute new state
        self.compute_observations()
        self.compute_rewards()
        self.check_termination()

        return self.obs_buf, self.reward_buf, self.reset_buf

    def compute_observations(self):
        """
        Populates the observation buffer using ego-centric transformations.
        """
        robot_pos, robot_quat = self.robot.get_world_poses()
        ball_pos, _ = self.ball.get_world_poses()

        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        root_vel = self.robot.get_velocities()

        # Calculate ego-centric ball position using Milestone 2.2 math utils
        rel_ball_pos = get_relative_pos(robot_pos, ball_pos, robot_quat)

        # Concatenate into the 33-dimensional observation vector
        self.obs_buf = torch.cat([
            joint_pos,                  # [N, 12]
            joint_vel,                  # [N, 12]
            root_vel[:, :3],            # [N, 3] Linear velocity
            root_vel[:, 3:],            # [N, 3] Angular velocity
            rel_ball_pos                # [N, 3] Ego-centric ball position
        ], dim=-1)

    def compute_rewards(self):
        """
        Calculates the shaped reward function.
        """
        robot_pos, _ = self.robot.get_world_poses()
        ball_pos, _ = self.ball.get_world_poses()

        # 1. Distance Reward (Dense): Encourage moving towards the ball
        dist_to_ball = torch.norm(ball_pos - robot_pos, dim=-1)
        dist_reward = torch.exp(-dist_to_ball)

        # 2. Survival Penalty: Punish the robot if the base drops too low
        base_height = robot_pos[:, 2]
        fall_penalty = torch.where(base_height < 0.25, -10.0, 0.0)

        self.reward_buf = dist_reward + fall_penalty

    def check_termination(self):
        """
        Flags environments for reset if they fall or finish the task.
        """
        robot_pos, _ = self.robot.get_world_poses()
        base_height = robot_pos[:, 2]

        # Reset if the robot falls down
        has_fallen = base_height < 0.25

        # Update the reset buffer (1 means reset needed)
        self.reset_buf = torch.where(has_fallen, 1, 0)


# ======================================================================
# Sanity Check Execution Block (Alternative to standard pytest)
# ======================================================================
if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from omni.isaac.core import World
    from messi_utils.usd_builder import build_shootout_scene

    # Initialize a test world with 1 environment for the basic sanity check
    num_test_envs = 1
    world = World(backend="torch", device="cuda:0")

    # Call the existing scene builder without the num_envs argument
    build_shootout_scene(world)

    env = MessiEnv(num_envs=num_test_envs, device="cuda:0")
    world.scene.add(env.robot)
    world.scene.add(env.ball)
    world.reset()

    print("[INFO] Testing MessiEnv Initialization...")

    # Force a reset on all environments
    all_envs = torch.arange(num_test_envs, device="cuda:0")
    env.reset_idx(all_envs)

    # Step the environment 10 times with zero-actions
    dummy_actions = torch.zeros((num_test_envs, 12), device="cuda:0")
    for _ in range(10):
        world.step(render=False)
        obs, rew, reset = env.step(dummy_actions)

    print(f"Observation Tensor Shape : {obs.shape} (Expected: [{num_test_envs}, 33])")
    print(f"Reward Tensor Shape      : {rew.shape} (Expected: [{num_test_envs}])")
    print(f"Reset Buffer Shape       : {reset.shape} (Expected: [{num_test_envs}])")
    print("[INFO] Sanity check passed. Shutting down.")

    simulation_app.close()

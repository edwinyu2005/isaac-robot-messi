"""
Milestone 2: State Tensor Extraction.
Validates Zero-Copy GPU tensor extraction for the RL MDP.
"""

import os
import sys

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import ArticulationView  # noqa: E402
from omni.isaac.core.prims import RigidPrimView  # noqa: E402
from messi_utils.usd_builder import build_shootout_scene  # noqa: E402


def main():
    # 1. CRITICAL: Set backend to torch to output PyTorch tensors directly on GPU
    print("[INFO] Initializing World with PyTorch backend...")
    world = World(
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,
        backend="torch",
        device="cuda:0"
    )

    build_shootout_scene(world)

    # 2. Initialize Views for batched tensor extraction
    print("[INFO] Creating ArticulationView and RigidPrimView...")
    robot_view = ArticulationView(
        prim_paths_expr="/World/RobotMessi",
        name="robot_view"
    )
    ball_view = RigidPrimView(
        prim_paths_expr="/World/Football",
        name="ball_view"
    )

    world.scene.add(robot_view)
    world.scene.add(ball_view)

    world.reset()

    print("[INFO] Starting tensor extraction loop...")
    frame = 0
    max_frames = 60  # Run for exactly 1 second (60 frames)

    while simulation_app.is_running() and frame < max_frames:
        world.step(render=True)

        # 3. Extract tensors directly from the GPU solver
        # Returns: tensor shape [num_envs, 3] and [num_envs, 4]
        robot_pos, robot_quat = robot_view.get_world_poses()

        # Returns: tensor shape [num_envs, num_joints] (12 joints for Go2)
        joint_pos = robot_view.get_joint_positions()

        ball_pos, _ = ball_view.get_world_poses()

        # Print the raw PyTorch tensors every 10 frames
        if frame % 10 == 0:
            print(f"\n--- Frame {frame} ---")
            print(f"Robot Root Pos  : {robot_pos.shape} | {robot_pos}")
            print(f"Robot Joint Pos : {joint_pos.shape}")
            print(f"Ball Root Pos   : {ball_pos.shape} | {ball_pos}")

        frame += 1

    print("[INFO] Tensor extraction test complete. Shutting down...")
    simulation_app.close()


if __name__ == "__main__":
    main()

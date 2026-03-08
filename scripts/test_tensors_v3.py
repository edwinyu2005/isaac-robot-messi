"""
Milestone 2.3: Environment Reset & Randomization.
Implements batched pose resetting and random ball placement.
"""

import os
import sys
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.articulations import ArticulationView  # noqa: E402
from omni.isaac.core.prims import RigidPrimView  # noqa: E402
from messi_utils.usd_builder import build_shootout_scene  # noqa: E402


def main():
    world = World(backend="torch", device="cuda:0")
    build_shootout_scene(world)

    robot_view = ArticulationView(prim_paths_expr="/World/RobotMessi", name="robot_view")
    ball_view = RigidPrimView(prim_paths_expr="/World/Football", name="ball_view")

    world.scene.add(robot_view)
    world.scene.add(ball_view)
    world.reset()

    # Define initial standing pose for Go2 (from previous calibrations)
    # Using a 12-dim tensor for the 12 joints of Unitree Go2
    default_joint_pos = torch.tensor([[0.0, 0.9, -1.8] * 4], device="cuda:0")

    print("[INFO] Starting Reset & Randomization Test...")

    for episode in range(5):
        print(f"\n--- Resetting Episode {episode} ---")

        # 1. Reset Robot: Move back to origin and set to standing pose
        indices = torch.tensor([0], device="cuda:0", dtype=torch.long)

        robot_view.set_world_poses(
            positions=torch.tensor([[0.0, 0.0, 0.5]], device="cuda:0"),
            orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0"),
            indices=indices
        )
        robot_view.set_joint_positions(default_joint_pos, indices=indices)
        robot_view.set_velocities(torch.zeros((1, 6), device="cuda:0"), indices=indices)

        # 2. Randomize Ball: Place ball at random distance (1-3m) and angle
        r = torch.rand(1, device="cuda:0") * 2.0 + 1.0  # 1.0 to 3.0 meters
        theta = (torch.rand(1, device="cuda:0") - 0.5) * torch.pi  # -90 to 90 degrees

        ball_x = r * torch.cos(theta)
        ball_y = r * torch.sin(theta)

        ball_view.set_world_poses(
            positions=torch.cat([ball_x.view(1, 1), ball_y.view(1, 1), 
                                torch.tensor([[0.5]], device="cuda:0")], dim=-1),
            indices=indices
        )

        # Step simulation for 30 frames to observe the randomized state
        for _ in range(30):
            world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

"""
Milestone 2.2: Verifying Ego-Centric Transformation.
Calculates the ball's position relative to the robot's heading.
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
from messi_utils.math_utils import get_relative_pos  # noqa: E402


def main():
    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/60.0, backend="torch", device="cuda:0")
    build_shootout_scene(world)

    robot_view = ArticulationView(prim_paths_expr="/World/RobotMessi", name="robot_view")
    ball_view = RigidPrimView(prim_paths_expr="/World/Football", name="ball_view")

    world.scene.add(robot_view)
    world.scene.add(ball_view)
    world.reset()

    print("[INFO] Testing Relative Coordinate Transformation...")

    for i in range(100):
        world.step(render=True)

        # Get world states
        robot_pos, robot_quat = robot_view.get_world_poses()
        ball_pos, _ = ball_view.get_world_poses()

        # COMPUTE RELATIVE POSITION: This is what the RL agent actually "sees"
        relative_ball_pos = get_relative_pos(robot_pos, ball_pos, robot_quat)

        if i % 20 == 0:
            print(f"\nFrame {i}")
            print(f"Robot World Pos : {robot_pos[0].cpu().numpy()}")
            print(f"Ball World Pos  : {ball_pos[0].cpu().numpy()}")
            print(f"BALL RELATIVE TO ROBOT: {relative_ball_pos[0].cpu().numpy()}")

    simulation_app.close()


if __name__ == "__main__":
    main()

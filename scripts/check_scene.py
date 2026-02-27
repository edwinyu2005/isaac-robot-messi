"""
Main entry point to visually verify the shootout scene.
"""

import os
import sys

# 1. MUST instantiate SimulationApp BEFORE modifying sys.path.
# SimulationApp initialization overwrites the Python environment paths.
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

# 2. Inject project root at index 0 (VIP Priority).
# This prevents namespace shadowing (e.g., Omniverse's built-in 'utils' module
# blocking our custom imports).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from omni.isaac.core import World  # noqa: E402
from messi_utils.usd_builder import build_shootout_scene  # noqa: E402


def main():
    print("[INFO] Instantiating World object (120Hz physics, 60Hz render)...")
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

    print("[INFO] Building procedural USD scene...")
    robot, football = build_shootout_scene(world)

    print("[INFO] Starting simulation loop. Close the GUI to exit.")
    world.reset()

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

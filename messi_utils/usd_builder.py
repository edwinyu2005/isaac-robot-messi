"""
Procedural USD scene builder for the shootout environment.
"""

import numpy as np

# Retained omni modules (not yet fully migrated by NVIDIA in 4.5.0)
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import add_reference_to_stage

# Modern Isaac Sim 4.5.0 API namespace
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicSphere, FixedCuboid
from isaacsim.core.api.robots import Robot


def build_shootout_scene(world: World):
    """
    Constructs the scene: ground, Unitree Go2, football, and goal posts.
    """
    world.scene.add_default_ground_plane(
        prim_path="/World/defaultGroundPlane",
        z_position=0.0,
        name="default_ground_plane",
    )

    # Create a bouncy physics material for the football
    # TWEAK 1: Maximize restitution to 1.0 to compensate for the ground's 0.0 restitution
    bouncy_material = PhysicsMaterial(
        prim_path="/World/Physics_Materials/bouncy_material",
        restitution=1.0,
        dynamic_friction=0.5,
    )

    robot_usd_path = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.5/Isaac/Robots/Unitree/Go2/go2.usd"
    )

    # Isaac Sim 4.5.0 Breaking Change Fix:
    # Model loading and physics wrapping are now decoupled.
    # Step 1: Explicitly reference the USD asset onto the stage.
    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/RobotMessi")

    # Step 2: Wrap the existing prim with the Robot class for RL control.
    robot = world.scene.add(
        Robot(
            prim_path="/World/RobotMessi",
            name="robot_messi",
            # TWEAK 2: Lower z-position to 0.22m so it spawns resting on its belly
            position=np.array([-0.5, 0.0, 0.22])
        )
    )

    football = world.scene.add(
        DynamicSphere(
            prim_path="/World/Football",
            name="football",
            position=np.array([1.5, 0.0, 2.0]),
            radius=0.11,
            mass=0.43,
            color=np.array([1.0, 1.0, 1.0]),
            # Apply the physics material to enable bouncing
            physics_material=bouncy_material,
        )
    )

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Goal/PostLeft",
            name="post_left",
            position=np.array([3.5, 1.5, 0.5]),
            scale=np.array([0.1, 0.1, 1.0]),
            color=np.array([0.8, 0.1, 0.1]),
        )
    )

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Goal/PostRight",
            name="post_right",
            position=np.array([3.5, -1.5, 0.5]),
            scale=np.array([0.1, 0.1, 1.0]),
            color=np.array([0.8, 0.1, 0.1]),
        )
    )

    return robot, football

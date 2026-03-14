"""
Milestone 3.3: Local Sandbox Training Script.
Wires up MessiEnv with skrl's PPO algorithm for a local smoke test.
"""

import os
import sys
import torch

# 1. Start SimulationApp FIRST
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# 2. Import Physics & Environment
from omni.isaac.core import World
from messi_utils.usd_builder import build_shootout_scene
from envs.messi_env import MessiEnv

# 3. Import skrl modules
from skrl.envs.wrappers.torch import Wrapper
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory


# ======================================================================
# Environment Wrapper: Adapts MessiEnv to skrl API
# ======================================================================
class SkrlMessiWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        # Physics step
        World().step(render=True)

        obs, rew, reset = self._env.step(actions)

        # Handle async resets
        reset_indices = reset.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_indices) > 0:
            self._env.reset_idx(reset_indices)

        # skrl expects: obs, reward, terminated, truncated, info
        truncated = torch.zeros_like(reset)
        return obs, rew, reset, truncated, {}

    def reset(self):
        all_envs = torch.arange(self.num_envs, device=self.device)
        self._env.reset_idx(all_envs)
        World().step(render=True)
        self._env.compute_observations()
        return self._env.obs_buf, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


# ======================================================================
# Neural Networks: Policy (Actor) and Value (Critic)
# ======================================================================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ======================================================================
# Main Training Loop
# ======================================================================
def main():
    world = World(backend="torch", device="cuda:0")
    build_shootout_scene(world)  # 1 environment for smoke test

    base_env = MessiEnv(num_envs=1, device="cuda:0")
    world.scene.add(base_env.robot)
    world.scene.add(base_env.ball)
    world.reset()

    # Wrap the environment
    env = SkrlMessiWrapper(base_env)

    # Instantiate the networks
    models = {
        "policy": Policy(env.observation_space, env.action_space, env.device),
        "value": Value(env.observation_space, env.action_space, env.device)
    }

    # Setup PPO agent configuration for quick local testing
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 128           # Data collected before updating
    cfg["learning_epochs"] = 4      # Number of passes over the data
    cfg["mini_batches"] = 2         # Minibatches per epoch
    cfg["discount_factor"] = 0.99
    cfg["learning_rate"] = 1e-3
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0

    memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=env.device)

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

    # Start training! (2000 timesteps is a short 30-60 second test)
    trainer = SequentialTrainer(cfg={"timesteps": 2000, "headless": False},
                                env=env,
                                agents=agent)
    print("[INFO] Starting PPO Local Smoke Test...")
    trainer.train()

    print("[INFO] Smoke test completed successfully!")
    simulation_app.close()


if __name__ == "__main__":
    main()

# Milestone 3: Reward Design & Training

## 3.1 Environment Class Integration
We have successfully encapsulated the physics pipeline, zero-copy tensor extraction, and randomized reset logic into a standard RL environment class (`envs/messi_env.py`).

### Key Implementation
- **Action Space**: 12-dimensional continuous tensor for joint-level PD control targets.
- **Observation Space**: 33-dimensional tensor combining robot proprioception (joint states, root velocities) and ego-centric ball coordinates.
- **Asynchronous Reset**: Implemented `reset_idx()` using `torch.long` tensors to reset only terminated environments without blocking parallel execution.

## 3.2 Local Sandbox Verification
We successfully integrated the custom `MessiEnv` with the `skrl` reinforcement learning library, establishing a complete end-to-end training pipeline for local hardware (RTX 3070).

### Key Implementation
- **skrl Integration**: Created `scripts/train_local.py` to instantiate the physics world and the PPO agent.
- **Environment Wrapper**: Implemented `SkrlMessiWrapper` to bridge Isaac Sim's asynchronous step logic with standard Gymnasium `Box` spaces and `skrl`'s API requirements (handling read-only properties and render flags).
- **PPO Configuration**: Configured a baseline PPO agent with simple MLP networks (Policy and Value) and a `RandomMemory` buffer (128 rollouts) to verify gradient computation.
- **Outcome**: Successfully completed a 2000-timestep smoke test, proving that physics simulation, tensor-based observations, action application, and neural network backpropagation are fully synchronized.

Run the local training script:
```bash
~/isaacsim/python.sh scripts/train_local.py
```

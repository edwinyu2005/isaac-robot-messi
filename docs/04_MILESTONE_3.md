# Milestone 3: Reward Design & Training

## 3.1 Environment Class Integration
We have successfully encapsulated the physics pipeline, zero-copy tensor extraction, and randomized reset logic into a standard RL environment class (`envs/messi_env.py`).

### Key Implementation
- **Action Space**: 12-dimensional continuous tensor for joint-level PD control targets.
- **Observation Space**: 33-dimensional tensor combining robot proprioception (joint states, root velocities) and ego-centric ball coordinates.
- **Asynchronous Reset**: Implemented `reset_idx()` using `torch.long` tensors to reset only terminated environments without blocking parallel execution.

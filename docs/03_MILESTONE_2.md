# Milestone 2: MDP & Tensor Pipeline

## 2.1 State Tensor Extraction
We have successfully transitioned from CPU-based logic to a **massively parallel GPU tensor pipeline**. By setting the world backend to `torch`, we extract physics states directly into PyTorch tensors without memory copying.

### Verification
Run the tensor extraction test to verify GPU data flow:
```bash
~/isaacsim/python.sh scripts/test_tensors.py
```
*Expected Output: Tensors with `device='cuda:0'` showing the robot's root pose and the football's bouncing trajectory.*

## 2.2 Ego-Centric Transformation
We implemented high-performance batched math utilities to transform global world coordinates into the robot's local frame. This ensures the RL agent perceives the environment relative to its own heading.

### Key Implementation
- **`quat_rotate_inverse`**: Rotates world vectors to local frame using quaternion conjugates.
- **`get_relative_pos`**: Computes relative displacement vectors for observations.

### Verification
Run the updated tensor test to verify spatial coordinate flipping:
```bash
~/isaacsim/python.sh scripts/test_tensors_v2.py
```
*Expected Output: Relative X-coordinates should correctly flip to negative values if the robot orientation is reversed relative to the ball.*

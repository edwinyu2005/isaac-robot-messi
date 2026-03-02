# Milestone 2: MDP & Tensor Pipeline

## 2.1 State Tensor Extraction (Completed)
We have successfully transitioned from CPU-based logic to a **massively parallel GPU tensor pipeline**. By setting the world backend to `torch`, we extract physics states directly into PyTorch tensors without memory copying.

### Verification
Run the tensor extraction test to verify GPU data flow:
```bash
~/isaacsim/python.sh scripts/test_tensors.py
```
*Expected Output: Tensors with `device='cuda:0'` showing the robot's root pose and the football's bouncing trajectory.*

## 2.2 Ego-Centric Transformation (In Progress)
The next step involves transforming global world coordinates into the robot's local frame. This is critical for RL observations, ensuring the agent learns relative spatial relationships.

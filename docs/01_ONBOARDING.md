# Onboarding & Setup Guide

## 1. Prerequisites
Unlike standard Python projects, **DO NOT use `venv` or `conda`**. NVIDIA Isaac Sim is a standalone 3D rendering and physics engine. You must install the core engine before running any scripts.

1. **Install Isaac Sim:** Go to [NVIDIA Isaac Sim 4.5.0 Download](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html), download and unzip **Isaac Sim 4.5**.
2. **Hardware Requirements:** * **Local Debugging:** NVIDIA RTX GPU (Minimum 8GB VRAM).
   * **Cloud Training:** NVIDIA RTX GPU (24GB+ VRAM for massive parallelism).
   * **OS:** Ubuntu 20.04/22.04 LTS.

## 2. Environment Setup
Install the external RL dependencies (like `skrl` and `tensorboard`) directly into Isaac Sim's embedded Python environment:

```bash
# Point the script to your specific Isaac Sim 4.5 path
./setup.sh ~/isaacsim # or other path
```

## 3. Critical Implementation Notes (Lessons Learned)

### Namespace Collisions
Isaac Sim contains a massive built-in Python environment, including a default `utils` module. To avoid `ModuleNotFoundError` when importing custom files, we renamed our utilities directory to `messi_utils` and strictly prioritize our project root in `sys.path`:

```python
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Use insert(0) to bypass Omniverse built-in module collisions
sys.path.insert(0, project_root) 
```

### Isaac Sim 4.5.0 API Breaking Changes
NVIDIA is actively migrating core modules to a new namespace. To avoid deprecation warnings and potential crashes (such as `Robot.__init__` losing the `usd_path` argument), always use the modern API:

* **Deprecated Approach:** `omni.isaac.core.robots` and `omni.isaac.core.materials`
* **Modern Approach (4.5.0+):** `isaacsim.core.api.robots` and `isaacsim.core.api.materials`

Asset loading must now be explicitly decoupled from physics wrapping using `add_reference_to_stage`.

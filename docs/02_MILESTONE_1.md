# Milestone 1: Physics & Visualization

## 1. Running the Scene Check
Before diving into reinforcement learning, verify that the local physics engine (PhysX) and the OpenUSD procedural generation are functioning correctly.

Run the following command from the project root:

```bash
# Execute the scene check script using Isaac Sim's bundled Python
~/isaacsim/python.sh scripts/check_scene.py
```

**Expected Output:** The Isaac Sim GUI will launch. A Unitree Go2 robot will spawn on the ground, and a football will drop from the sky, bouncing with realistic restitution. The simulation will auto-terminate after 5 seconds.

## 2. Generating High-Quality Demo GIFs
We use a custom FFmpeg CLI tool to convert raw `.webm` screen recordings (captured via Ubuntu's native screen recorder) into optimized, lightweight GIFs suitable for GitHub PRs.

### Usage Example
Use the `make_gif.py` script to crop the Isaac Sim viewport, slow down the playback, and compress the file size:

```bash
# Convert webm to an optimized, cropped, and slow-motion GIF
python3 scripts/make_gif.py \
    --input "~/Videos/recording.webm" \
    --output "media/milestone_1.gif" \
    --start "00:00:16" \
    --end "00:00:18" \
    --speed 0.5 \
    --target_w 720 \
    --fps 10
```

*Tip: Slowing playback to 0.5x is highly recommended for visualizing high-speed contact physics during set-piece execution.*

# EyeFollowers - Boids Simulation

Eye-tracking controlled boids simulation with dynamic performance optimization.

## Features

- **Spatial Partitioning**: O(n) neighbor lookups using grid-based spatial hashing (vs O(n²))
- **Dynamic Boid Count**: Automatically adjusts boid count to maintain 60 FPS
- **Optimized Rendering**: Batch rendering with vectorized NumPy operations
- **Adaptive Performance**: Scales from 50 to 2000+ boids based on system capability

## Performance Optimizations

1. **Spatial Grid**: Divides screen into cells for fast neighbor queries
2. **Vectorized Math**: NumPy operations for position/velocity updates
3. **Smart Gaze Sampling**: Updates gaze every 2 frames instead of every frame
4. **Dynamic Scaling**: Adds/removes boids in real-time to hit FPS target

## Files

- `boids.py` - Main simulation with all optimizations
- `gaze.py` - Eye tracking implementation
- `calibration.py` - Calibration UI for gaze tracker

## Usage

```bash
python boids.py
```

## Controls

- `c` - Toggle gaze cursor
- `s` - Toggle stats display
- `q` or `ESC` - Quit

## Requirements

- pygame
- opencv-python
- numpy
- mediapipe

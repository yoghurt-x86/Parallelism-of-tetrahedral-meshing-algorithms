# libigl + HIP GPU Acceleration Project

## Standard Build (CPU only)
```bash
nix develop
cmake .
make example
./libigl/example
```

## GPU-Accelerated Build (with HIP)
```bash
nix develop .#hip
cmake .
make example
./libigl/example
```

## Features
- **CPU Mode**: Standard libigl geometry processing
- **GPU Mode**: GPU-accelerated vertex operations using HIP
  - Scale vertices on GPU
  - Translate vertices on GPU
  - Interactive UI buttons for testing GPU functions

## Requirements
- Nix with flakes enabled
- For GPU acceleration: AMD GPU with ROCm support

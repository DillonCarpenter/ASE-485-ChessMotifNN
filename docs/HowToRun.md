---
marp: true
theme: default
paginate: true
---

# Developer Setup Guide

## Environment Overview

- VS Code
- Docker & Dev Containers
- WSL2 (Windows)
- PyTorch (CUDA optional)

---

# Clone Repository

```bash
git clone https://github.com/DillonCarpenter/ASE-485-ChessMotifNN
cd <repo-name>
```

Move to WSL home for better performance (recommended on Windows):

```bash
cd ~
git clone https://github.com/DillonCarpenter/ASE-485-ChessMotifNN
```
---
# Docker + WSL2 (Windows)

1. Install Docker Desktop
2. Enable WSL2 backend
3. Enable integration with your Linux distro
4. Verify:

```bash
docker --version
```

---

# VS Code Setup

1. Install VS Code
2. Install **Dev Containers** extension
3. Open project folder
4. Reopen in container:
   - Ctrl + Shift + P
   - Dev Containers: Reopen in Container
   - Can also access through command panel in view

---

# Running the Project

Inside Dev Container (recommended):

1. Navigate to src/Chess_Motif_NN/small_chess_network.py
2. Run and debug using Python

---

# PyTorch & CUDA

- CUDA available → GPU acceleration
- Not available → CPU fallback
- No extra config needed

Check availability:

```python
import torch
print(torch.cuda.is_available())
```

Or if you are running the container, torch is already available

---

# Troubleshooting

- Docker not found → restart Docker/PC
- Dev container fails → check logs
- GPU not detected → ensure drivers

---


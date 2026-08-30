# Two Towers and Beyond - Unified TAMP System

**A complete Task and Motion Planning (TAMP) system for robotic block manipulation**

---

## Quick Start

### Run Any Goal

```bash
# Goal 1: Build two towers (RED-GREEN-BLUE, YELLOW-MAGENTA-CYAN)
python3 demo.py --goal 1

# Goal 1 from Scene 2 (pre-stacked tower)
python3 demo.py --goal 1 --scene 2

# Goal 2: Build single 5-block tower (top→bottom: MAGENTA-YELLOW-BLUE-RED-GREEN)
python3 demo.py --goal 2

# Goal 2 from Scene 2 (pre-stacked tower)
python3 demo.py --goal 2 --scene 2

# Goal 3: Build 8-block tower (tallest possible)
python3 demo.py --goal 3

# Goal 4: Build yellow cross + green hollow square (18 blocks)
python3 demo.py --goal 4
```



##  Installation

### Prerequisites

1. **Create conda environment:**
   ```bash
   conda create -y -n rbe550 python=3.11
   conda activate rbe550
   ```

2. **Install PyTorch** (with CUDA 13.0 support):
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

3. **Install OMPL Python bindings:**
   - Download from [OMPL releases](https://github.com/ompl/ompl/releases) for your system
   - Install: `pip install ompl-1.7.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`

4. **Install Genesis:**
   ```bash
   pip install genesis-world
   ```

5. **Install pyperplan:**
   ```bash
   pip install pyperplan
   ```



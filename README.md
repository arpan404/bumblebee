# 🐝 Bumblebee – Human-Like Mouse & Keyboard Controller

Bumblebee is an AI-powered Python package that provides **realistic, human-like control** of the mouse and keyboard. Unlike traditional automation tools, Bumblebee leverages deep learning to predict **natural movement patterns**, resulting in automated interactions that feel organic and smooth.

---

## ✨ Features

- **AI-Powered Cursor Movement:**  
  Utilizes an RNN augmented with an LSTM layer to generate smooth, human-like mouse trajectories. These paths are enhanced with natural noise and variable speeds, creating movements that closely mimic real human behavior.

- **Smart Keystroke Simulation:**  
  Simulates typing with natural delays, realistic punctuation handling, and variability in keystroke timing to emulate human typing patterns.

- **Reinforcement Learning (Coming Soon):**  
  Future updates will incorporate RL to further enhance cursor behavior, enabling more diverse and adaptive movement patterns.
---

## 🚀 How It Works

### 🖱️ Cursor Movement

Bumblebee predicts the cursor path using an RNN-LSTM model trained on extensive real-world cursor data. This approach avoids the robotic jumps seen in conventional automation tools, ensuring smooth and natural motion.

Currently, Bumblebee's mouse moves on almost the same path if the same initial and destination positions are provided, with some noise.

> **Coming Soon:**  An enhanced model incorporating Reinforcement Learning (RL) that will generate different form of paths even if same initial and destination is provided.

### ⌨️ Keyboard Control

The keyboard simulation is driven by:
- **Mathematical Models:**  
  Considering distances between keys and natural typing rhythms.
- **Timing Variations:**  
  Simulated delays for each keystroke to mimic human typing quirks.
- **Punctuation & Special Characters Handling:**  
  Adjusted behaviors to reflect realistic typing patterns.

### 🏗️ Under the Hood

- **Simulation:**  
  Uses [PyAutoGUI](https://pyautogui.readthedocs.io/) for executing mouse events and [pynput] (https://pynput.readthedocs.io/) for executing keyboard events.
- **Deep Learning:**  
  Powered by PyTorch models trained on self-collected datasets of human interactions for predicting intermediate points between starting position and destination.

---

## ⚙️ How to Use

1. Install Bumblebee

```bash
uv add the-bumblebee
```

For a local checkout, install with:

```bash
uv sync
```
2. Import the core modules and use them as following:

#### Mouse Control Examples

```python
from bumblebee import Mouse

# Initialize controllers
mouse = Mouse()


# Set mouse speed (pixels per second)
mouse.set_speed(1000)

# Move mouse to specific coordinates
mouse.move(100, 200)

# Drag mouse from the current position to a new location
mouse.drag_to(233, 244)

# Simulate a mouse click (button options: "left", "middle", "right", "primary", "secondary")
mouse.click(button="left")

# Move to a position then click
mouse.move_to_and_click(destX=150, destY=250, button="left")
```

#### Keyboard Control Example

Initialize the Keyboard controller and adjust its typing behavior:

```python
from bumblebee import Keyboard

# Initialize the Keyboard controller with default parameters:
#   - typing_speed: 100 (default percentage of ideal speed)
#   - consistency: 95 (default consistency percentage, affecting delay variability)
#   - typo_rate: 5 (default typo frequency percentage)
keyboard = Keyboard()

# Adjust typing speed.
# The typing speed is provided as a percentage relative to the original Bumblebee typing speed.
# Acceptable types: int or float.
keyboard.set_speed(new_speed=400)  # Increase the typing speed to 400%

# Adjust typo rate.
# The new typo rate must be a value between 0 and 100 (int or float).
keyboard.set_typo_rate(3)  # Set typo rate to 3%

# Adjust consistency.
# A higher consistency (0-100) increases typing speed with less delay variability.
keyboard.set_consistency(99)  # Set consistency to 99%

# Type a sample text.
text = "Bumblebee is great."
keyboard.type(text)  # 'text' must be a string.
```

---

## 🛠️ Development

This repository uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv sync --group dev
uv run python -c "import bumblebee; print(bumblebee.__file__)"
uv run black src
uv run isort src
```

Runtime package code lives in `src/bumblebee`. Training notebooks and local datasets are intentionally not stored in this repository while the next model architecture is being developed.

---

### Interested in contributing? 
Please take a moment to review our [CONTRIBUTING.md](https://github.com/socioy/bumblebee/blob/master/CONTRIBUTING.md) file for detailed guidelines on how to join our community and help shape Bumblebee’s future.

🐝 **Making automation feel more human, one movement at a time.**

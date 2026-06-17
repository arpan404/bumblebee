# Examples

## Real display smoke test

Move the real cursor around your current display with the packaged RL model:

```python
import time
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

print("Starting in 3 seconds. Move cursor to a corner to abort.")
time.sleep(3)

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider, fail_safe=True)

width, height = mouse.screen_size()
points = [
    (width * 0.35, height * 0.35),
    (width * 0.65, height * 0.35),
    (width * 0.65, height * 0.65),
    (width * 0.35, height * 0.65),
]

for x, y in points:
    mouse.move(x, y)
    time.sleep(0.3)
```

Safety:

- PyAutoGUI fail-safe is enabled.
- Move the cursor to a screen corner to abort.
- Do not run keyboard examples unless a safe text field is focused.

## Basic mouse automation

```python
from bumblebee import Mouse

mouse = Mouse(fail_safe=True)
mouse.set_speed(1200)
mouse.move(400, 300)
mouse.click()
mouse.move_relative(50, 0)
mouse.double_click()
```

## Click inside a detected UI bounding box

```python
from bumblebee import Mouse, MouseBounds

mouse = Mouse()
button = MouseBounds(100, 200, 260, 240)
clicked = mouse.click_in_bounds(button, padding=6)
print("clicked", clicked)
```

If your detector returns `x, y, width, height`:

```python
mouse.click_in_rect(100, 200, 160, 40, padding=6)
```

## Packaged RL mouse path

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

## Custom path provider

```python
import numpy as np
from bumblebee import Mouse


def curved_path(start: np.ndarray, destination: np.ndarray) -> np.ndarray:
    midpoint = (start + destination) / 2
    midpoint[1] -= 80
    return np.array([start, midpoint, destination])


mouse = Mouse(path_provider=curved_path)
mouse.move(700, 450)
```

## Execute a precomputed path

```python
import numpy as np
from bumblebee import Mouse

path = np.array([
    [100, 100, 1.0],
    [180, 120, 0.9],
    [260, 180, 0.8],
    [340, 240, 1.0],
])

mouse = Mouse()
mouse.move_path(path, destination=(340, 240))
```

## Keyboard typing

```python
from bumblebee import Keyboard

keyboard = Keyboard()
keyboard.set_profile("careful")
keyboard.type("Hello from Bumblebee.", wpm=70, typo_rate=1)
```

## Keyboard shortcuts

```python
keyboard.hotkey("ctrl", "a")
keyboard.copy()
keyboard.paste()
keyboard.undo()
keyboard.redo()
```

Use `cmd` instead of `ctrl` for explicit macOS shortcuts:

```python
keyboard.hotkey("cmd", "a")
```

## Type multiple lines

```python
keyboard.type_lines([
    "First line",
    "Second line",
    "Third line",
])
```

## Type or paste long text

```python
keyboard.type_or_paste("A long body of text...", threshold=500)
```

## Fake-controller test pattern

```python
import random
from bumblebee import Mouse, Keyboard


class FakeMouse:
    MINIMUM_DURATION = 0
    PAUSE = 0
    FAILSAFE = True

    def __init__(self):
        self.xy = (0, 0)
        self.events = []

    def position(self):
        return self.xy

    def size(self):
        return (1000, 800)

    def moveTo(self, x, y, duration=0):
        self.xy = (x, y)
        self.events.append(("moveTo", x, y, duration))

    def click(self, **kwargs):
        self.events.append(("click", kwargs))

    def dragTo(self, x, y, duration=0, button="left"):
        self.xy = (x, y)
        self.events.append(("dragTo", x, y, duration, button))

    def mouseDown(self, button="left"):
        self.events.append(("mouseDown", button))

    def mouseUp(self, button="left"):
        self.events.append(("mouseUp", button))

    def scroll(self, clicks):
        self.events.append(("scroll", clicks))


class FakeKeyboard:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(("press", str(key)))

    def release(self, key):
        self.events.append(("release", str(key)))


mouse_controller = FakeMouse()
mouse = Mouse(controller=mouse_controller, rng=random.Random(1), sleep=lambda _: None)
mouse.move(100, 100)
print(mouse_controller.events)

keyboard_controller = FakeKeyboard()
keyboard = Keyboard(
    controller=keyboard_controller,
    rng=random.Random(1),
    sleep=lambda _: None,
)
keyboard.type("Hello", typo_rate=0)
print(keyboard_controller.events)
```

# Mouse API

The mouse runtime API is in `bumblebee.mouse` and is exported from `bumblebee`.

```python
from bumblebee import Mouse, MouseBounds
```

## Safety

`Mouse(fail_safe=True)` is the default. PyAutoGUI's fail-safe lets you move the cursor to a screen corner to abort automation.

```python
mouse = Mouse(fail_safe=True)
```

Use automation only on systems and applications you are allowed to control.

## Basic movement

```python
from bumblebee import Mouse

mouse = Mouse()
mouse.set_speed(1200)        # pixels per second
mouse.move(500, 300)         # absolute move
mouse.move_to(700, 450)      # alias for move
mouse.move_relative(25, -10)
```

## Clicks

```python
mouse.click()
mouse.click(button="right")
mouse.double_click()
mouse.right_click()
mouse.click_at(300, 400)
mouse.click_at(300, 400, clicks=2, interval=0.08)
mouse.move_to_and_click(500, 300)
mouse.move_to_and_double_click(500, 300)
```

Valid buttons:

```text
left, middle, right, primary, secondary
```

## Drag and scroll

```python
mouse.drag_to(800, 500)
mouse.drag_relative(100, 0)
mouse.drag_to(800, 500, human_like=False)
mouse.scroll(-3)  # negative usually scrolls down
```

Low-level button controls:

```python
mouse.mouse_down("left")
mouse.mouse_up("left")
```

## Profiles

```python
mouse.set_profile("precise")
mouse.set_profile("fast")
mouse.set_profile("natural")
mouse.set_profile("messy")
```

Available profiles:

| Profile | Purpose |
| --- | --- |
| `default` | Safe, direct path, no added jitter/curve. |
| `precise` | Slower, lower speed variation. |
| `fast` | Faster moves with fewer generated points. |
| `natural` | Adds mild curve and jitter. |
| `messy` | Adds larger curve/jitter and timing variation. |

Create a temporary profile clone:

```python
fast_mouse = mouse.with_profile("fast")
fast_mouse.move(1000, 600)
```

Modify the active profile:

```python
mouse.configure_profile(target_radius_px=10, max_segment_distance_px=8)
```

## Random bounded clicks

Use `MouseBounds` when an AI detector or UI locator returns a rectangle and any point inside the rectangle is acceptable.

```python
from bumblebee import Mouse, MouseBounds

mouse = Mouse()
bounds = MouseBounds(100, 200, 260, 240)  # left, top, right, bottom

point = mouse.random_point_in_bounds(bounds, padding=6)
mouse.move_to_random_in_bounds(bounds, padding=6)
mouse.click_in_bounds(bounds, padding=6)
```

Using `x, y, width, height` rectangles:

```python
point = mouse.random_point_in_rect(100, 200, 160, 40, padding=6)
mouse.move_to_random_in_rect(100, 200, 160, 40, padding=6)
mouse.click_in_rect(100, 200, 160, 40, padding=6)
```

Notes:

- `padding` shrinks the rectangle before sampling.
- `clamp_to_screen=True` clips the rectangle to the current screen.
- If the bounds do not intersect the screen, Bumblebee raises `ValueError`.

## Paths

### Preview a path

```python
path = mouse.generate_path(500, 600)
print(path.shape)  # (N, 3)
```

The third column is a speed factor.

### Execute a custom path

```python
import numpy as np

path = np.array([
    [100, 100],
    [200, 130],
    [300, 240],
    [400, 300],
])

mouse.move_path(path)
```

Accepted shapes:

- `(N, 2)` -> Bumblebee adds speed factors.
- `(N, 3)` -> third column is used as speed factor.

### Destination correction

When a destination is known, Bumblebee protects runtime execution:

- If the path reaches/crosses the target radius early, later points are trimmed.
- If the final point is inside the target radius, it is snapped to the exact destination.
- If the path misses, a final correction segment is appended.

```python
mouse.move_path(path, destination=(500, 600))
```

You can override this per call:

```python
mouse.move_path(
    path,
    destination=(500, 600),
    force_destination=True,
    trim_after_target_reached=True,
    target_radius_px=8,
)
```

### Path simplification and densification

Bumblebee prepares paths before sending them to PyAutoGUI:

- Very dense points are simplified using `min_segment_distance_px`.
- Very long segments are densified using `max_segment_distance_px`.
- Final destination is preserved.

Defaults:

```text
min_segment_distance_px = 5.0
max_segment_distance_px = 8.0
```

This keeps paths smooth without calling PyAutoGUI for every tiny point.

## Packaged RL model as path provider

Requires `the-bumblebee[rl]` or `uv sync --group train` in a checkout.

```python
from bumblebee import Mouse
from bumblebee.rl.policy import SB3MousePolicyPathProvider

provider = SB3MousePolicyPathProvider.from_packaged(deterministic=False)
mouse = Mouse(path_provider=provider)
mouse.move(700, 450)
```

The model produces a path. The runtime mouse module then validates, densifies, trims/snaps/corrects, and executes that path.

## Custom path provider

A path provider receives `start` and `destination` NumPy arrays and returns `(N, 2)` or `(N, 3)` points.

```python
import numpy as np
from bumblebee import Mouse


def path_provider(start: np.ndarray, destination: np.ndarray) -> np.ndarray:
    midpoint = (start + destination) / 2
    return np.array([start, midpoint, destination])


mouse = Mouse(path_provider=path_provider)
mouse.move(700, 450)
```

## Test-friendly usage

Mouse accepts dependency injection:

```python
mouse = Mouse(controller=fake_controller, rng=random.Random(1), sleep=lambda _: None)
```

The controller must provide the PyAutoGUI methods Bumblebee calls, such as `position()`, `size()`, `moveTo()`, `click()`, `dragTo()`, `mouseDown()`, `mouseUp()`, and `scroll()`.

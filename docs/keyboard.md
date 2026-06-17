# Keyboard API

The keyboard runtime API is in `bumblebee.keyboard` and is exported from `bumblebee`.

```python
from bumblebee import Keyboard
```

## Basic typing

```python
keyboard = Keyboard()
keyboard.type("Hello, Bumblebee!")
keyboard.write("write() is an alias for type().")
```

Configure typing behavior:

```python
keyboard.set_speed(180)        # percent of default speed
keyboard.set_wpm(75)           # words per minute
keyboard.set_consistency(96)   # higher means less timing variance
keyboard.set_typo_rate(2)      # percent
```

Per-call overrides:

```python
keyboard.type(
    "Careful typing.",
    wpm=70,
    typo_rate=1,
    correct_typos=True,
    pause_at_punctuation=True,
)
```

## Profiles

```python
keyboard.set_profile("fast")
keyboard.set_profile("careful")
keyboard.set_profile("messy")
keyboard.set_profile("developer")
```

Available profiles:

| Profile | Purpose |
| --- | --- |
| `default` | Balanced speed, consistency, and typo rate. |
| `fast` | Faster typing with low typo rate. |
| `careful` | Slower, very consistent, very low typo rate. |
| `messy` | Higher typo rate and less consistent timing. |
| `developer` | Good defaults for code/punctuation-heavy input. |

## Low-level keys

```python
keyboard.press("shift")
keyboard.release("shift")
keyboard.tap("enter")
keyboard.tap("backspace", times=3)
keyboard.hotkey("ctrl", "c")
keyboard.hotkey("cmd", "a")  # macOS
```

Common key aliases include:

```text
ctrl, control, cmd, command, shift, alt, option, enter, return,
esc, escape, tab, space, backspace, delete, up, down, left, right,
home, end, page_up, page_down
```

## Convenience controls

```python
keyboard.enter()
keyboard.tab()
keyboard.escape()
keyboard.backspace(times=2)
keyboard.delete(times=2)
keyboard.space(times=2)
keyboard.arrow("left", times=3)
keyboard.up()
keyboard.down()
keyboard.left()
keyboard.right()
```

## Editing helpers

```python
keyboard.select_all()
keyboard.copy()
keyboard.cut()
keyboard.paste()
keyboard.undo()
keyboard.redo()
keyboard.clear()
keyboard.delete_word()
keyboard.move_to_line_start()
keyboard.move_to_line_end()
```

`primary_modifier` defaults to Command on macOS and Control elsewhere. You can override it:

```python
keyboard = Keyboard(primary_modifier="ctrl")
keyboard.set_primary_modifier("cmd")
```

## Multiple lines

```python
keyboard.type_lines(["First line", "Second line"])
keyboard.type_lines(["Message", "Submitted"], submit=True)
```

## Clipboard-assisted text

Long text can be pasted instead of typed character-by-character.

```python
keyboard.type_or_paste("Long text goes here", threshold=500)
```

Direct clipboard paste:

```python
keyboard.paste_text("Text copied to clipboard and pasted", restore_clipboard=True)
```

Notes:

- Clipboard helpers temporarily replace the clipboard.
- `restore_clipboard=True` restores the previous clipboard contents after pasting.
- Clipboard helpers use `pyperclip`.

## Typo behavior

When typo correction is enabled, Bumblebee may type a nearby wrong key, pause, press backspace, and then type the correct key.

```python
keyboard.type("Typo correction demo.", typo_rate=10, correct_typos=True)
```

To intentionally leave typos uncorrected:

```python
keyboard.type("Messy demo.", typo_rate=10, correct_typos=False)
```

## Test-friendly usage

Keyboard accepts a fake controller, RNG, and sleep function:

```python
import random
from bumblebee import Keyboard


class FakeController:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(("press", str(key)))

    def release(self, key):
        self.events.append(("release", str(key)))


fake = FakeController()
keyboard = Keyboard(controller=fake, rng=random.Random(1), sleep=lambda _: None)
keyboard.type("Hello", typo_rate=0)
print(fake.events)
```

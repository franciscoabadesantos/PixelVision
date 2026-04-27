# HSV Reference

OpenCV uses HSV with:

- `H`: 0-179
- `S`: 0-255
- `V`: 0-255

## Common Starting Ranges

Red or orange rooftops:

```python
import numpy as np

lower = np.array([0, 30, 30])
upper = np.array([20, 255, 255])
```

Alternative red range:

```python
lower = np.array([140, 30, 30])
upper = np.array([180, 255, 255])
```

Vegetation exclusion:

```python
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])
```

Water exclusion:

```python
lower = np.array([100, 50, 50])
upper = np.array([140, 255, 255])
```

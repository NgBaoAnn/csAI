# ⚡ NumPy Optimization (Python)

This skill provides advanced NumPy techniques to accelerate grid-based games and AI agents. Essential for meeting the 1s decision time in the Hide and Seek Arena.

## 🚀 When to Use
- Managing 21x21 grid states.
- Pre-calculating distances (Dijkstra, Distance Transform).
- Fast collision checks.
- Eliminating Python `for` loops in grid operations.

## 📋 Optimization Techniques

### 🟢 Pre-calculating Distance Maps (Important!)
Use Dijkstra once for the static map (walls) or dynamically update if you have multiple targets. This is much faster than repeatedly running A* in real-time steps.

```python
import numpy as np

def precalculate_distances(grid, target):
    """Calculate Manhattan distance from all points to target on a grid with walls."""
    dist_map = np.full(grid.shape, np.inf)
    dist_map[target] = 0
    
    # Breadth-first search using vectorized grid operations
    mask = (grid == 0)
    for _ in range(grid.size):
        old_map = dist_map.copy()
        # Propagate distance in 4 cardinal directions
        shifted_up = np.roll(dist_map, -1, axis=0)
        shifted_down = np.roll(dist_map, 1, axis=0)
        shifted_left = np.roll(dist_map, -1, axis=1)
        shifted_right = np.roll(dist_map, 1, axis=1)
        
        # Minimum of neighbors + 1
        dist_map = np.minimum(dist_map, 1 + np.minimum(
            np.minimum(shifted_up, shifted_down),
            np.minimum(shifted_left, shifted_right)
        ))
        
        # Reset walls and OOB to infinity
        dist_map[~mask] = np.inf
        # Exit if no changes (fully propagated)
        if np.all(dist_map == old_map): break
    return dist_map
```

### 🔴 Coordinate Operations
Work with entire arrays of coordinates rather than individual tuples.
- `np.argwhere(grid == 0)` to get all walkable tiles.
- `np.linalg.norm(pos1 - pos2, ord=1)` for Manhattan distance.

### 🟡 Masking and Indexing
Instead of:
```python
for r in range(21):
    for c in range(21):
        if grid[r, c] == 1:
            # wall logic
```
Use:
```python
walls = (grid == 1)
# Proceed with walls mask
```

---

## 💡 Performance Tips
1. **Reuse Arrays:** Create your `dist_map` only once if possible (as an instance attribute) and reset values rather than allocating new memory.
2. **Axis Ordering:** Use `C-order` (row-major) for faster access in typical row-column grid logic.
3. **Avoid Copying:** Use `out` parameters (e.g., `np.add(a, b, out=a)`) for in-place updates.

# 🔍 Search Algorithms (Python)

This skill provides specialized knowledge and implementations for search and adversarial search algorithms in Python, specifically optimized for grid-based game environments like Pacman.

## 🚀 When to Use
- Implementing pathfinding (A*, BFS, DFS).
- Developing adversarial agents (Minimax, Expectimax, Alpha-Beta pruning).
- Optimizing search performance in 2D grids.

## 📋 Algorithm Implementations

### 🟢 A* Search (Pathfinding)
A* is ideal for the Seeker (Pacman) to find the shortest path to the Hider.
- **Heuristic:** Manhattan distance for grids.
- **Optimization:** Use `heapq` for priority queues.

```python
import heapq

def a_star(maze, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: abs(goal[0]-start[0]) + abs(goal[1]-start[1])}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
                if maze[neighbor[0]][neighbor[1]] == 1: continue
            else: continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): continue
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = gscore[neighbor] + abs(goal[0]-neighbor[0]) + abs(goal[1]-neighbor[1])
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False
```

### 🔴 Minimax with Alpha-Beta Pruning
Essential for the Hider (Ghost) to survive against the Seeker.
- **State:** Grid positions.
- **Depth:** Balanced to stay under 1s decision time.
- **Evaluation:** Maximize distance from seeker while keeping paths open.

---

## 💡 Best Practices
1. **Memoization:** Cache search results for identical map states.
2. **Early Exit:** Use `Manhattan distance < 2` as an immediate termination condition during evaluation.
3. **Beam Search:** Consider Beam Search if the full Minimax tree is too deep for 1s.

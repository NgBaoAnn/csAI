import sys
from collections import deque, defaultdict
from pathlib import Path

import numpy as np

# Add src to path to import BasePacmanAgent
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Super Elite Hunter Agent.
    Implements Adversarial Search (Minimax with Alpha-Beta) to outsmart hiders.
    Uses APSP for instant pathing and probability-based belief tracking.
    """

    CARDINALS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hunter Elite (Adversarial)"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.capture_distance = max(1, int(kwargs.get("capture_distance", 2)))

        self.last_known_enemy_pos = None
        self.last_seen_step = 0
        self.visit_counts = defaultdict(int)

        self.dist_matrix = None
        self.walkable_tiles = []
        self.belief_set = set()

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> tuple:
        if self.dist_matrix is None:
            self._precompute_apsp(map_state)

        self.visit_counts[my_position] += 1
        self._update_belief(map_state, my_position, enemy_position, step_number)

        # 1. Resolve Global State
        if enemy_position is None:
            # SEARCH MODE: Roam towards belief set centroid or hotspots
            return self._search_mode(my_position, map_state)

        # 2. ADVERSARIAL MODE: Use Minimax to find the best move against an optimal hider
        best_move = Move.STAY
        best_steps = 1
        best_score = -float('inf')

        # We evaluate potential actions (move, steps)
        # For each move, we simulate ghost's response
        for move in self.CARDINALS:
            for s in range(1, self.pacman_speed + 1):
                nxt_pos = self._apply_pac_move(my_position, move, s, map_state)
                # If we essentially stayed, skip
                if nxt_pos == my_position: break
                
                # Evaluate this position via Minimax (depth 3 for response simulation)
                score = self._alphabeta(map_state, nxt_pos, enemy_position, 3, -float('inf'), float('inf'), False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_steps = s
                
                # If we hit a wall before requested steps, don't try more steps
                if self._apply_move(nxt_pos, move, map_state) == nxt_pos:
                    break

        return (best_move, best_steps)

    def _alphabeta(self, map_state, pac_pos, ghost_pos, depth, alpha, beta, maximizing):
        dist = self.dist_matrix[pac_pos[0], pac_pos[1], ghost_pos[0], ghost_pos[1]]
        
        # Base cases
        if dist < self.capture_distance:
            return 10000 + depth # Favor faster capture
        if depth == 0:
            return self._evaluate(pac_pos, ghost_pos, map_state)

        if maximizing:
            max_eval = -float('inf')
            for move in self.CARDINALS:
                # Seeker can move up to 2 steps (but must be same direction in this env)
                for s in range(1, self.pacman_speed + 1):
                    nxt = self._apply_pac_move(pac_pos, move, s, map_state)
                    if nxt == pac_pos: break
                    
                    eval = self._alphabeta(map_state, nxt, ghost_pos, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha: break
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            # Ghost moves 1 step
            for move in self.CARDINALS + (Move.STAY,):
                nxt = self._apply_move(ghost_pos, move, map_state)
                # Ghost might choose a move that makes capture harder
                eval = self._alphabeta(map_state, pac_pos, nxt, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def _evaluate(self, pac_pos, ghost_pos, map_state):
        dist = self.dist_matrix[pac_pos[0], pac_pos[1], ghost_pos[0], ghost_pos[1]]
        
        # We want to minimize distance
        score = -dist * 20
        
        # We want to be in "intercept" position (ghost between us and another wall or further from open space)
        # But for now, simple distance + openness of ghost's current cell
        # (lower openness for ghost is good for us -> trap)
        g_openness = self._local_openness(ghost_pos, map_state, 1)
        score -= g_openness * 5
        
        return score

    def _precompute_apsp(self, map_state):
        h, w = map_state.shape
        self.dist_matrix = np.full((h, w, h, w), 9999, dtype=np.int32)
        self.walkable_tiles = []
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    self.walkable_tiles.append((r, c))
                    sr, sc = r, c
                    self.dist_matrix[sr, sc, sr, sc] = 0
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        d = self.dist_matrix[sr, sc, curr_r, curr_c]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                                if self.dist_matrix[sr, sc, nr, nc] == 9999:
                                    self.dist_matrix[sr, sc, nr, nc] = d + 1
                                    queue.append((nr, nc))

    def _update_belief(self, map_state, my_pos, enemy_pos, step_number):
        if enemy_pos is not None:
            self.last_known_enemy_pos = enemy_pos
            self.last_seen_step = step_number
            self.belief_set = {enemy_pos}
            return

        if not self.belief_set:
            self.belief_set = set(self.walkable_tiles)

        next_belief = set()
        for r, c in self.belief_set:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                    next_belief.add((nr, nc))
        
        self.belief_set = {p for p in next_belief if not self._is_visible(my_pos, p, map_state)}
        if not self.belief_set:
            self.belief_set = set(self.walkable_tiles)

    def _is_visible(self, a, b, map_state):
        if a[0] != b[0] and a[1] != b[1]: return False
        if a[0] == b[0]:
            for c in range(min(a[1], b[1]) + 1, max(a[1], b[1])):
                if map_state[a[0], c] == 1: return False
        else:
            for r in range(min(a[0], b[0]) + 1, max(a[0], b[0])):
                if map_state[r, a[1]] == 1: return False
        return True

    def _search_mode(self, my_pos, map_state):
        best_cell = None
        best_score = -float('inf')
        for p in self.walkable_tiles:
            d = self.dist_matrix[my_pos[0], my_pos[1], p[0], p[1]]
            if d == 0: continue
            score = -20 * self.visit_counts[p] - d + 5 * self._local_openness(p, map_state, 3)
            # If in belief set, huge bonus
            if p in self.belief_set: score += 100
            
            if score > best_score:
                best_score = score
                best_cell = p
        
        if best_cell:
            for move in self.CARDINALS:
                nxt = self._apply_move(my_pos, move, map_state)
                if nxt != my_pos and self.dist_matrix[nxt[0], nxt[1], best_cell[0], best_cell[1]] < self.dist_matrix[my_pos[0], my_pos[1], best_cell[0], best_cell[1]]:
                    return (move, 1)
        return (Move.STAY, 1)

    def _local_openness(self, center, map_state, radius):
        r0, c0 = center
        count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius: continue
                nr, nc = r0+dr, c0+dc
                if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                    count += 1
        return count

    def _apply_move(self, pos, move, map_state):
        dr, dc = move.value
        nr, nc = pos[0]+dr, pos[1]+dc
        if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
            return (nr, nc)
        return pos

    def _apply_pac_move(self, pos, move, steps, map_state):
        curr = pos
        for _ in range(steps):
            nxt = self._apply_move(curr, move, map_state)
            if nxt == curr: break
            curr = nxt
        return curr

    def _direction_from_to(self, src, dst):
        dr, dc = dst[0]-src[0], dst[1]-src[1]
        if abs(dr) + abs(dc) == 0: return Move.STAY
        # Since steps can be > 1, we normalize
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        for m in self.CARDINALS:
            if m.value == (sr, sc): return m
        return None

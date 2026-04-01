import sys
from pathlib import Path
from collections import deque

# Add src to path so you can import framework classes
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow')


class PacmanAgent(BasePacmanAgent):
    """
    Ultra Seeker Agent v3 — Fully Optimized (23120189)
    ===================================================
    Architecture:
      - APSP precomputed for Pacman (speed-aware) and Ghost (speed=1).
      - Minimax depth-4 with alpha-beta pruning (adversarial chase).
      - Ghost-mobility + exit-control evaluation heuristic.
      - Belief-set tracking with BFS-bounded pruning (Fog of War).
      - Adaptive exploration with exponential visit decay.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ultra_Seeker_23120189"
        self.speed = int(kwargs.get("pacman_speed", 2))

        # ── Precomputed data (built once) ──
        self.dist_matrix = None
        self.ghost_dist = None
        self.static_map = None
        self.walkable_tiles = []
        self.walkable_set = set()
        self._ghost_moves = {}        # pos → list of (r,c) ghost can reach
        self._nbr_count = None        # [H,W] walkable neighbor count per cell
        self._pac_dests = {}          # pos → list of (r,c) pacman destinations

        # ── Belief tracking ──
        self.belief_set = set()
        self.last_seen_pos = None
        self.last_seen_step = 0

        # ── Exploration ──
        self.visit_count = None
        self._last_map = None

    # ═══════════════════════════════════════════════════════
    #  PRE-COMPUTATION (runs once on first step)
    # ═══════════════════════════════════════════════════════

    def _init_static_map(self, map_state):
        """Build permanent wall map + caches for ghost/pacman moves."""
        self.static_map = (map_state == 1).astype(np.int8)
        h, w = self.static_map.shape
        self.walkable_tiles = [
            (r, c) for r in range(h) for c in range(w)
            if self.static_map[r, c] == 0
        ]
        self.walkable_set = frozenset(self.walkable_tiles)
        self.visit_count = np.zeros((h, w), dtype=np.float64)
        self._nbr_count = np.zeros((h, w), dtype=np.int8)

        # Cache ghost moves and neighbor counts
        for r, c in self.walkable_tiles:
            moves = [(r, c)]  # STAY
            cnt = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.walkable_set:
                    moves.append((nr, nc))
                    cnt += 1
            self._ghost_moves[(r, c)] = moves
            self._nbr_count[r, c] = cnt

        # Cache pacman reachable destinations (just positions, no Move enums)
        for r, c in self.walkable_tiles:
            dests = [(r, c)]  # STAY
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for s in range(1, self.speed + 1):
                    tr, tc = r + dr * s, c + dc * s
                    if not (0 <= tr < h and 0 <= tc < w
                            and self.static_map[tr, tc] == 0):
                        break
                    dests.append((tr, tc))
            self._pac_dests[(r, c)] = dests

    def _precompute_apsp(self):
        """BFS-based APSP mapped safely from cached pre-calculated adjacency lists."""
        h, w = self.static_map.shape
        dm = np.full((h, w, h, w), 255, dtype=np.uint8)
        gd = np.full((h, w, h, w), 255, dtype=np.uint8)

        for sr, sc in self.walkable_tiles:
            # Pacman APSP
            dm[sr, sc, sr, sc] = 0
            q = deque([(sr, sc, 0)])
            p_dist = dm[sr, sc]
            while q:
                r, c, d = q.popleft()
                nd = d + 1
                for nr, nc in self._pac_dests[(r, c)]:
                    if p_dist[nr, nc] == 255:
                        p_dist[nr, nc] = nd
                        q.append((nr, nc, nd))

            # Ghost APSP
            gd[sr, sc, sr, sc] = 0
            q2 = deque([(sr, sc, 0)])
            g_dist = gd[sr, sc]
            while q2:
                r, c, d = q2.popleft()
                nd = d + 1
                for nr, nc in self._ghost_moves[(r, c)]:
                    if g_dist[nr, nc] == 255:
                        g_dist[nr, nc] = nd
                        q2.append((nr, nc, nd))

        self.dist_matrix = dm
        self.ghost_dist = gd

    # ═══════════════════════════════════════════════════════
    #  MOVEMENT HELPERS
    # ═══════════════════════════════════════════════════════

    def _get_actions(self, my_position):
        """Return list of ((Move, steps), (r, c)) for Pacman from my_position."""
        h, w = self.static_map.shape
        mr, mc = my_position
        actions = [((Move.STAY, 1), my_position)]
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            dr, dc = move.value
            for s in range(1, self.speed + 1):
                tr, tc = mr + dr * s, mc + dc * s
                if not (0 <= tr < h and 0 <= tc < w
                        and self.static_map[tr, tc] == 0):
                    break
                actions.append(((move, s), (tr, tc)))
        return actions

    # ═══════════════════════════════════════════════════════
    #  BELIEF SET TRACKING (Fog of War)
    # ═══════════════════════════════════════════════════════

    def _update_belief(self, map_state, my_position, enemy_position, step_number):
        """Maintain possible ghost positions under Fog of War."""
        if enemy_position is not None:
            self.belief_set = {enemy_position}
            self.last_seen_pos = enemy_position
            self.last_seen_step = step_number
            return

        # Expand by ghost's 1-step movement (using cached moves)
        expanded = set()
        for pos in self.belief_set:
            expanded.update(self._ghost_moves.get(pos, [pos]))

        # Prune visible-and-empty cells
        visible_empty = set(zip(*np.where(map_state == 0)))
        expanded -= visible_empty
        expanded.discard(my_position)

        # Fallback if Ghost slipped through edge cases (e.g. flashes & hides)
        if not expanded and self.last_seen_pos is not None:
            max_steps = step_number - self.last_seen_step
            lr, lc = self.last_seen_pos
            expanded = {(r, c) for r, c in self.walkable_tiles
                        if self.ghost_dist[lr, lc, r, c] <= max_steps}
            expanded -= visible_empty
            expanded.discard(my_position)

        # Fallback if Ghost slipped through edge cases (e.g. flashes & hides)
        if not expanded and self.last_seen_pos is not None:
            max_steps = step_number - self.last_seen_step
            lr, lc = self.last_seen_pos
            expanded = {(r, c) for r, c in self.walkable_tiles
                        if self.ghost_dist[lr, lc, r, c] <= max_steps}
            expanded -= visible_empty
            expanded.discard(my_position)

        # Bound by BFS ghost travel since last sighting
        if self.last_seen_pos is not None and step_number - self.last_seen_step <= 60:
            max_steps = step_number - self.last_seen_step
            lr, lc = self.last_seen_pos
            bounded = {(r, c) for r, c in expanded
                       if self.ghost_dist[lr, lc, r, c] <= max_steps}
            expanded = bounded if bounded else expanded

        self.belief_set = expanded if expanded else set(self.walkable_tiles)

    # ═══════════════════════════════════════════════════════
    #  EVALUATION FUNCTION (for minimax)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _manhattan(pr, pc, gr, gc):
        return abs(pr - gr) + abs(pc - gc)

    def _evaluate(self, pr, pc, gr, gc):
        """
        Position evaluation. Lower = better for Pacman (minimizer).
        Components:
          1. APSP distance (×10) + Manhattan proximity bonus
          2. Ghost mobility penalty (fewer escapes → better for Pacman)
          3. Exit control bonus (Pacman blocking ghost exits)
        """
        manh = self._manhattan(pr, pc, gr, gc)
        if manh == 0:
            return -1000  # Same cell = always capture

        d = int(self.dist_matrix[pr, pc, gr, gc])

        # Ghost mobility: fewer neighbors = more trapped
        mobility = int(self._nbr_count[gr, gc])
        trap_bonus = (4 - mobility) * 4

        # Exit control: ghost exits Pacman can reach in ≤1 turn
        blocked = 0
        for gnr, gnc in self._ghost_moves.get((gr, gc), []):
            if (gnr, gnc) != (gr, gc):
                if self.dist_matrix[pr, pc, gnr, gnc] <= 1:
                    blocked += 1

        # Adjacency bonus: being 1 Manhattan away is very threatening
        adj_bonus = 8 if manh <= 1 else (4 if manh <= 2 else 0)

        return d * 10 - trap_bonus - blocked * 5 - adj_bonus

    # ═══════════════════════════════════════════════════════
    #  STRATEGY A: GHOST VISIBLE → DEEP ADVERSARIAL CHASE
    # ═══════════════════════════════════════════════════════

    def _chase_ghost(self, my_position, ghost_position, actions):
        """
        Minimax depth-6 with alpha-beta pruning.
        Handles simultaneous movement: checks if Pacman can guarantee
        capture against ALL ghost escape options.
        """
        gr, gc = ghost_position
        ghost_nexts = self._ghost_moves.get((gr, gc), [(gr, gc)])

        # Guaranteed capture: find a Pacman move that lands on/adjacent
        # to ghost regardless of which way ghost moves
        for action, (pr, pc) in actions:
            if all(self._manhattan(pr, pc, gnr, gnc) == 0
                   for gnr, gnc in ghost_nexts):
                return action  # Captures no matter what ghost does

        # Sort: closest to ghost first (best-first for minimizer)
        sorted_actions = sorted(
            actions,
            key=lambda a: int(self.dist_matrix[a[1][0], a[1][1], gr, gc])
        )

        best_action = sorted_actions[0][0]
        best_val = float('inf')
        beta = float('inf')
        DEPTH = 5  # 5 more plies after root = 6 total

        for action, (pr, pc) in sorted_actions:
            val = self._ab_max(pr, pc, gr, gc, DEPTH, float('-inf'), beta)
            if val < best_val:
                best_val = val
                best_action = action
            beta = min(beta, val)

        return best_action

    def _ab_max(self, pr, pc, gr_old, gc_old, depth, alpha, beta):
        """Ghost's turn (maximizer)."""
        ghost_moves = self._ghost_moves.get((gr_old, gc_old), [(gr_old, gc_old)])
        # Sort descending by distance from Pacman (best for maximizer)
        g_sorted = sorted(ghost_moves,
                          key=lambda g: -int(self.dist_matrix[pr, pc, g[0], g[1]]))

        value = float('-inf')
        for gnr, gnc in g_sorted:
            # Check capture after BOTH have moved
            if self._manhattan(pr, pc, gnr, gnc) == 0:
                val = -1000 - depth
            elif depth <= 0:
                val = self._evaluate(pr, pc, gnr, gnc)
            else:
                val = self._ab_min(pr, pc, gnr, gnc, depth - 1, alpha, beta)
            
            value = max(value, val)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value

    def _ab_min(self, pr_old, pc_old, gr_new, gc_new, depth, alpha, beta):
        """Pacman's turn (minimizer)."""
        # Use cached destinations for speed
        pac_dests = self._pac_dests.get((pr_old, pc_old), [(pr_old, pc_old)])
        # Sort ascending by distance to ghost (best for minimizer)
        p_sorted = sorted(pac_dests,
                          key=lambda p: int(self.dist_matrix[p[0], p[1], gr_new, gc_new]))

        value = float('inf')
        for npr, npc in p_sorted:
            val = self._ab_max(npr, npc, gr_new, gc_new, depth, alpha, beta)
            value = min(value, val)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    # ═══════════════════════════════════════════════════════
    #  STRATEGY B: GHOST HIDDEN → ADAPTIVE HUNT
    # ═══════════════════════════════════════════════════════

    def _hunt_ghost(self, my_position, actions, step_number):
        """
        Adaptive hunting when ghost is hidden:
          - Small belief (<40%) → centroid chase with belief sampling
          - Large belief (≥40%) → patrol towards fog-adjacent / unvisited
          - Always: decayed visit penalty to prevent circling
        """
        if not self.belief_set:
            return (Move.STAY, 1)

        # Sort belief list for deterministic behavior
        belief_list = sorted(list(self.belief_set))
        n_belief = len(belief_list)
        n_walkable = len(self.walkable_tiles)
        belief_ratio = n_belief / max(n_walkable, 1)

        # Determine primary target
        if belief_ratio < 0.4:
            target = self._get_centroid_target(belief_list)
        else:
            target = self._get_patrol_target(my_position)

        # Sample belief for average distance computation
        MAX_SAMPLE = 50
        if n_belief > MAX_SAMPLE:
            step_sz = max(1, n_belief // MAX_SAMPLE)
            sample = belief_list[::step_sz]
        else:
            sample = belief_list
        n_sample = len(sample)

        # Evaluate each action
        best_action = (Move.STAY, 1)
        best_score = float('inf')

        for action, (pr, pc) in actions:
            d_target = int(self.dist_matrix[pr, pc, target[0], target[1]])
            avg_d = sum(int(self.dist_matrix[pr, pc, br, bc])
                        for br, bc in sample) / n_sample

            # Decayed visit penalty
            visit_pen = self.visit_count[pr, pc] * 1.2

            # Weight shifts based on belief size
            w_target = 0.6 if belief_ratio < 0.4 else 0.3
            w_avg = 1.0 - w_target

            score = w_target * d_target + w_avg * avg_d + visit_pen

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _get_centroid_target(self, belief_list):
        """Walkable cell nearest to belief centroid."""
        cr = sum(r for r, c in belief_list) / len(belief_list)
        cc = sum(c for r, c in belief_list) / len(belief_list)
        return min(belief_list,
                   key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))

    def _get_patrol_target(self, my_position):
        """Best patrol target: fog-adjacent, unvisited, closer is slightly better."""
        mr, mc = my_position
        belief_list = sorted(list(self.belief_set))

        fog_adjacent = set()
        if self._last_map is not None:
            unseen = set(zip(*np.where(self._last_map == -1)))
            for r, c in unseen:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in self.walkable_set:
                        fog_adjacent.add((nr, nc))

        best_cell = belief_list[0]
        best_score = float('inf')
        for r, c in belief_list:
            visits = self.visit_count[r, c]
            dist = int(self.dist_matrix[mr, mc, r, c])
            dist_bonus = dist * 0.1  # Prefer strictly closer
            fog_bonus = -5.0 if (r, c) in fog_adjacent else 0.0
            score = visits * 3.0 + dist_bonus + fog_bonus
            if score < best_score:
                best_score = score
                best_cell = (r, c)
        return best_cell

    # ═══════════════════════════════════════════════════════
    #  MAIN STEP
    # ═══════════════════════════════════════════════════════

    def step(self, map_state, my_position, enemy_position, step_number):
        # First-call initialisation
        if self.dist_matrix is None:
            self._init_static_map(map_state)
            self._precompute_apsp()
            self.belief_set = set(self.walkable_tiles)

        # Exponential visit decay (prevents stale visit counts)
        self.visit_count *= 0.97
        self.visit_count[self.visit_count < 0.01] = 0.0  # clamp tiny values
        self.visit_count[my_position[0], my_position[1]] += 1.0

        # Save observation for fog detection
        self._last_map = map_state

        # Update belief set
        self._update_belief(map_state, my_position, enemy_position, step_number)

        # Enumerate legal actions
        actions = self._get_actions(my_position)

        # Strategy selection
        if enemy_position is not None:
            return self._chase_ghost(my_position, enemy_position, actions)
        else:
            return self._hunt_ghost(my_position, actions, step_number)

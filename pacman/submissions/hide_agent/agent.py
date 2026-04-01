"""
Ghost (Hide) Agent - Hybrid 4-Layer Strategy
CSC14003 - Nhập môn Trí tuệ Nhân tạo

Chiến lược:
  Layer 1: Topology Foundation (precomputed) - dead-ends, intersections, escape values
  Layer 2: Strategic Navigation - BFS scoring khi Pacman xa
  Layer 3: Minimax + Alpha-Beta - adversarial search khi Pacman gần
  Layer 4: Stealth & Memory - xử lý Fog of War (enemy_position = None)
"""

import sys
import random
from pathlib import Path
from collections import deque

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


# ============================================================
#  PACMAN AGENT (basic BFS chaser - not the focus)
# ============================================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_enemy = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_enemy = enemy_position
        target = enemy_position or self.last_enemy
        if target is None:
            return self._explore(my_position, map_state)
        path = self._bfs_path(my_position, target, map_state)
        if path:
            move = path[0]
            steps = 1
            pos = self._apply(my_position, move, map_state)
            for s in range(2, self.speed + 1):
                nxt = self._apply(pos, move, map_state)
                if nxt == pos:
                    break
                pos = nxt
                steps = s
            return (move, steps)
        return self._explore(my_position, map_state)

    def _explore(self, pos, m):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)
        for mv in moves:
            if self._valid(self._raw_apply(pos, mv), m):
                return (mv, 1)
        return (Move.STAY, 1)

    def _bfs_path(self, start, goal, m):
        if start == goal:
            return []
        visited = {start}
        queue = deque()
        for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            np_ = self._raw_apply(start, mv)
            if self._valid(np_, m):
                if np_ == goal:
                    return [mv]
                visited.add(np_)
                queue.append((np_, [mv]))
        while queue:
            pos, path = queue.popleft()
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                np_ = self._raw_apply(pos, mv)
                if np_ == goal:
                    return path + [mv]
                if self._valid(np_, m) and np_ not in visited:
                    visited.add(np_)
                    queue.append((np_, path + [mv]))
        return None

    def _raw_apply(self, pos, move):
        d = move.value
        return (pos[0]+d[0], pos[1]+d[1])

    def _apply(self, pos, move, m):
        np_ = self._raw_apply(pos, move)
        return np_ if self._valid(np_, m) else pos

    def _valid(self, pos, m):
        r, c = pos
        h, w = m.shape
        return 0 <= r < h and 0 <= c < w and m[r, c] != 1


# ============================================================
#  GHOST (HIDE) AGENT - Hybrid 4-Layer
# ============================================================
class GhostAgent(BaseGhostAgent):

    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    DELTAS = {Move.UP:(-1,0), Move.DOWN:(1,0), Move.LEFT:(0,-1), Move.RIGHT:(0,1), Move.STAY:(0,0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Topology (lazy init)
        self._ready = False
        self.fmap = None          # full map (walls=1, empty=0)
        self.H = self.W = 0
        self.nbrs = {}            # pos -> [(pos, Move)]
        self.dead_ends = set()
        self.intersections = set()
        self.escape_val = {}      # pos -> int
        self.exposure = {}        # pos -> int (cross-visibility exposure)

        # Memory / Fog of War
        self.last_enemy_pos = None
        self.last_seen_step = 0
        self.enemy_hist = []
        self.my_hist = []
        self.prev_move = None

    # --------------------------------------------------------
    #  MAIN STEP
    # --------------------------------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        if not self._ready:
            self._init_topology(map_state)

        self.my_hist.append(my_position)

        if enemy_position is not None:
            # ---- CAN SEE PACMAN ----
            self.last_enemy_pos = enemy_position
            self.last_seen_step = step_number
            self.enemy_hist.append(enemy_position)

            dist = self._bfs_dist(my_position, enemy_position)

            if dist <= 6:
                move = self._minimax_decision(my_position, enemy_position, depth=5)
            else:
                move = self._strategic_move(my_position, enemy_position)
        else:
            # ---- FOG OF WAR ----
            move = self._stealth_move(my_position, step_number)

        self.prev_move = move
        return move

    # --------------------------------------------------------
    #  LAYER 1: TOPOLOGY PRECOMPUTATION
    # --------------------------------------------------------
    def _init_topology(self, map_state):
        self.fmap = map_state.copy()
        self.fmap[self.fmap == -1] = 0  # unseen = empty (walls always known)
        self.H, self.W = self.fmap.shape

        # Build neighbor cache
        for r in range(self.H):
            for c in range(self.W):
                if self.fmap[r, c] == 0:
                    ns = []
                    for mv in self.ALL_MOVES:
                        dr, dc = self.DELTAS[mv]
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < self.H and 0 <= nc < self.W and self.fmap[nr, nc] == 0:
                            ns.append(((nr, nc), mv))
                    self.nbrs[(r, c)] = ns

        # Classify cells
        for pos, ns in self.nbrs.items():
            deg = len(ns)
            if deg <= 1:
                self.dead_ends.add(pos)
            elif deg >= 3:
                self.intersections.add(pos)

        # Escape value: degree + nearby intersections within 4 BFS steps
        for pos in self.nbrs:
            base = len(self.nbrs[pos])
            if pos in self.dead_ends:
                self.escape_val[pos] = 0
                continue
            cnt = 0
            visited = {pos}
            q = deque([(pos, 0)])
            while q:
                cur, d = q.popleft()
                if d >= 4:
                    continue
                if cur in self.intersections and cur != pos:
                    cnt += 1
                for (np_, _) in self.nbrs.get(cur, []):
                    if np_ not in visited:
                        visited.add(np_)
                        q.append((np_, d+1))
            self.escape_val[pos] = base + cnt * 3

        # Exposure score: total visible cells on 4 cardinal rays
        for pos in self.nbrs:
            r, c = pos
            exp = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                for d in range(1, 21):
                    nr, nc = r+dr*d, c+dc*d
                    if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
                        break
                    if self.fmap[nr, nc] == 1:
                        break
                    exp += 1
            self.exposure[pos] = exp

        self._ready = True

    # --------------------------------------------------------
    #  LAYER 2: STRATEGIC NAVIGATION (Pacman visible, far)
    # --------------------------------------------------------
    def _strategic_move(self, my_pos, pac_pos):
        pac_dist = self._bfs_dist_map(pac_pos)
        best, best_sc = Move.STAY, float('-inf')

        for mv in self.ALL_MOVES:
            np_ = self._apply(my_pos, mv)
            if np_ == my_pos and mv != Move.STAY:
                continue
            sc = self._score_pos(np_, my_pos, pac_pos, pac_dist)
            # Anti-oscillation: small penalty for reversing last move
            if self._is_reverse(mv):
                sc -= 40
            if sc > best_sc:
                best_sc = sc
                best = mv
        return best

    def _score_pos(self, pos, ghost_pos, pac_pos, pac_dist_map):
        sc = 0.0
        # BFS distance from Pacman (999 = unreachable fallback)
        d = pac_dist_map.get(pos, 999)
        sc += d * 15
        # Escape value
        sc += self.escape_val.get(pos, 0) * 20
        # Dead-end: heavy penalty
        if pos in self.dead_ends:
            sc -= 600
        # Intersection bonus (counter speed=2)
        if pos in self.intersections:
            sc += 45
        # Perpendicular movement bonus
        if ghost_pos != pos:
            gdr = pos[0] - ghost_pos[0]
            gdc = pos[1] - ghost_pos[1]
            pdr = ghost_pos[0] - pac_pos[0]
            pdc = ghost_pos[1] - pac_pos[1]
            cross = abs(gdr*pdc - gdc*pdr)
            sc += cross * 25
        # Low exposure
        sc -= self.exposure.get(pos, 0) * 3
        return sc

    # --------------------------------------------------------
    #  LAYER 3: MINIMAX + ALPHA-BETA (Pacman visible, close)
    # --------------------------------------------------------
    def _minimax_decision(self, g_pos, p_pos, depth=5):
        best_mv, best_sc = Move.STAY, float('-inf')
        alpha, beta = float('-inf'), float('inf')

        moves = self._valid_moves(g_pos)
        # Move ordering: prefer moves farther from Pacman (better pruning)
        moves.sort(key=lambda mv: self._manhattan(self._apply(g_pos, mv), p_pos), reverse=True)

        for mv in moves:
            ng = self._apply(g_pos, mv)
            sc = self._mini(ng, p_pos, depth-1, alpha, beta)
            if sc > best_sc:
                best_sc, best_mv = sc, mv
            alpha = max(alpha, best_sc)
        return best_mv

    def _maxi(self, g, p, depth, alpha, beta):
        if self._manhattan(g, p) < 2:
            return -10000
        if depth <= 0:
            return self._mm_eval(g, p)
        val = float('-inf')
        for mv in self._valid_moves(g):
            ng = self._apply(g, mv)
            val = max(val, self._mini(ng, p, depth-1, alpha, beta))
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return val

    def _mini(self, g, p, depth, alpha, beta):
        if self._manhattan(g, p) < 2:
            return -10000
        if depth <= 0:
            return self._mm_eval(g, p)
        val = float('inf')
        for np_ in self._pacman_actions(p):
            val = min(val, self._maxi(g, np_, depth-1, alpha, beta))
            beta = min(beta, val)
            if beta <= alpha:
                break
        return val

    def _pacman_actions(self, p):
        """All positions Pacman can reach (speed 1-2 in same direction)."""
        res = {p}  # STAY
        for mv in self.ALL_MOVES:
            p1 = self._apply(p, mv)  # 1 step
            res.add(p1)
            if p1 != p:  # only try 2nd step if 1st was valid
                p2 = self._apply(p1, mv)  # 2 steps same direction
                res.add(p2)
        # Sort: positions closest to Ghost first (better pruning for min)
        return list(res)

    def _mm_eval(self, g, p):
        sc = 0.0
        sc += self._manhattan(g, p) * 15
        sc += self.escape_val.get(g, 0) * 18
        if g in self.dead_ends:
            sc -= 600
        if g in self.intersections:
            sc += 35
        sc -= self.exposure.get(g, 0) * 2
        return sc

    # --------------------------------------------------------
    #  LAYER 4: STEALTH MODE (Fog of War)
    # --------------------------------------------------------
    def _stealth_move(self, my_pos, step_number):
        est_pac = self._predict_pacman(step_number)
        best, best_sc = Move.STAY, float('-inf')

        for mv in self.ALL_MOVES:
            np_ = self._apply(my_pos, mv)
            if np_ == my_pos and mv != Move.STAY:
                continue
            sc = 0.0
            # Distance from estimated Pacman
            if est_pac:
                sc += self._bfs_dist(np_, est_pac) * 10
            # Low exposure → hide better
            sc -= self.exposure.get(np_, 100) * 8
            # Escape value
            sc += self.escape_val.get(np_, 0) * 15
            # Dead-end avoidance
            if np_ in self.dead_ends:
                sc -= 600
            # Intersection bonus
            if np_ in self.intersections:
                sc += 35
            # Keep moving
            if mv == Move.STAY:
                sc -= 20
            # Anti-oscillation
            if self._is_reverse(mv):
                sc -= 50
            if len(self.my_hist) >= 2 and np_ == self.my_hist[-2]:
                sc -= 60
            if sc > best_sc:
                best_sc, best = sc, mv
        return best

    def _predict_pacman(self, cur_step):
        if self.last_enemy_pos is None:
            return None
        elapsed = cur_step - self.last_seen_step
        if elapsed > 20:
            return None
        if len(self.enemy_hist) >= 2:
            last, prev = self.enemy_hist[-1], self.enemy_hist[-2]
            dr, dc = last[0]-prev[0], last[1]-prev[1]
            t = min(elapsed, 6)
            pr = max(0, min(self.H-1, last[0]+dr*t))
            pc = max(0, min(self.W-1, last[1]+dc*t))
            pred = (int(pr), int(pc))
            if self._is_valid(pred):
                return pred
            return self._nearest_valid(pred)
        return self.last_enemy_pos

    def _nearest_valid(self, pos):
        r, c = pos
        for rad in range(1, max(self.H, self.W)):
            for dr in range(-rad, rad+1):
                for dc in range(-rad, rad+1):
                    if abs(dr)+abs(dc) <= rad:
                        cand = (r+dr, c+dc)
                        if self._is_valid(cand):
                            return cand
        return None

    # --------------------------------------------------------
    #  UTILITIES
    # --------------------------------------------------------
    def _is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.H and 0 <= c < self.W and self.fmap[r, c] == 0

    def _apply(self, pos, mv):
        dr, dc = self.DELTAS[mv]
        np_ = (pos[0]+dr, pos[1]+dc)
        return np_ if self._is_valid(np_) else pos

    def _valid_moves(self, pos):
        ms = []
        for mv in self.ALL_MOVES:
            if self._apply(pos, mv) != pos:
                ms.append(mv)
        return ms if ms else [Move.STAY]

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _bfs_dist(self, start, goal):
        if start == goal:
            return 0
        visited = {start}
        q = deque([(start, 0)])
        while q:
            pos, d = q.popleft()
            for (np_, _) in self.nbrs.get(pos, []):
                if np_ == goal:
                    return d+1
                if np_ not in visited:
                    visited.add(np_)
                    q.append((np_, d+1))
        return 999

    def _bfs_dist_map(self, start):
        dm = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            for (np_, _) in self.nbrs.get(pos, []):
                if np_ not in dm:
                    dm[np_] = dm[pos]+1
                    q.append(np_)
        return dm

    def _is_reverse(self, mv):
        if self.prev_move is None:
            return False
        rev = {Move.UP: Move.DOWN, Move.DOWN: Move.UP,
               Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}
        return mv == rev.get(self.prev_move)

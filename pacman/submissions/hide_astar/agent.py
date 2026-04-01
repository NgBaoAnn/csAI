"""
Agent 2: A* Escape Corridor Ghost (Improved)
=============================================
Tìm và di chuyển tới nút giao (>= 3 lối thoát) an toàn nhất.
Cải thiện: tính toán BFS distance thay vì Manhattan, corridor analysis
sâu hơn, scoring tốt hơn.
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque
import random


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.position_history = []
        self.last_move = None
        self._exit_cache = None
        self._intersections = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self.position_history.append(my_position)
        if len(self.position_history) > 12:
            self.position_history.pop(0)
        if self._exit_cache is None:
            self._precompute(map_state)

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            m = self._random_valid_move(my_position, map_state)
            self.last_move = m
            return m

        dist_pacman = self._bfs_distances(threat, map_state)
        best_move = Move.STAY
        best_score = -float("inf")

        for move in self.MOVES:
            np_ = self._apply(my_position, move, map_state)
            if np_ == my_position:
                continue
            score = self._evaluate(np_, my_position, threat, map_state, dist_pacman)
            if score > best_score:
                best_score = score
                best_move = move

        self.last_move = best_move
        return best_move

    def _evaluate(self, np_, my_pos, threat, map_state, dist_pacman):
        score = 0.0
        dp = dist_pacman.get(np_, 0)
        score += dp * 15

        exits = self._exit_cache.get(np_, 0)
        if exits <= 1:
            score -= 800
        elif exits == 2:
            if self._corridor_deadend(np_, my_pos, map_state, 8):
                score -= 400
        else:
            score += exits * 25

        # Distance to nearest safe intersection
        inter_dist = self._nearest_safe_intersection_dist(np_, dist_pacman, map_state)
        if inter_dist < float("inf"):
            score += max(0, 80 - inter_dist * 8)

        # Multi-step connectivity
        total_exits = 0
        for m2 in self.MOVES:
            p2 = self._apply(np_, m2, map_state)
            if p2 != np_:
                total_exits += self._exit_cache.get(p2, 0)
        score += total_exits * 5

        # Anti-oscillation
        for i, pos in enumerate(reversed(self.position_history[-6:])):
            if np_ == pos:
                score -= 40 * (6 - i)

        return score

    def _nearest_safe_intersection_dist(self, start, dist_pacman, map_state):
        h, w = map_state.shape
        visited = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            d = visited[pos]
            if d > 12:
                break
            if self._exit_cache.get(pos, 0) >= 3:
                dp = dist_pacman.get(pos, 0)
                if dp > d + 2:
                    return d
            for move in self.MOVES:
                dr, dc = move.value
                np_ = (pos[0] + dr, pos[1] + dc)
                if (0 <= np_[0] < h and 0 <= np_[1] < w
                        and map_state[np_[0], np_[1]] != 1 and np_ not in visited):
                    visited[np_] = d + 1
                    q.append(np_)
        return float("inf")

    def _corridor_deadend(self, pos, came_from, map_state, depth):
        h, w = map_state.shape
        curr, prev = pos, came_from
        for _ in range(depth):
            exits = self._exit_cache.get(curr, 0)
            if exits >= 3:
                return False
            if exits <= 1:
                return True
            moved = False
            for move in self.MOVES:
                dr, dc = move.value
                np_ = (curr[0] + dr, curr[1] + dc)
                if (np_ != prev and 0 <= np_[0] < h and 0 <= np_[1] < w
                        and map_state[np_[0], np_[1]] != 1):
                    prev, curr = curr, np_
                    moved = True
                    break
            if not moved:
                return True
        return False

    def _precompute(self, map_state):
        h, w = map_state.shape
        self._exit_cache = {}
        self._intersections = []
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    cnt = 0
                    for move in self.MOVES:
                        dr, dc = move.value
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                            cnt += 1
                    self._exit_cache[(r, c)] = cnt
                    if cnt >= 3:
                        self._intersections.append((r, c))

    def _bfs_distances(self, start, map_state):
        h, w = map_state.shape
        dist = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            d = dist[pos]
            for move in self.MOVES:
                dr, dc = move.value
                np_ = (pos[0] + dr, pos[1] + dc)
                if (0 <= np_[0] < h and 0 <= np_[1] < w
                        and map_state[np_[0], np_[1]] != 1 and np_ not in dist):
                    dist[np_] = d + 1
                    q.append(np_)
        return dist

    def _apply(self, pos, move, map_state):
        h, w = map_state.shape
        dr, dc = move.value
        np_ = (pos[0] + dr, pos[1] + dc)
        if 0 <= np_[0] < h and 0 <= np_[1] < w and map_state[np_[0], np_[1]] != 1:
            return np_
        return pos

    def _random_valid_move(self, pos, map_state):
        moves = list(self.MOVES)
        random.shuffle(moves)
        h, w = map_state.shape
        for move in moves:
            dr, dc = move.value
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < h and 0 <= c < w and map_state[r, c] != 1:
                return move
        return Move.STAY

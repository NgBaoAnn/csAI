"""
Agent 5: Potential Field + Wall Hugging Ghost (Improved)
========================================================
Kết hợp trường thế năng với chiến thuật bám tường. Khi Pacman gần,
Ghost chạy vòng quanh cụm tường để vô hiệu hóa speed-2 của Pacman.
Cải thiện: BFS distance thay Manhattan cho repulsive field, mạnh hơn
trong CLOSE mode.
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
    OPPOSITE = {Move.UP: Move.DOWN, Move.DOWN: Move.UP,
                Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT, Move.STAY: Move.STAY}
    CLOSE_THRESHOLD = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.position_history = []
        self.last_move = None
        self._exit_cache = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self.position_history.append(my_position)
        if len(self.position_history) > 12:
            self.position_history.pop(0)
        if self._exit_cache is None:
            self._exit_cache = self._precompute_exits(map_state)

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            m = self._random_valid_move(my_position, map_state)
            self.last_move = m
            return m

        bfs_dist = self._bfs_distance(threat, my_position, map_state)
        dist_pacman = self._bfs_distances(threat, map_state)

        if bfs_dist <= self.CLOSE_THRESHOLD:
            move = self._wall_hugging(my_position, threat, map_state, dist_pacman, bfs_dist)
        else:
            move = self._potential_field(my_position, threat, map_state, dist_pacman)

        self.last_move = move
        return move

    def _potential_field(self, my_pos, threat, map_state, dist_pacman):
        best_move = Move.STAY
        best_score = -float("inf")

        for move in self.MOVES:
            np_ = self._apply(my_pos, move, map_state)
            if np_ == my_pos:
                continue

            score = 0.0
            dp = dist_pacman.get(np_, 0)
            score += dp * 15

            exits = self._exit_cache.get(np_, 0)
            if exits >= 3:
                score += 50
            elif exits <= 1:
                score -= 600
            else:
                score += 10

            # 2-step connectivity
            for m2 in self.MOVES:
                p2 = self._apply(np_, m2, map_state)
                if p2 != np_:
                    e2 = self._exit_cache.get(p2, 0)
                    score += e2 * 5
                    if e2 <= 1:
                        score -= 15

            # Anti-oscillation
            for i, pos in enumerate(reversed(self.position_history[-6:])):
                if np_ == pos:
                    score -= 35 * (6 - i)

            if self.last_move and move == self.OPPOSITE.get(self.last_move):
                score -= 20

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _wall_hugging(self, my_pos, threat, map_state, dist_pacman, bfs_dist):
        """Wall-hugging: run along walls to negate Pacman's straight-line speed."""
        best_move = Move.STAY
        best_score = -float("inf")

        for move in self.MOVES:
            np_ = self._apply(my_pos, move, map_state)
            if np_ == my_pos:
                continue

            score = 0.0

            # BFS distance from Pacman (most important when close)
            dp = dist_pacman.get(np_, 0)
            score += dp * 25

            # Wall adjacency — prefer cells next to walls (winding path)
            walls = self._count_adjacent_walls(np_, map_state)
            score += walls * 18

            # Exits
            exits = self._exit_cache.get(np_, 0)
            if exits <= 1:
                score -= 700
            elif exits >= 3:
                score += 30
            elif exits == 2 and walls >= 1:
                score += 35  # Corridor along wall — great for wall hugging

            # Avoid being on same axis as Pacman (easy for speed-2 to catch)
            if bfs_dist <= 4:
                if np_[0] == threat[0]:
                    score -= 40  # Same row
                if np_[1] == threat[1]:
                    score -= 40  # Same column

            # Prefer turns over straight lines (negates speed-2)
            if self.last_move and move != self.last_move and move != self.OPPOSITE.get(self.last_move):
                score += 20  # Turning is good

            # Anti-reversal
            if self.last_move and move == self.OPPOSITE.get(self.last_move):
                score -= 50

            # Anti-oscillation
            for i, pos in enumerate(reversed(self.position_history[-6:])):
                if np_ == pos:
                    score -= 40 * (6 - i)

            # 2-step lookahead
            next_max_dp = 0
            for m2 in self.MOVES:
                p2 = self._apply(np_, m2, map_state)
                if p2 != np_:
                    dp2 = dist_pacman.get(p2, 0)
                    e2 = self._exit_cache.get(p2, 0)
                    if dp2 > next_max_dp and e2 > 1:
                        next_max_dp = dp2
            score += next_max_dp * 5

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ── BFS ───────────────────────────────────────────────────────

    def _bfs_distance(self, start, goal, map_state):
        if start == goal:
            return 0
        h, w = map_state.shape
        visited = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            d = visited[pos]
            for move in self.MOVES:
                dr, dc = move.value
                np_ = (pos[0] + dr, pos[1] + dc)
                if (0 <= np_[0] < h and 0 <= np_[1] < w
                        and map_state[np_[0], np_[1]] != 1 and np_ not in visited):
                    if np_ == goal:
                        return d + 1
                    visited[np_] = d + 1
                    q.append(np_)
        return float("inf")

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

    # ── Helpers ───────────────────────────────────────────────────

    def _precompute_exits(self, map_state):
        h, w = map_state.shape
        cache = {}
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    cnt = sum(1 for m in self.MOVES
                              for dr, dc in [m.value]
                              if 0 <= r+dr < h and 0 <= c+dc < w and map_state[r+dr, c+dc] != 1)
                    cache[(r, c)] = cnt
        return cache

    def _count_adjacent_walls(self, pos, map_state):
        h, w = map_state.shape
        count = 0
        for move in self.MOVES:
            dr, dc = move.value
            r, c = pos[0] + dr, pos[1] + dc
            if r < 0 or r >= h or c < 0 or c >= w or map_state[r, c] == 1:
                count += 1
        return count

    def _apply(self, pos, move, map_state):
        h, w = map_state.shape
        dr, dc = move.value
        np_ = (pos[0] + dr, pos[1] + dc)
        if 0 <= np_[0] < h and 0 <= np_[1] < w and map_state[np_[0], np_[1]] != 1:
            return np_
        return pos

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

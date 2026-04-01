"""
Agent 1: BFS Furthest Point Ghost (Improved)
=============================================
Sử dụng BFS để tìm điểm xa Pacman nhất trong mê cung mà Ghost có thể
tới trước, rồi di chuyển về phía đó. Cải thiện: tránh ngõ cụt mạnh hơn,
tính toán safety margin chính xác hơn cho Pacman speed-2.
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

        move = self._choose_best_move(my_position, threat, map_state)
        self.last_move = move
        return move

    def _choose_best_move(self, my_pos, threat, map_state):
        dist_pacman = self._bfs_distances(threat, map_state)
        best_move = Move.STAY
        best_score = -float("inf")

        for move in self.MOVES:
            np_ = self._apply(my_pos, move, map_state)
            if np_ == my_pos:
                continue
            score = self._score_move(np_, my_pos, threat, map_state, dist_pacman)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _score_move(self, np_, my_pos, threat, map_state, dist_pacman):
        score = 0.0
        # BFS distance from Pacman at new position
        dp = dist_pacman.get(np_, 0)
        score += dp * 15

        # Count exits (dead-end detection)
        exits = self._exit_cache.get(np_, 0)
        if exits <= 1:
            score -= 800
        elif exits == 2:
            # Check corridor safety — look ahead for dead-ends
            if self._corridor_deadend(np_, my_pos, map_state, 8):
                score -= 400
            else:
                score += 5
        else:
            score += exits * 20

        # Two-step lookahead connectivity
        next_exits = 0
        for m2 in self.MOVES:
            p2 = self._apply(np_, m2, map_state)
            if p2 != np_:
                next_exits += self._exit_cache.get(p2, 0)
        score += next_exits * 5

        # Anti-oscillation — penalize recently visited positions
        for i, pos in enumerate(reversed(self.position_history[-6:])):
            if np_ == pos:
                score -= 40 * (6 - i)

        # Anti-reversal
        if self.last_move and np_ == self.position_history[-2] if len(self.position_history) >= 2 else False:
            score -= 50

        return score

    def _corridor_deadend(self, pos, came_from, map_state, depth):
        h, w = map_state.shape
        curr = pos
        prev = came_from
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
                    prev = curr
                    curr = np_
                    moved = True
                    break
            if not moved:
                return True
        return False

    def _precompute_exits(self, map_state):
        h, w = map_state.shape
        cache = {}
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    cnt = 0
                    for move in self.MOVES:
                        dr, dc = move.value
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                            cnt += 1
                    cache[(r, c)] = cnt
        return cache

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

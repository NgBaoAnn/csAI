"""
Agent 4: Monte Carlo Simulation Ghost (Improved)
=================================================
Mô phỏng ngẫu nhiên với Pacman model mạnh hơn (BFS-greedy + speed 2).
Ghost model trong simulation dùng heuristic né tránh thông minh.
Time-based control đảm bảo luôn trong 1 giây.
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
import time


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def step(self, map_state, my_position, enemy_position, step_number):
        return Move.STAY


class GhostAgent(BaseGhostAgent):
    MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    SIM_STEPS = 30
    CAPTURE_DIST = 2
    TIME_BUDGET = 0.7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.position_history = []
        self._exit_cache = None

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.time()

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self.position_history.append(my_position)
        if len(self.position_history) > 12:
            self.position_history.pop(0)
        if self._exit_cache is None:
            self._exit_cache = self._precompute_exits(map_state)

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._random_valid_move(my_position, map_state)

        # Get valid moves with safety filter
        valid_moves = []
        for move in self.MOVES:
            np_ = self._apply(my_position, move, map_state)
            if np_ != my_position:
                exits = self._exit_cache.get(np_, 0)
                # Only enter dead-ends if no other choice
                if exits > 1 or len(valid_moves) == 0:
                    valid_moves.append(move)

        if not valid_moves:
            # All moves lead to dead ends — pick least bad
            for move in self.MOVES:
                np_ = self._apply(my_position, move, map_state)
                if np_ != my_position:
                    valid_moves.append(move)
        if not valid_moves:
            return Move.STAY

        # Monte Carlo simulations with time control
        scores = {m: 0.0 for m in valid_moves}
        counts = {m: 0 for m in valid_moves}
        idx = 0

        while time.time() - start_time < self.TIME_BUDGET:
            move = valid_moves[idx % len(valid_moves)]
            survived, steps_survived = self._simulate(
                my_position, threat, map_state, move
            )
            # Score: survival rate + bonus for lasting longer
            reward = 1.0 if survived else steps_survived / self.SIM_STEPS
            scores[move] += reward
            counts[move] += 1
            idx += 1

        # Choose best move
        best_move = valid_moves[0]
        best_rate = -1.0

        for move in valid_moves:
            if counts[move] > 0:
                rate = scores[move] / counts[move]
                np_ = self._apply(my_position, move, map_state)
                # Anti-oscillation
                if np_ in self.position_history[-4:]:
                    rate -= 0.08
                if rate > best_rate:
                    best_rate = rate
                    best_move = move

        return best_move

    def _simulate(self, ghost_start, pacman_start, map_state, first_move):
        g_pos = self._apply(ghost_start, first_move, map_state)
        p_pos = self._sim_pacman(pacman_start, ghost_start, map_state)

        if self._manhattan(g_pos, p_pos) < self.CAPTURE_DIST:
            return False, 0

        for step in range(self.SIM_STEPS):
            g_move = self._sim_ghost(g_pos, p_pos, map_state)
            g_pos = self._apply(g_pos, g_move, map_state)
            p_pos = self._sim_pacman(p_pos, g_pos, map_state)

            if self._manhattan(g_pos, p_pos) < self.CAPTURE_DIST:
                return False, step + 1

        return True, self.SIM_STEPS

    def _sim_ghost(self, g_pos, p_pos, map_state):
        if random.random() < 0.85:
            best_move = Move.STAY
            best_score = -float("inf")
            for move in self.MOVES:
                np_ = self._apply(g_pos, move, map_state)
                if np_ == g_pos:
                    continue
                dist = self._manhattan(np_, p_pos)
                exits = self._exit_cache.get(np_, 0)
                score = dist * 5 + exits * 3
                if exits <= 1:
                    score -= 30
                score += random.random() * 2  # Small randomness
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move
        else:
            return self._random_valid_move(g_pos, map_state)

    def _sim_pacman(self, p_pos, g_pos, map_state):
        """Simulate smart Pacman with speed 2."""
        best_pos = p_pos
        best_dist = float("inf")
        h, w = map_state.shape

        for move in self.MOVES:
            cur = p_pos
            for _ in range(2):  # Speed 2
                dr, dc = move.value
                np_ = (cur[0] + dr, cur[1] + dc)
                if 0 <= np_[0] < h and 0 <= np_[1] < w and map_state[np_[0], np_[1]] != 1:
                    cur = np_
                else:
                    break
            dist = self._manhattan(cur, g_pos)
            if dist < best_dist:
                best_dist = dist
                best_pos = cur

        return best_pos

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

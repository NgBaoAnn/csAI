"""
Agent 3: Minimax with Alpha-Beta Pruning Ghost (Improved)
=========================================================
Tìm kiếm đối kháng depth-8. Ghost maximize, Pacman minimize.
Pacman speed-2 được mô hình chính xác. Hàm đánh giá kết hợp
BFS distance + connectivity + dead-end penalty + wall adjacency.
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
    ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
    CAPTURE_DIST = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.position_history = []
        self._exit_cache = None
        self._time_start = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        self._time_start = time.time()

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        self.position_history.append(my_position)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        if self._exit_cache is None:
            self._exit_cache = self._precompute_exits(map_state)

        threat = enemy_position or self.last_known_enemy_pos
        if threat is None:
            return self._random_valid_move(my_position, map_state)

        # Iterative deepening minimax
        best_move = self._greedy_escape(my_position, threat, map_state)
        for depth in range(2, 12, 2):
            if time.time() - self._time_start > 0.6:
                break
            move = self._minimax_root(my_position, threat, map_state, depth)
            if move is not None:
                best_move = move

        return best_move

    def _minimax_root(self, ghost_pos, pacman_pos, map_state, depth):
        best_move = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        moves = self._get_ghost_moves(ghost_pos, map_state)
        # Sort: prioritize moves away from Pacman for better pruning
        moves.sort(key=lambda m: -self._manhattan(
            self._apply_g(ghost_pos, m, map_state), pacman_pos))

        for move in moves:
            if time.time() - self._time_start > 0.7:
                break
            new_ghost = self._apply_g(ghost_pos, move, map_state)

            # Anti-oscillation penalty
            penalty = 0
            if new_ghost in self.position_history[-4:]:
                penalty = 30

            score = self._minimax(
                new_ghost, pacman_pos, map_state,
                depth - 1, alpha, beta, False
            ) - penalty

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move

    def _minimax(self, ghost_pos, pacman_pos, map_state,
                 depth, alpha, beta, is_ghost):
        # Time check
        if time.time() - self._time_start > 0.7:
            return self._evaluate(ghost_pos, pacman_pos, map_state)

        dist = self._manhattan(ghost_pos, pacman_pos)
        if dist < self.CAPTURE_DIST:
            return -100000 + depth * 50  # Prefer later capture

        if depth == 0:
            return self._evaluate(ghost_pos, pacman_pos, map_state)

        if is_ghost:
            max_eval = -float("inf")
            for move in self._get_ghost_moves(ghost_pos, map_state):
                new_g = self._apply_g(ghost_pos, move, map_state)
                val = self._minimax(new_g, pacman_pos, map_state,
                                    depth - 1, alpha, beta, False)
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move, steps in self._get_pacman_actions(pacman_pos, map_state):
                new_p = self._apply_p(pacman_pos, move, steps, map_state)
                val = self._minimax(ghost_pos, new_p, map_state,
                                    depth - 1, alpha, beta, True)
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, ghost_pos, pacman_pos, map_state):
        dist = self._manhattan(ghost_pos, pacman_pos)
        if dist < self.CAPTURE_DIST:
            return -100000

        exits = self._exit_cache.get(ghost_pos, 0)
        score = dist * 15  # Manhattan as primary metric

        # Connectivity scoring
        score += exits * 30
        if exits <= 1:
            score -= 600
        elif exits >= 3:
            score += 40

        # 2-step connectivity
        next_exits = 0
        for m in self.MOVES:
            p = self._apply_g(ghost_pos, m, map_state)
            if p != ghost_pos:
                e = self._exit_cache.get(p, 0)
                next_exits += e
                if e <= 1:
                    next_exits -= 3
        score += next_exits * 8

        return score

    def _greedy_escape(self, my_pos, threat, map_state):
        best_move = Move.STAY
        best_score = -float("inf")
        for move in self.MOVES:
            np_ = self._apply_g(my_pos, move, map_state)
            if np_ == my_pos:
                continue
            dist = self._manhattan(np_, threat)
            exits = self._exit_cache.get(np_, 0)
            score = dist * 10 + exits * 25
            if exits <= 1:
                score -= 500
            if np_ in self.position_history[-4:]:
                score -= 30
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # ── Move generation ───────────────────────────────────────────

    def _get_ghost_moves(self, pos, map_state):
        moves = []
        for move in self.MOVES:
            if self._apply_g(pos, move, map_state) != pos:
                moves.append(move)
        if not moves:
            moves.append(Move.STAY)
        return moves

    def _get_pacman_actions(self, pos, map_state):
        actions = []
        h, w = map_state.shape
        for move in self.MOVES:
            dr, dc = move.value
            p1 = (pos[0] + dr, pos[1] + dc)
            if 0 <= p1[0] < h and 0 <= p1[1] < w and map_state[p1[0], p1[1]] != 1:
                actions.append((move, 1))
                p2 = (p1[0] + dr, p1[1] + dc)
                if 0 <= p2[0] < h and 0 <= p2[1] < w and map_state[p2[0], p2[1]] != 1:
                    actions.append((move, 2))
        if not actions:
            actions.append((Move.STAY, 1))
        return actions

    def _apply_p(self, pos, move, steps, map_state):
        if move == Move.STAY:
            return pos
        h, w = map_state.shape
        cur = pos
        for _ in range(steps):
            dr, dc = move.value
            np_ = (cur[0] + dr, cur[1] + dc)
            if 0 <= np_[0] < h and 0 <= np_[1] < w and map_state[np_[0], np_[1]] != 1:
                cur = np_
            else:
                break
        return cur

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

    def _apply_g(self, pos, move, map_state):
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

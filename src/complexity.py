from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.tree import DecisionTreeClassifier

if TYPE_CHECKING:
    from game import AFCBAGame


def _get_move_depth(game: AFCBAGame, move_id: int) -> int:
    if move_id not in game.moves:
        return 0
    move = game.moves[move_id]
    if move.target_move_id == 0:
        return 1
    return 1 + _get_move_depth(game, move.target_move_id)


def get_dispute_tree_max_depth(game: AFCBAGame) -> int:
    if not game.moves:
        return 0
    max_depth = 0
    for move_id, move in game.moves.items():
        if not move.is_backtracked:
            depth = _get_move_depth(game, move_id)
            max_depth = max(max_depth, depth)
    return max_depth


def get_dispute_tree_mean_depth(game: AFCBAGame) -> float:
    if not game.moves:
        return 0.0
    targeted_move_ids = set()
    for move in game.moves.values():
        if not move.is_backtracked and move.target_move_id != 0:
            targeted_move_ids.add(move.target_move_id)
    leaf_depths = []
    for move_id, move in game.moves.items():
        if not move.is_backtracked and move_id not in targeted_move_ids:
            leaf_depths.append(_get_move_depth(game, move_id))
    if not leaf_depths:
        return 0.0
    return float(np.mean(leaf_depths))


def get_dispute_tree_breadth(game: AFCBAGame) -> int:
    if not game.moves:
        return 0
    targeted_move_ids = set()
    for move in game.moves.values():
        if not move.is_backtracked and move.target_move_id != 0:
            targeted_move_ids.add(move.target_move_id)
    return sum(
        1
        for move_id, move in game.moves.items()
        if not move.is_backtracked and move_id not in targeted_move_ids
    )


def get_dispute_tree_complexity(game: AFCBAGame) -> dict[str, float]:
    breadth = get_dispute_tree_breadth(game)
    return {
        "max_depth": get_dispute_tree_max_depth(game),
        "mean_depth": get_dispute_tree_mean_depth(game),
        "max_breadth": breadth,
        "mean_breadth": float(breadth),
    }


def get_decision_tree_max_depth(tree: DecisionTreeClassifier) -> int:
    return tree.get_depth()


def get_decision_tree_mean_depth(tree: DecisionTreeClassifier) -> float:
    tree_structure = tree.tree_
    n_nodes = tree_structure.node_count
    node_depth = np.zeros(n_nodes, dtype=np.int64)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        left_child = tree_structure.children_left[node_id]
        right_child = tree_structure.children_right[node_id]
        if left_child != -1:
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))
    is_leaf = tree_structure.children_left == -1
    leaf_depths = node_depth[is_leaf]
    leaf_samples = tree_structure.n_node_samples[is_leaf]
    if leaf_samples.sum() == 0:
        return 0.0
    return float(np.average(leaf_depths, weights=leaf_samples))


def get_decision_tree_breadth(tree: DecisionTreeClassifier) -> int:
    tree_structure = tree.tree_
    is_leaf = tree_structure.children_left == -1
    return int(np.sum(is_leaf))


def get_decision_tree_complexity(tree: DecisionTreeClassifier) -> dict[str, float]:
    breadth = get_decision_tree_breadth(tree)
    return {
        "max_depth": get_decision_tree_max_depth(tree),
        "mean_depth": get_decision_tree_mean_depth(tree),
        "max_breadth": breadth,
        "mean_breadth": float(breadth),
    }


def aggregate_game_complexity(games: list[AFCBAGame]) -> dict[str, float]:
    if not games:
        return {
            "max_depth": 0,
            "mean_depth": 0.0,
            "max_breadth": 0,
            "mean_breadth": 0.0,
            "n_games": 0,
        }
    max_depths = []
    mean_depths = []
    breadths = []
    for game in games:
        if game is not None and game.moves:
            max_depths.append(get_dispute_tree_max_depth(game))
            mean_depths.append(get_dispute_tree_mean_depth(game))
            breadths.append(get_dispute_tree_breadth(game))
    if not max_depths:
        return {
            "max_depth": 0,
            "mean_depth": 0.0,
            "max_breadth": 0,
            "mean_breadth": 0.0,
            "n_games": 0,
        }
    return {
        "max_depth": max(max_depths),
        "mean_depth": float(np.mean(mean_depths)),
        "max_breadth": max(breadths),
        "mean_breadth": float(np.mean(breadths)),
        "n_games": len(max_depths),
    }

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argumentation import AFCBAFramework
DEFAULT_TIMEOUT_LOG_DIR = Path("logs/timeouts")


@dataclass
class GameTimeoutInfo:
    case_id: Any
    outcome: Any
    elapsed_seconds: float
    move_count: int
    argument_count: int
    defeat_count: int
    max_depth_reached: int
    focus_case_data: dict = field(default_factory=dict)
    initial_argument_names: list[str] = field(default_factory=list)
    last_moves: list[str] = field(default_factory=list)
    expanded_argument_count: int = 0
    counterexample_generator_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": str(self.case_id),
            "outcome": str(self.outcome),
            "elapsed_seconds": self.elapsed_seconds,
            "move_count": self.move_count,
            "argument_count": self.argument_count,
            "defeat_count": self.defeat_count,
            "max_depth_reached": self.max_depth_reached,
            "focus_case_data": self.focus_case_data,
            "initial_argument_names": self.initial_argument_names,
            "last_moves": self.last_moves,
            "expanded_argument_count": self.expanded_argument_count,
            "counterexample_generator_count": self.counterexample_generator_count,
        }


class GameTimeoutError(Exception):
    def __init__(self, message: str, info: GameTimeoutInfo):
        super().__init__(message)
        self.info = info


def log_timeout(
    info: GameTimeoutInfo,
    output_dir: Path = DEFAULT_TIMEOUT_LOG_DIR,
) -> Path:
    # output_dir.mkdir(parents=True, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # filename = f"timeout_case_{info.case_id}_{timestamp}.json"
    # output_path = output_dir / filename
    # log_data = {
    #     "timestamp": datetime.now().isoformat(),
    #     "timeout_info": info.to_dict(),
    #     "summary": (
    #         f"Game timed out after {info.elapsed_seconds:.1f}s with "
    #         f"{info.move_count} moves, {info.argument_count} arguments"
    #     ),
    # }
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(log_data, f, indent=2, default=str)
    output_path = "skipped logging"
    return output_path


class TimeoutChecker:
    def __init__(
        self,
        timeout_seconds: float | None = None,
        max_moves: int | None = None,
        check_interval: int = 100,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_moves = max_moves
        self.check_interval = check_interval
        self.start_time: float | None = None
        self.move_count = 0
        self._checks_since_last_time_check = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.move_count = 0
        self._checks_since_last_time_check = 0

    def increment_moves(self) -> None:
        self.move_count += 1

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time

    def check(self) -> tuple[bool, str]:
        self._checks_since_last_time_check += 1
        if self.max_moves is not None and self.move_count >= self.max_moves:
            return True, f"move_limit_exceeded ({self.move_count} >= {self.max_moves})"
        if self.timeout_seconds is not None:
            if self._checks_since_last_time_check >= self.check_interval:
                self._checks_since_last_time_check = 0
                elapsed = self.elapsed_seconds
                if elapsed >= self.timeout_seconds:
                    return (
                        True,
                        f"time_limit_exceeded ({elapsed:.1f}s >= {self.timeout_seconds}s)",
                    )
        return False, ""

    def is_enabled(self) -> bool:
        return self.timeout_seconds is not None or self.max_moves is not None

    def is_approaching_limit(self, threshold: float = 0.8) -> bool:
        if not self.is_enabled():
            return False
        if self.max_moves is not None:
            if self.move_count >= self.max_moves * threshold:
                return True
        if self.timeout_seconds is not None:
            if self.elapsed_seconds >= self.timeout_seconds * threshold:
                return True
        return False


def collect_timeout_diagnostics(
    framework: AFCBAFramework,
    moves: dict,
    case_id: Any,
    outcome: Any,
    elapsed_seconds: float,
    initial_arguments: list | None = None,
) -> GameTimeoutInfo:
    max_depth = 0
    for move_id, move in moves.items():
        depth = _calculate_move_depth(moves, move_id)
        max_depth = max(max_depth, depth)
    last_move_descriptions = []
    sorted_move_ids = sorted(moves.keys(), reverse=True)[:10]
    for move_id in sorted(sorted_move_ids):
        move = moves[move_id]
        last_move_descriptions.append(move.description())
    focus_data = {}
    if hasattr(framework, "focus_case"):
        try:
            focus_data = framework.focus_case.to_dict()
        except Exception:
            pass
    return GameTimeoutInfo(
        case_id=case_id,
        outcome=outcome,
        elapsed_seconds=elapsed_seconds,
        move_count=len(moves),
        argument_count=len(framework.arguments),
        defeat_count=len(framework.defeats),
        max_depth_reached=max_depth,
        focus_case_data=focus_data,
        initial_argument_names=[arg.name for arg in (initial_arguments or [])],
        last_moves=last_move_descriptions,
        expanded_argument_count=len(framework.expanded_arguments),
        counterexample_generator_count=len(framework.counterexample_generators),
    )


def _calculate_move_depth(moves: dict, move_id: int) -> int:
    if move_id not in moves:
        return 0
    move = moves[move_id]
    if move.target_move_id == 0:
        return 0
    return 1 + _calculate_move_depth(moves, move.target_move_id)

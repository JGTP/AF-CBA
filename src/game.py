import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from argumentation import AFCBAArgument, AFCBAFramework, DifferenceCache
from heuristics import Heuristic, create_heuristic
from timeout import (
    GameTimeoutError,
    TimeoutChecker,
    collect_timeout_diagnostics,
    log_timeout,
)

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_BOTH_JUSTIFIED_LOG_DIR = Path("logs/both_justified")


def _sanitise(text: str) -> str:
    if text is None:
        return "<None>"
    return text.replace("\x00", "").replace("\r", "")


def log_both_justified(
    case_id: Any,
    focus_case: pd.Series,
    games: dict[Any, "AFCBAGame"],
    output_dir: Path = DEFAULT_BOTH_JUSTIFIED_LOG_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"both_justified_case_{case_id}_{timestamp}.json"
    output_path = output_dir / filename

    cited_cases_data = {}
    case_differences = {}
    transformation_info = {}
    counterexample_info = {}

    case_base = None
    for game in games.values():
        if hasattr(game, "framework") and hasattr(game.framework, "case_base"):
            case_base = game.framework.case_base
            break

    dimension_info = {}
    if case_base is not None:
        for dim_name, dim in case_base.dimensions.items():
            dimension_info[dim_name] = {
                "direction": dim.direction.value,
                "coefficient": dim.coefficient,
            }

    games_info = {}
    for outcome, game in games.items():
        winning_strategy = game.get_winning_strategy()

        game_focus_case = (
            game.framework.focus_case if hasattr(game, "framework") else None
        )

        citation_differences = {}

        for move in winning_strategy:
            if move.argument:
                content = move.argument.get_content_data()
                arg_type = content.get("type", "")

                if arg_type == "Citation":
                    case_name = content.get("case_name")
                    responding_to = content.get("responding_to")

                    if case_name and case_base is not None:
                        try:
                            case_idx = int(case_name)
                            case_data = case_base.cases.loc[case_idx]

                            diffs = case_base.calculate_differences(
                                case_data, game_focus_case
                            )

                            worse_dims, better_dims = (
                                case_base.get_distinguishing_dimensions(
                                    case_data, game_focus_case
                                )
                            )

                            citation_differences[case_name] = set(diffs)

                            if case_name not in cited_cases_data:
                                cited_cases_data[case_name] = {}

                            cited_cases_data[case_name][str(outcome)] = {
                                "for_focus_outcome": outcome,
                                "precedent_outcome": case_data[case_base.target_column],
                                "same_outcome": case_data[case_base.target_column]
                                == outcome,
                                "differences_D_c_f": sorted(diffs),
                                "num_differences": len(diffs),
                                "worse_dims": sorted(worse_dims),
                                "better_dims": sorted(better_dims),
                                "values": case_data.to_dict(),
                            }

                            case_differences[case_name] = sorted(diffs)

                            if responding_to:
                                target_diffs_set = citation_differences.get(
                                    responding_to, set()
                                )
                                ce_diffs_set = set(diffs)

                                ce_key = f"{case_name}->{responding_to}"
                                counterexample_info[ce_key] = {
                                    "for_outcome": outcome,
                                    "counterexample_case": case_name,
                                    "target_case": responding_to,
                                    "counterexample_outcome": case_data[
                                        case_base.target_column
                                    ],
                                    "counterexample_diffs_D_ce_f": sorted(ce_diffs_set),
                                    "target_diffs_D_t_f": sorted(target_diffs_set),
                                    "counterexample_values": case_data.to_dict(),
                                    "differences_analysis": {
                                        "dims_only_in_counterexample": sorted(
                                            ce_diffs_set - target_diffs_set
                                        ),
                                        "dims_only_in_target": sorted(
                                            target_diffs_set - ce_diffs_set
                                        ),
                                        "dims_in_both": sorted(
                                            ce_diffs_set & target_diffs_set
                                        ),
                                        "counterexample_subset_of_target": ce_diffs_set
                                        < target_diffs_set,
                                        "target_subset_of_counterexample": target_diffs_set
                                        < ce_diffs_set,
                                        "sets_are_equal": ce_diffs_set
                                        == target_diffs_set,
                                    },
                                    "is_valid_counterexample": not (
                                        target_diffs_set < ce_diffs_set
                                    ),
                                }

                        except (KeyError, ValueError, TypeError) as e:
                            if case_name not in cited_cases_data:
                                cited_cases_data[case_name] = {}
                            cited_cases_data[case_name][str(outcome)] = {
                                "error": str(e)
                            }

        for move in winning_strategy:
            if move.argument:
                content = move.argument.get_content_data()
                if content.get("type") == "Transformed":
                    original_name = content.get("original_precedent_name")

                    defeated_counterexample = None
                    if move.target_argument:
                        target_content = move.target_argument.get_content_data()
                        if target_content.get("type") == "Citation":
                            defeated_counterexample = target_content.get("case_name")

                    transform_key = f"Transformed({original_name})->defeats({defeated_counterexample})"

                    try:
                        orig_idx = int(original_name)
                        orig_case = case_base.cases.loc[orig_idx]
                        worse_dims, better_dims = (
                            case_base.get_distinguishing_dimensions(
                                orig_case, game_focus_case
                            )
                        )

                        counterexample_diffs = []
                        if defeated_counterexample:
                            counterexample_diffs = sorted(
                                citation_differences.get(defeated_counterexample, set())
                            )

                        transformation_info[transform_key] = {
                            "for_outcome": outcome,
                            "original_precedent": original_name,
                            "defeated_counterexample": defeated_counterexample,
                            "counterexample_diffs_D_ce_f": counterexample_diffs,
                            "counterexample_has_empty_diffs": len(counterexample_diffs)
                            == 0,
                            "transformation_series": {
                                "worse_dims_compensated": sorted(worse_dims),
                                "better_dims_compensating": sorted(better_dims),
                                "compensation_move": f"Compensates({original_name}, [{','.join(sorted(better_dims))}] for [{','.join(sorted(worse_dims))}])",
                            },
                            "note": "BUG if counterexample_has_empty_diffs=True: transformation should not defeat counterexamples with D(ce,f)=∅",
                        }
                    except (KeyError, ValueError, TypeError) as e:
                        transformation_info[transform_key] = {"error": str(e)}

        games_info[str(outcome)] = {
            "total_moves": len(game.moves),
            "winning_strategy_length": len(winning_strategy),
            "winning_strategy": [move.description() for move in winning_strategy],
        }

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "case_id": str(case_id),
        "focus_case": focus_case.to_dict(),
        "justified_outcomes": list(games_info.keys()),
        "games": games_info,
        "cited_cases": cited_cases_data,
        "transformation_info": transformation_info,
        "counterexample_analysis": counterexample_info,
        "case_differences_vs_focus": case_differences,
        "dimension_directions": dimension_info,
        "note": "Both outcomes can be justified, which should not be possible.",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    return output_path


class Player(Enum):
    PRO = "PRO"
    CON = "CON"

    def other(self) -> "Player":
        return Player.CON if self == Player.PRO else Player.PRO


class Role(Enum):
    P = "P"
    O = "O"

    def other(self) -> "Role":
        return Role.O if self == Role.P else Role.P


class GameOutcome(Enum):
    PRO_WINS = "PRO_WINS"
    CON_WINS = "CON_WINS"
    UNDECIDED = "UNDECIDED"
    TIMEOUT = "TIMEOUT"


@dataclass
class GameMove:
    move_id: int
    player: Player
    role: Role
    argument: AFCBAArgument
    target_move_id: int
    target_argument: AFCBAArgument | None = None
    is_backtracked: bool = False

    def backtrack(self):
        self.is_backtracked = True

    def description(self) -> str:
        backtrack_str = " [BACKTRACKED]" if self.is_backtracked else ""
        arg_name = _sanitise(str(self.argument.name)) if self.argument else "<None>"
        if self.target_move_id == 0:
            return (
                f"Move {self.move_id}: {self.player.value},{self.role.value} "
                f"puts forward {arg_name}{backtrack_str}"
            )
        else:
            target_name = (
                _sanitise(str(self.target_argument.name))
                if self.target_argument
                else "unknown"
            )
            return (
                f"Move {self.move_id}: {self.player.value},{self.role.value} "
                f"attacks {target_name} with {arg_name} "
                f"(targeting move {self.target_move_id}){backtrack_str}"
            )


class AFCBAGame:
    def __init__(
        self,
        framework: AFCBAFramework,
        predicted_outcome: Any,
        verbose: bool = False,
        output_file: str | Path | None = None,
        timeout_seconds: float | None = None,
        max_moves: int | None = None,
        case_id: Any = None,
    ):
        self.framework = framework
        self.predicted_outcome = predicted_outcome
        self.moves = {}
        self.move_counter = 1
        self.verbose = verbose
        self._output_handle: TextIO | None = None
        self.case_id = case_id
        self._initial_arguments: list[AFCBAArgument] = []
        self._timeout_checker = TimeoutChecker(
            timeout_seconds=timeout_seconds, max_moves=max_moves
        )
        if output_file is None and self.verbose:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.output_file = DEFAULT_LOG_DIR / f"game_{timestamp}.txt"
        else:
            self.output_file = Path(output_file) if output_file else None

    def _log(self, message: str):
        if not self.verbose or not self._output_handle:
            return
        self._output_handle.write(_sanitise(message) + "\n")

    def _check_timeout(self) -> None:
        if not self._timeout_checker.is_enabled():
            return
        exceeded, reason = self._timeout_checker.check()
        if exceeded:
            info = collect_timeout_diagnostics(
                framework=self.framework,
                moves=self.moves,
                case_id=self.case_id,
                outcome=self.predicted_outcome,
                elapsed_seconds=self._timeout_checker.elapsed_seconds,
                initial_arguments=self._initial_arguments,
            )
            log_path = log_timeout(info)
            raise GameTimeoutError(
                f"Game timeout: {reason}. Diagnostics logged to {log_path}", info=info
            )

    def play(self, initial_arguments: list[AFCBAArgument]) -> GameOutcome:
        self._initial_arguments = initial_arguments
        self._timeout_checker.start()
        if self.output_file and self.verbose:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self._output_handle = open(self.output_file, "w", encoding="utf-8")
        try:
            return self._play_game(initial_arguments)
        except GameTimeoutError:
            if self.verbose:
                self._log(
                    f"\n⚠️  GAME TIMEOUT after {self._timeout_checker.move_count} moves"
                )
            return GameOutcome.TIMEOUT
        finally:
            if self._output_handle:
                self._output_handle.close()
                self._output_handle = None

    def _play_game(self, initial_arguments: list[AFCBAArgument]) -> GameOutcome:
        if self.verbose:
            self._log("=" * 60)
            self._log(f"GROUNDED GAME START - Outcome: {self.predicted_outcome}")
            self._log("=" * 60)
            arg_names = [_sanitise(str(arg.name)) for arg in initial_arguments]
            self._log(f"Initial arguments: {arg_names}")
            self._log("")
        for arg in initial_arguments:
            self.framework.add_argument(arg)
        self._play_recursive(Player.PRO, initial_arguments, 0)
        outcome = self._determine_outcome()
        if self.verbose:
            self._log("")
            self._log("=" * 60)
            self._log(f"GAME RESULT: {outcome.value}")
            self._log(f"Total moves: {len(self.moves)}")
            self._log("=" * 60)
        return outcome

    def _play_recursive(
        self,
        player: Player,
        possible_arguments: list[AFCBAArgument],
        target_move_id: int,
    ):
        self._check_timeout()
        if not possible_arguments:
            if self.verbose:
                indent = "  " * self._get_move_depth(target_move_id)
                self._log(f"{indent}  \\- {player.value} has no moves, backtracking...")
            return
        for argument in possible_arguments:
            if self._is_last_player(player):
                return
            role = Role.P if player == Player.PRO else Role.O
            if not self._is_legal_move(player, argument, target_move_id, role):
                if self.verbose:
                    indent = "  " * (self._get_move_depth(target_move_id) + 1)
                    arg_name = _sanitise(str(argument.name))
                    self._log(f"{indent}[skip] {arg_name} - illegal move")
                continue
            this_move_id = self.move_counter
            if role == Role.P and this_move_id != target_move_id + 1:
                self._backtrack_moves(target_move_id + 1, this_move_id)
            current_move = GameMove(
                move_id=this_move_id,
                player=player,
                role=role,
                argument=argument,
                target_move_id=target_move_id,
                target_argument=(
                    self.moves[target_move_id].argument if target_move_id > 0 else None
                ),
            )
            self.moves[this_move_id] = current_move
            self.move_counter += 1
            self._timeout_checker.increment_moves()
            if self.verbose:
                indent = "  " * self._get_move_depth(this_move_id)
                self._log(f"{indent}>> {current_move.description()}")
            if self._played_before_in_line(argument, target_move_id, player):
                if self.verbose:
                    indent = "  " * self._get_move_depth(this_move_id)
                    self._log(f"{indent}  \\- Repeated argument, line ends")
                return
            replies = self.framework.get_attackers(argument)
            if self.verbose and replies:
                indent = "  " * self._get_move_depth(this_move_id)
                reply_names = [_sanitise(str(r.name)) for r in replies]
                self._log(f"{indent}  |- Possible replies: {reply_names}")
            self._play_recursive(player.other(), replies, current_move.move_id)

    def _case_cited_in_line(self, case_name: str, target_move_id: int) -> bool:
        if target_move_id == 0:
            return False
        current_move = self.moves[target_move_id]
        if current_move.argument:
            content = current_move.argument.get_content_data()
            if (
                content.get("type") == "Citation"
                and content.get("case_name") == case_name
            ):
                return True
        return self._case_cited_in_line(case_name, current_move.target_move_id)

    def _is_last_player(self, player: Player) -> bool:
        if not self.moves:
            return False
        last_move_id = max(self.moves.keys())
        return self.moves[last_move_id].player == player

    def _is_legal_move(self, player, argument, target_move_id, role):
        content = argument.get_content_data()
        if content.get("type") == "Citation":
            case_name = content.get("case_name")
            if case_name and self._case_cited_in_line(case_name, target_move_id):
                return False
        return not (
            role == Role.P
            and self._played_before_in_line(argument, target_move_id, None)
        )

    def _played_before_in_line(
        self, argument: AFCBAArgument, target_move_id: int, player: Player | None
    ) -> bool:
        if target_move_id == 0:
            return False
        current_move = self.moves[target_move_id]
        if current_move.argument.name == argument.name:
            if player is None or current_move.player == player:
                return True
        return self._played_before_in_line(
            argument, current_move.target_move_id, player
        )

    def _backtrack_moves(self, start_move_id: int, end_move_id: int):
        if self.verbose:
            self._log(
                f"    [backtrack] Marking moves {start_move_id}-{end_move_id - 1}"
            )
        for move_id in range(start_move_id, end_move_id):
            if move_id in self.moves:
                self.moves[move_id].backtrack()

    def _determine_outcome(self) -> GameOutcome:
        if not self.moves:
            return GameOutcome.UNDECIDED
        last_player = None
        for move_id in sorted(self.moves.keys(), reverse=True):
            move = self.moves[move_id]
            if not move.is_backtracked:
                last_player = move.player
                break
        if last_player == Player.PRO:
            return GameOutcome.PRO_WINS
        elif last_player == Player.CON:
            return GameOutcome.CON_WINS
        else:
            return GameOutcome.UNDECIDED

    def get_winning_strategy(self) -> list[GameMove]:
        main_line = []
        for move_id in sorted(self.moves.keys()):
            move = self.moves[move_id]
            if not move.is_backtracked:
                main_line.append(move)
        return main_line

    def get_game_description(self) -> str:
        if not self.moves:
            return "Empty game"
        lines = ["Grounded Game Tree:", f"Query: {self.predicted_outcome}", ""]
        for move_id in sorted(self.moves.keys()):
            move = self.moves[move_id]
            indent = "  " * self._get_move_depth(move_id)
            lines.append(f"{indent}{move.description()}")
        return "\n".join(lines)

    def _get_move_depth(self, move_id: int) -> int:
        if move_id not in self.moves:
            return 0
        move = self.moves[move_id]
        if move.target_move_id == 0:
            return 0
        return 1 + self._get_move_depth(move.target_move_id)


class AFCBAClassifier:
    def __init__(
        self,
        case_base,
        heuristic: Heuristic | str = "majority",
        verbose: bool = False,
        output_dir: str | Path = DEFAULT_LOG_DIR,
        timeout_seconds: float | None = None,
        max_moves: int | None = None,
        early_termination: bool = False,
        **heuristic_kwargs,
    ):
        self.case_base = case_base
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self._case_counter = 0
        self.timeout_seconds = timeout_seconds
        self.max_moves = max_moves
        self.early_termination = early_termination
        self._timeout_cases: list[Any] = []
        self._both_justified_cases: list[Any] = []
        if isinstance(heuristic, Heuristic):
            self._heuristic = heuristic
        else:
            self._heuristic = create_heuristic(heuristic, **heuristic_kwargs)
        self._heuristic.fit(case_base.cases, case_base.target_column)

    def _get_output_file(self, case_id: Any, outcome: Any) -> Path:
        return self.output_dir / f"game_case_{case_id}_outcome_{outcome}.txt"

    def classify(
        self,
        focus_case: pd.Series,
        case_id: Any = None,
        return_game: bool = False,
    ) -> tuple[Any, bool] | tuple[Any, bool, "AFCBAGame | None"]:
        if case_id is None:
            case_id = self._case_counter
            self._case_counter += 1
        outcomes = list(self.case_base.cases[self.case_base.target_column].unique())
        outcome_counts = self.case_base.cases[
            self.case_base.target_column
        ].value_counts()
        majority_outcome = outcome_counts.idxmax()
        outcomes_ordered = sorted(outcomes, key=lambda o: o != majority_outcome)
        justified_outcomes = []
        justified_games: dict[Any, AFCBAGame] = {}
        for outcome in outcomes_ordered:
            result, game = self._can_justify_outcome(focus_case, outcome, case_id)
            if result is True:
                justified_outcomes.append(outcome)
                justified_games[outcome] = game
                if self.early_termination:
                    if return_game:
                        return outcome, False, game
                    return outcome, False
        if len(justified_outcomes) == 1:
            winning_game = justified_games[justified_outcomes[0]]
            if return_game:
                return justified_outcomes[0], False, winning_game
            return justified_outcomes[0], False
        elif len(justified_outcomes) >= 2:
            self._both_justified_cases.append(case_id)
            # log_both_justified(case_id, focus_case, justified_games)
            predicted_outcome = self._heuristic.predict(focus_case)
            if return_game:
                return predicted_outcome, True, justified_games[predicted_outcome]
            return predicted_outcome, True
        else:
            if return_game:
                return self._heuristic.predict(focus_case), True, None
            return self._heuristic.predict(focus_case), True

    def _can_justify_outcome(
        self, focus_case: pd.Series, outcome: Any, case_id: Any
    ) -> tuple[bool | None, AFCBAGame | None]:
        focus_with_outcome = focus_case.copy()
        focus_with_outcome[self.case_base.target_column] = outcome
        difference_cache = DifferenceCache(self.case_base, focus_with_outcome)
        best_precedents = self.case_base.find_best_precedents(
            focus_with_outcome, difference_cache=difference_cache
        )
        if len(best_precedents) == 0:
            return False, None
        framework = AFCBAFramework(
            self.case_base, focus_with_outcome, difference_cache=difference_cache
        )
        initial_arguments = []
        for idx, precedent in best_precedents.iterrows():
            precedent_name = str(idx)
            citation = AFCBAArgument.citation(
                precedent, focus_with_outcome, precedent_name, introduced_by="PRO"
            )
            initial_arguments.append(citation)
        output_file = self._get_output_file(case_id, outcome)
        game = AFCBAGame(
            framework,
            outcome,
            verbose=self.verbose,
            output_file=output_file,
            timeout_seconds=self.timeout_seconds,
            max_moves=self.max_moves,
            case_id=case_id,
        )
        outcome_result = game.play(initial_arguments)
        if outcome_result == GameOutcome.TIMEOUT:
            self._timeout_cases.append(case_id)
            return None, None
        if outcome_result == GameOutcome.PRO_WINS:
            return True, game
        return False, None

    def find_justification(
        self, focus_case: pd.Series, predicted_outcome: Any, case_id: Any = None
    ) -> list[GameMove] | None:
        if case_id is None:
            case_id = self._case_counter
            self._case_counter += 1
        focus_with_outcome = focus_case.copy()
        focus_with_outcome[self.case_base.target_column] = predicted_outcome
        difference_cache = DifferenceCache(self.case_base, focus_with_outcome)
        best_precedents = self.case_base.find_best_precedents(
            focus_with_outcome, difference_cache=difference_cache
        )
        if len(best_precedents) == 0:
            return None
        framework = AFCBAFramework(
            self.case_base, focus_with_outcome, difference_cache=difference_cache
        )
        initial_arguments = []
        for idx, precedent in best_precedents.iterrows():
            precedent_name = str(idx)
            citation = AFCBAArgument.citation(
                precedent, focus_with_outcome, precedent_name, introduced_by="PRO"
            )
            initial_arguments.append(citation)
        output_file = self._get_output_file(case_id, predicted_outcome)
        game = AFCBAGame(
            framework,
            predicted_outcome,
            verbose=self.verbose,
            output_file=output_file,
            timeout_seconds=self.timeout_seconds,
            max_moves=self.max_moves,
            case_id=case_id,
        )
        outcome = game.play(initial_arguments)
        if outcome == GameOutcome.PRO_WINS:
            return game.get_winning_strategy()
        return None

    @property
    def heuristic(self) -> Heuristic:
        return self._heuristic

    @property
    def timeout_cases(self) -> list[Any]:
        return self._timeout_cases.copy()

    @property
    def both_justified_cases(self) -> list[Any]:
        return self._both_justified_cases.copy()

    def clear_timeout_cases(self) -> None:
        self._timeout_cases.clear()

    def clear_both_justified_cases(self) -> None:
        self._both_justified_cases.clear()

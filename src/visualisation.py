import math
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from argumentation import AFCBAArgument, AFCBAFramework
from game import AFCBAGame, GameMove, Player


class GameVisualiser:
    PRO_COLOURS = {
        "face": "#a8d5a2",
        "edge": "#2e7d32",
    }
    CON_COLOURS = {
        "face": "#ffcdd2",
        "edge": "#c62828",
    }
    GROUNDED_COLOURS = {
        "face": "#c8e6c9",
        "edge": "#1b5e20",
    }
    DEFAULT_COLOURS = {
        "face": "#e0e0e0",
        "edge": "#616161",
    }

    def __init__(self, figsize: tuple[float, float] = (18, 14)):
        self.figsize = figsize

    def visualise_winning_strategy(
        self,
        moves: list[GameMove],
        output_path: str | Path,
        title: str = None,
    ):
        if not moves:
            print("No moves to visualise")
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title or "AF-CBA Winning Strategy", fontsize=16, fontweight="bold")
        positions = self._compute_tree_layout(moves)
        self._draw_edges(ax, moves, positions)
        self._draw_nodes(ax, moves, positions)
        self._add_legend(ax)
        ax.set_aspect("equal")
        ax.axis("off")
        if positions:
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            margin = 2.5
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Visualisation saved to: {output_path}")

    def _compute_tree_layout(
        self,
        moves: list[GameMove],
        horizontal_spacing: float = 3.5,
        vertical_spacing: float = 3.0,
    ) -> dict[int, tuple[float, float]]:
        if not moves:
            return {}
        children: dict[int, list[int]] = {0: []}
        for move in moves:
            children.setdefault(move.move_id, [])
            children.setdefault(move.target_move_id, [])
            children[move.target_move_id].append(move.move_id)
        for parent_id in children:
            children[parent_id].sort()
        move_map = {m.move_id: m for m in moves}
        subtree_widths = self._calculate_subtree_widths(children, move_map)
        positions = {}
        self._position_subtree(
            node_id=0,
            children=children,
            positions=positions,
            subtree_widths=subtree_widths,
            x_offset=0,
            y=0,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            move_map=move_map,
        )
        positions.pop(0, None)
        return positions

    def _calculate_subtree_widths(
        self,
        children: dict[int, list[int]],
        move_map: dict[int, GameMove],
    ) -> dict[int, float]:
        widths = {}

        def calculate(node_id: int) -> float:
            if node_id in widths:
                return widths[node_id]
            child_ids = children.get(node_id, [])
            if not child_ids:
                widths[node_id] = 1.0
            else:
                total = sum(calculate(c) for c in child_ids)
                widths[node_id] = max(1.0, total)
            return widths[node_id]

        calculate(0)
        for move in move_map.values():
            calculate(move.move_id)
        return widths

    def _position_subtree(
        self,
        node_id: int,
        children: dict[int, list[int]],
        positions: dict[int, tuple[float, float]],
        subtree_widths: dict[int, float],
        x_offset: float,
        y: float,
        horizontal_spacing: float,
        vertical_spacing: float,
        move_map: dict[int, GameMove],
    ):
        child_ids = children.get(node_id, [])
        if node_id != 0:
            subtree_width = subtree_widths.get(node_id, 1.0)
            x = x_offset + (subtree_width * horizontal_spacing) / 2
            positions[node_id] = (x, y)
        if child_ids:
            current_x = x_offset
            for child_id in child_ids:
                child_width = subtree_widths.get(child_id, 1.0)
                self._position_subtree(
                    node_id=child_id,
                    children=children,
                    positions=positions,
                    subtree_widths=subtree_widths,
                    x_offset=current_x,
                    y=y - vertical_spacing,
                    horizontal_spacing=horizontal_spacing,
                    vertical_spacing=vertical_spacing,
                    move_map=move_map,
                )
                current_x += child_width * horizontal_spacing

    def _draw_edges(
        self,
        ax: plt.Axes,
        moves: list[GameMove],
        positions: dict[int, tuple[float, float]],
    ):
        move_map = {m.move_id: m for m in moves}

        def get_box_height(move: GameMove) -> float:
            header, dims_text = self._format_move_label(move)
            header_lines = header.count("\n") + 1
            dims_lines = dims_text.count("\n") + 1 if dims_text else 0
            header_height = 0.25 * header_lines + 0.15
            dims_height = 0.18 * dims_lines if dims_text else 0
            return max(0.5, header_height + dims_height + 0.2)

        for move in moves:
            if move.target_move_id == 0:
                continue
            if move.move_id not in positions:
                continue
            if move.target_move_id not in positions:
                continue
            from_pos = positions[move.move_id]
            to_pos = positions[move.target_move_id]
            from_height = get_box_height(move) / 2
            target_move = move_map.get(move.target_move_id)
            if target_move:
                to_height = get_box_height(target_move) / 2
            else:
                to_height = 0.4
            start_x = from_pos[0]
            start_y = from_pos[1] + from_height
            end_x = to_pos[0]
            end_y = to_pos[1] - to_height
            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#666666",
                    lw=1.5,
                    alpha=0.7,
                    connectionstyle="arc3,rad=0",
                ),
            )

    def _draw_nodes(
        self,
        ax: plt.Axes,
        moves: list[GameMove],
        positions: dict[int, tuple[float, float]],
    ):
        for move in moves:
            if move.move_id not in positions:
                continue
            pos = positions[move.move_id]
            if move.player == Player.PRO:
                colours = self.PRO_COLOURS
            else:
                colours = self.CON_COLOURS
            header, dims_text = self._format_move_label(move)
            header_lines = header.count("\n") + 1
            dims_lines = dims_text.count("\n") + 1 if dims_text else 0
            box_width = 3
            header_height = 0.25 * header_lines + 0.15
            dims_height = 0.18 * dims_lines if dims_text else 0
            box_height = max(0.5, header_height + dims_height + 0.2)
            rect = patches.FancyBboxPatch(
                (pos[0] - box_width / 2, pos[1] - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor=colours["face"],
                edgecolor=colours["edge"],
                linewidth=2.5,
                alpha=0.95,
            )
            ax.add_patch(rect)
            if dims_text:
                header_y = pos[1] + box_height / 2 - header_height / 2 - 0.1
                ax.text(
                    pos[0],
                    header_y,
                    header,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    linespacing=0.9,
                )
                dims_y = pos[1] - box_height / 2 + dims_height / 2 + 0.1
                ax.text(
                    pos[0],
                    dims_y,
                    dims_text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontfamily="monospace",
                    linespacing=0.85,
                    color="#444444",
                )
            else:
                ax.text(
                    pos[0],
                    pos[1],
                    header,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    linespacing=0.9,
                )

    def _format_move_label(self, move: GameMove, max_dims: int = 4) -> tuple[str, str]:
        if not move.argument:
            return "?", ""
        content = move.argument.get_content_data()
        arg_type = content.get("type", "")
        if arg_type == "Citation":
            case_name = content.get("case_name", "?")
            responding_to = content.get("responding_to")
            if responding_to:
                return f"Counter({case_name})", ""
            return f"Cite({case_name})", ""
        elif arg_type == "Worse":
            precedent = content.get("precedent_name", "?")
            dims = content.get("worse_dimensions", [])
            dims_str = self._format_dimensions(dims, max_dims)
            return f"Worse({precedent})", dims_str
        elif arg_type == "Compensates":
            precedent = content.get("precedent_name", "?")
            worse_dims = content.get("worse_dimensions", [])
            better_dims = content.get("better_dimensions", [])
            worse_str = self._format_dimensions(worse_dims, max_dims)
            better_str = self._format_dimensions(better_dims, max_dims)
            header = f"Comp({precedent})"
            dims_text = (
                f"+ {better_str}\n— {worse_str}"
                if better_str != "∅"
                else f"— {worse_str}"
            )
            return header, dims_text
        elif arg_type == "Transformed":
            original = content.get("original_precedent_name", "?")
            return f"Trans({original})", ""
        arg_name = move.argument.name
        if len(arg_name) > 15:
            return arg_name[:12] + "...", ""
        return arg_name, ""

    def _format_dimensions(self, dims: list[str], max_dims: int = 4) -> str:
        if not dims:
            return "∅"
        lines = []
        for d in dims[:max_dims]:
            if len(d) > 15:
                lines.append(d[:13] + "..")
            else:
                lines.append(d)
        if len(dims) > max_dims:
            lines.append(f"+{len(dims) - max_dims} more")
        return "\n".join(lines)

    def _add_legend(self, ax: plt.Axes):
        legend_elements = [
            patches.Patch(
                facecolor=self.PRO_COLOURS["face"],
                edgecolor=self.PRO_COLOURS["edge"],
                linewidth=2,
                label="PRO (Proponent)",
            ),
            patches.Patch(
                facecolor=self.CON_COLOURS["face"],
                edgecolor=self.CON_COLOURS["edge"],
                linewidth=2,
                label="CON (Opponent)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            framealpha=0.9,
            fontsize=11,
        )

    def visualise_framework_arguments(
        self,
        framework: AFCBAFramework,
        output_path: str | Path,
        title: str = None,
        highlight_grounded: bool = True,
        layout: str = "hierarchical",
    ):
        arguments = list(framework.arguments.values())
        defeats = [(defeat.from_arg, defeat.to_arg) for defeat in framework.defeats]
        grounded_args = set()
        if highlight_grounded:
            grounded_args = framework.get_grounded_extension()
        self._create_framework_graph(
            arguments=arguments,
            defeats=defeats,
            output_path=output_path,
            title=title or "AF-CBA Framework",
            layout=layout,
            grounded_args=grounded_args,
        )

    def _create_framework_graph(
        self,
        arguments: list[AFCBAArgument],
        defeats: list[tuple[AFCBAArgument, AFCBAArgument]],
        output_path: str | Path,
        title: str,
        layout: str,
        grounded_args: set[AFCBAArgument],
        node_size: float = 0.35,
    ):
        if not arguments:
            print("No arguments to visualise")
            return
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(title, fontsize=16, fontweight="bold")
        if layout == "circular":
            positions = self._circular_layout(arguments)
        else:
            positions = self._spring_layout(arguments, defeats)
        for from_arg, to_arg in defeats:
            if from_arg.name in positions and to_arg.name in positions:
                from_pos = positions[from_arg.name]
                to_pos = positions[to_arg.name]
                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    unit_x = dx / length
                    unit_y = dy / length
                    start_x = from_pos[0] + node_size * unit_x
                    start_y = from_pos[1] + node_size * unit_y
                    end_x = to_pos[0] - node_size * unit_x
                    end_y = to_pos[1] - node_size * unit_y
                    ax.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->", color="#c62828", lw=1.5, alpha=0.7
                        ),
                    )
        for arg in arguments:
            if arg.name not in positions:
                continue
            pos = positions[arg.name]
            if arg in grounded_args:
                colours = self.GROUNDED_COLOURS
            else:
                colours = self.DEFAULT_COLOURS
            circle = patches.Circle(
                pos,
                node_size,
                facecolor=colours["face"],
                edgecolor=colours["edge"],
                linewidth=2,
                alpha=0.9,
            )
            ax.add_patch(circle)
            label = self._shorten_argument_name(arg.name)
            ax.text(
                pos[0],
                pos[1],
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_aspect("equal")
        ax.axis("off")
        if positions:
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            margin = node_size * 3
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        legend_elements = [
            patches.Patch(
                facecolor=self.GROUNDED_COLOURS["face"],
                edgecolor=self.GROUNDED_COLOURS["edge"],
                linewidth=2,
                label="Grounded (accepted)",
            ),
            patches.Patch(
                facecolor=self.DEFAULT_COLOURS["face"],
                edgecolor=self.DEFAULT_COLOURS["edge"],
                linewidth=2,
                label="Not grounded",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Visualisation saved to: {output_path}")

    def _shorten_argument_name(self, name: str) -> str:
        if len(name) <= 15:
            return name
        if name.startswith("Citation("):
            case_part = name[9:].split(")")[0]
            return f"Cite({case_part})"
        elif name.startswith("Counterexample("):
            case_part = name[15:].split("->")[0]
            return f"Cntr({case_part})"
        elif name.startswith("Worse("):
            parts = name[6:].split(",")
            if parts:
                return f"Worse({parts[0][:8]})"
            return "Worse"
        elif name.startswith("Compensates("):
            return "Comp."
        elif name.startswith("Transformed("):
            return "Trans."
        return name[:12] + "..."

    def _circular_layout(
        self, arguments: list[AFCBAArgument]
    ) -> dict[str, tuple[float, float]]:
        positions = {}
        n = len(arguments)
        if n == 0:
            return positions
        if n == 1:
            return {arguments[0].name: (0, 0)}
        radius = max(2.0, n * 0.4)
        angle_step = 2 * math.pi / n
        for i, arg in enumerate(arguments):
            angle = i * angle_step - math.pi / 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[arg.name] = (x, y)
        return positions

    def _spring_layout(
        self,
        arguments: list[AFCBAArgument],
        defeats: list[tuple[AFCBAArgument, AFCBAArgument]],
        iterations: int = 100,
    ) -> dict[str, tuple[float, float]]:
        if not arguments:
            return {}
        n = len(arguments)
        if n == 1:
            return {arguments[0].name: (0, 0)}
        import random

        random.seed(42)
        positions = {}
        for arg in arguments:
            positions[arg.name] = (random.uniform(-3, 3), random.uniform(-3, 3))
        connections = set()
        for from_arg, to_arg in defeats:
            connections.add((from_arg.name, to_arg.name))
        k = 2.0
        for _ in range(iterations):
            forces = {name: [0.0, 0.0] for name in positions}
            names = list(positions.keys())
            for i, name1 in enumerate(names):
                for name2 in names[i + 1 :]:
                    pos1 = positions[name1]
                    pos2 = positions[name2]
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    force = k * k / dist
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[name1][0] += fx
                    forces[name1][1] += fy
                    forces[name2][0] -= fx
                    forces[name2][1] -= fy
            for name1, name2 in connections:
                if name1 in positions and name2 in positions:
                    pos1 = positions[name1]
                    pos2 = positions[name2]
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    force = dist / k
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[name1][0] += fx
                    forces[name1][1] += fy
                    forces[name2][0] -= fx
                    forces[name2][1] -= fy
            damping = 0.1
            for name in positions:
                fx, fy = forces[name]
                positions[name] = (
                    positions[name][0] + damping * fx,
                    positions[name][1] + damping * fy,
                )
        return positions


def visualise_game_result(game: AFCBAGame, output_path: str | Path, title: str = None):
    winning_strategy = game.get_winning_strategy()
    if not winning_strategy:
        print("No winning strategy to visualise")
        return
    visualiser = GameVisualiser()
    visualiser.visualise_winning_strategy(
        moves=winning_strategy,
        output_path=output_path,
        title=title or f"Winning Strategy ({len(winning_strategy)} moves)",
    )


def visualise_framework_state(
    framework: AFCBAFramework, output_path: str | Path, title: str = None
):
    visualiser = GameVisualiser()
    visualiser.visualise_framework_arguments(
        framework=framework, output_path=output_path, title=title
    )

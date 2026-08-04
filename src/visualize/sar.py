"""Top-down SAR scene snapshot + multi-agent trajectory drawing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon

from visualize.zones import FancyAxes, draw_diamond, draw_path, hide_ticks

DEFAULT_EXTENTS = (-3.5, -3.5, 3.5, 3.5)

# Distinct agent path colors (tab10-ish, readable on light floor).
_AGENT_COLORS = (
    "#991fb4",
    "#6556A1",
    "#84c1da",
    "#35a451",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

_COLOR_BORDER = "#212121"
_COLOR_INTERIOR = "#757575"
_COLOR_BUILDING_FILL = "#d7b4a5"
_COLOR_BUILDING_WALL = "#8d4a2b"
_COLOR_SURFACE = "#f1da0a"
_COLOR_ENTRAPPED = "#ef0000"


@dataclass
class OrientedBox:
    xy: np.ndarray  # (2,)
    half_xy: np.ndarray  # (2,) half-extents in local frame
    yaw: float = 0.0


@dataclass
class CasualtyMark:
    xy: np.ndarray
    radius: float
    rescued: bool = False


@dataclass
class SarSceneSnapshot:
    buildings: list[OrientedBox] = field(default_factory=list)
    building_walls: list[OrientedBox] = field(default_factory=list)
    interior_walls: list[OrientedBox] = field(default_factory=list)
    border_walls: list[OrientedBox] = field(default_factory=list)
    surface_casualties: list[CasualtyMark] = field(default_factory=list)
    entrapped_casualties: list[CasualtyMark] = field(default_factory=list)
    extents: tuple[float, float, float, float] = DEFAULT_EXTENTS


def _yaw_from_xmat(xmat: np.ndarray) -> float:
    m = np.asarray(xmat, dtype=float).reshape(3, 3)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def _box_from_engine(engine: Any, body_name: str) -> OrientedBox | None:
    try:
        body = engine.data.body(body_name)
        size = engine.model.geom(body_name).size
    except Exception:  # noqa: BLE001 — missing body/geom
        return None
    xy = np.asarray(body.xpos, dtype=float)[:2].copy()
    yaw = _yaw_from_xmat(body.xmat)
    half = np.asarray(size[:2], dtype=float).copy()
    return OrientedBox(xy=xy, half_xy=half, yaw=yaw)


def _boxes_for_geom(geom: Any) -> list[OrientedBox]:
    """Read MuJoCo boxes for a Geom with plural name → body ``name[:-1]{i}``."""
    if geom is None:
        return []
    engine = getattr(geom, "engine", None)
    num = int(getattr(geom, "num", 0) or 0)
    name = getattr(geom, "name", None)
    if engine is None or num <= 0 or not name:
        return []
    prefix = name[:-1]
    out: list[OrientedBox] = []
    for i in range(num):
        box = _box_from_engine(engine, f"{prefix}{i}")
        if box is not None:
            out.append(box)
    return out


def _building_footprints(task: Any) -> list[OrientedBox]:
    try:
        from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import building_geom
    except ImportError:
        return []
    buildings = building_geom(task)
    if buildings is None or int(getattr(buildings, "num", 0) or 0) <= 0:
        return []
    engine = getattr(buildings, "engine", None)
    half = float(getattr(buildings, "size", 0.5) or 0.5)
    prefix = buildings.name[:-1]
    out: list[OrientedBox] = []
    for i in range(int(buildings.num)):
        body_name = f"{prefix}{i}"
        yaw = 0.0
        if engine is not None:
            try:
                body = engine.data.body(body_name)
                xy = np.asarray(body.xpos, dtype=float)[:2].copy()
                yaw = _yaw_from_xmat(body.xmat)
            except Exception:  # noqa: BLE001
                pos = np.asarray(buildings.pos[i], dtype=float)[:2]
                xy = pos.copy()
        else:
            xy = np.asarray(buildings.pos[i], dtype=float)[:2].copy()
        out.append(OrientedBox(xy=xy, half_xy=np.array([half, half], dtype=float), yaw=yaw))
    return out


def _casualty_marks(geom: Any) -> list[CasualtyMark]:
    if geom is None or int(getattr(geom, "num", 0) or 0) <= 0:
        return []
    radius = float(getattr(geom, "size", 0.05) or 0.05)
    rescued = list(getattr(geom, "rescued", []) or [])
    marks: list[CasualtyMark] = []
    for i in range(int(geom.num)):
        pos = np.asarray(geom.pos[i], dtype=float)[:2].copy()
        is_rescued = bool(rescued[i]) if i < len(rescued) else False
        marks.append(CasualtyMark(xy=pos, radius=radius, rescued=is_rescued))
    return marks


def _scene_extents(task: Any) -> tuple[float, float, float, float]:
    conf = getattr(task, "placements_conf", None)
    extents = getattr(conf, "extents", None) if conf is not None else None
    if extents is not None and len(extents) >= 4:
        return (
            float(extents[0]),
            float(extents[1]),
            float(extents[2]),
            float(extents[3]),
        )
    return DEFAULT_EXTENTS


def snapshot_sar_scene(task: Any) -> SarSceneSnapshot:
    """Capture top-down SAR layout from a live task (post-reset)."""
    building_walls: list[OrientedBox] = []
    for name in getattr(task, "_geoms", []) or []:
        try:
            from safety_gymnasium.tasks.safe_multi_agent.utils.sar_utils import (
                is_building_ltl_wall,
            )
        except ImportError:
            break
        if not is_building_ltl_wall(name):
            continue
        building_walls.extend(_boxes_for_geom(getattr(task, name, None)))

    arena = None
    if hasattr(task, "_arena_ltl_walls"):
        try:
            arena = task._arena_ltl_walls()
        except Exception:  # noqa: BLE001
            arena = None
    if arena is None:
        arena = getattr(task, "ltl_walls", None)
        if arena is not None and getattr(arena, "name", None) != "ltl_walls":
            arena = None

    return SarSceneSnapshot(
        buildings=_building_footprints(task),
        building_walls=building_walls,
        interior_walls=_boxes_for_geom(getattr(task, "walls", None)),
        border_walls=_boxes_for_geom(arena),
        surface_casualties=_casualty_marks(getattr(task, "surface_casualtys", None)),
        entrapped_casualties=_casualty_marks(getattr(task, "entrapped_casualtys", None)),
        extents=_scene_extents(task),
    )


def _oriented_box_patch(
    center: np.ndarray,
    half_xy: np.ndarray,
    yaw: float,
    *,
    facecolor,
    edgecolor,
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 2,
) -> Polygon:
    hx, hy = float(half_xy[0]), float(half_xy[1])
    corners = np.array(
        [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]],
        dtype=float,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    world = corners @ rot.T + np.asarray(center, dtype=float)[:2]
    return Polygon(
        world,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def setup_sar_axis(ax, extents: tuple[float, float, float, float] = DEFAULT_EXTENTS) -> None:
    xmin, ymin, xmax, ymax = extents
    pad = 0.15
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.grid(True, which="both", color="gray", linestyle="dashed", linewidth=1, alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
        ax.spines[side].set_color("gray")
        ax.spines[side].set_linewidth(0.5)
    hide_ticks(ax.xaxis)
    hide_ticks(ax.yaxis)


def draw_sar_scene(ax, scene: SarSceneSnapshot) -> None:
    """Paint buildings, walls, and casualties on a top-down axis."""
    for box in scene.border_walls:
        ax.add_patch(
            _oriented_box_patch(
                box.xy,
                box.half_xy,
                box.yaw,
                facecolor=to_rgba(_COLOR_BORDER, 0.85),
                edgecolor=_COLOR_BORDER,
                linewidth=1.2,
                zorder=1,
            )
        )
    for box in scene.interior_walls:
        ax.add_patch(
            _oriented_box_patch(
                box.xy,
                box.half_xy,
                box.yaw,
                facecolor=to_rgba(_COLOR_INTERIOR, 0.8),
                edgecolor=_COLOR_INTERIOR,
                linewidth=0.8,
                zorder=2,
            )
        )
    for box in scene.buildings:
        ax.add_patch(
            _oriented_box_patch(
                box.xy,
                box.half_xy,
                box.yaw,
                facecolor=to_rgba(_COLOR_BUILDING_FILL, 0.25),
                edgecolor=to_rgba(_COLOR_BUILDING_FILL, 0.6),
                linewidth=0.8,
                zorder=3,
            )
        )
    for box in scene.building_walls:
        ax.add_patch(
            _oriented_box_patch(
                box.xy,
                box.half_xy,
                box.yaw,
                facecolor=to_rgba(_COLOR_BUILDING_WALL, 0.9),
                edgecolor=_COLOR_BUILDING_WALL,
                linewidth=1.0,
                zorder=4,
            )
        )
    for mark in scene.surface_casualties:
        color = _COLOR_SURFACE
        if mark.rescued:
            circ = Circle(
                mark.xy,
                mark.radius * 2.5,
                facecolor="none",
                edgecolor=to_rgba(color, 0.5),
                linewidth=1.5,
                linestyle="--",
                zorder=6,
            )
        else:
            circ = Circle(
                mark.xy,
                mark.radius * 2.5,
                facecolor=to_rgba(color, 0.9),
                edgecolor=color,
                linewidth=1.0,
                zorder=6,
            )
        ax.add_patch(circ)
    for mark in scene.entrapped_casualties:
        color = _COLOR_ENTRAPPED
        if mark.rescued:
            circ = Circle(
                mark.xy,
                mark.radius * 2.5,
                facecolor="none",
                edgecolor=to_rgba(color, 0.5),
                linewidth=1.5,
                linestyle="--",
                zorder=6,
            )
        else:
            circ = Circle(
                mark.xy,
                mark.radius * 2.5,
                facecolor=to_rgba(color, 0.9),
                edgecolor=color,
                linewidth=1.0,
                zorder=6,
            )
        ax.add_patch(circ)


def draw_agent_paths(
    ax,
    paths_by_agent: dict[str, Sequence[np.ndarray]],
    *,
    legend: bool = True,
) -> None:
    """Draw per-agent trajectories; start=diamond, end=dot."""
    for i, (key, points) in enumerate(sorted(paths_by_agent.items())):
        if not points:
            continue
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        pts = [np.asarray(p, dtype=float)[:2] for p in points]
        draw_path(ax, pts, color=color, linewidth=2.5)
        draw_diamond(ax, pts[0], color=color, size=0.12)
        ax.plot(pts[-1][0], pts[-1][1], marker="o", color=color, markersize=5, zorder=11)
        if legend:
            ax.plot([], [], color=color, linewidth=2.5, label=key)
    if legend and paths_by_agent:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)


def draw_sar_trajectories(
    scenes: Sequence[SarSceneSnapshot],
    paths_list: Sequence[dict[str, Sequence[np.ndarray]]],
    titles: Sequence[str],
    num_cols: int,
    num_rows: int,
):
    """Grid of top-down SAR episodes (one scene + all agent paths per cell)."""
    if len(scenes) != len(paths_list):
        raise ValueError("Number of scenes and path dicts must match")
    if len(scenes) != len(titles):
        raise ValueError("Number of scenes and titles must match")
    if num_cols * num_rows < len(scenes):
        raise ValueError("Number of scenes exceeds subplot grid")

    fig = plt.figure(figsize=(7.5 * num_cols, 7.0 * num_rows))
    for i, (scene, paths, title) in enumerate(zip(scenes, paths_list, titles)):
        ax = fig.add_subplot(
            num_rows, num_cols, i + 1, axes_class=FancyAxes, edgecolor="gray", linewidth=0.5,
        )
        ax.set_title(title, fontsize=9)
        setup_sar_axis(ax, scene.extents)
        draw_sar_scene(ax, scene)
        draw_agent_paths(ax, paths, legend=(i == 0))
    plt.tight_layout(pad=2.5)
    return fig

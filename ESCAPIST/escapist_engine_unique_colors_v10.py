from __future__ import annotations

import colorsys
import math
import webbrowser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as error:
    raise SystemExit(
        "Plotly is required. Install it with: python -m pip install plotly"
    ) from error


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# None gives a new independent realization on every run.  Use an integer to
# reproduce a particular environment.
SEED = None

# Finite observation window [-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH]^2 x [0, T].
SPACE_HALF_WIDTH = 8.0
TOTAL_TIME = 10.0
# A finer time mesh reveals the small-scale irregularity of standard Brownian
# motion without changing its law: every increment is still N(0, dt I_2).
TIME_STEPS = 121

# Poisson intensity and the i.i.d. discrete radius law.  If K is geometric,
#
#     P(K = k) = (1 - q) q^k,       k = 0, 1, 2, ...,
#     R_K = RADIUS_BASE * RADIUS_LEVEL_GROWTH**K,
#
# then most detection tubes are small, but increasingly large levels remain
# possible.  The condition q * RADIUS_LEVEL_GROWTH**2 < 1 guarantees
# E[R^2] < infinity, as required in spatial dimension d=2.
POISSON_INTENSITY = 0.080
RADIUS_BASE = 0.40
RADIUS_LEVEL_GROWTH = 1.50
RADIUS_LEVEL_TAIL = 0.35  # q
# Moment order whose finiteness is checked.  The interface changes this value
# together with RADIUS_LEVEL_TAIL before starting a simulation.
RADIUS_MOMENT_ORDER = 2.0

# The target is a point.  Increase TARGET_RADIUS to simulate a thick target.
TARGET_RADIUS = 0.0
MAX_TARGET_SPEED = 4.5
NUM_SPATIAL_GRID_POINTS = 61
PATH_BOUNDARY_MARGIN = 0.50
NUMERICAL_SAFETY_MARGIN = 0.025
# A slightly larger clearance floor makes separation from the translucent tube
# boundary visible in the rendered figure while remaining a valid vacant path.
PATH_CLEARANCE_FLOOR = 0.10

# Among collision-free transitions, the fugitive prefers to use its available
# speed and to stay away from the occupied set.  There is no prescribed curve,
# direction, winding number or endpoint.
MOVEMENT_SHORTFALL_COST_WEIGHT = 18.0
PREFERRED_PATH_CLEARANCE = 0.30
LOW_CLEARANCE_COST_WEIGHT = 3.00
CLEARANCE_PATH_COST_WEIGHT = 0.10

# Recursive branching factor.  With two large coarse time layers and factor
# five, the complete regular subtree has 5**2 = 25 terminal escape paths.
BRANCHING_FACTOR = 5
MAX_BRANCH_ENDPOINT_CANDIDATES = 4000
BRANCH_DIVERSITY_WEIGHT = 1.20

# Coarse boxes used to display the space-time tessellation.  Collision and
# speed checks remain on the fine grid above.
TESSELLATION_SPATIAL_DIVISIONS = 4
TESSELLATION_TIME_DIVISIONS = 2

# The entrance A is a disk in the lower time face, as in Definition 2.4.
ENTRANCE_RADIUS = 3.00

# Initial Poisson points are also sampled in a halo around the displayed box.
# A particle outside the box can therefore enter it during the simulation.
BROWNIAN_HALO_SIGMAS = 4.5
EXTRA_HALO = 2.0

# Output controls
SAVE_VIDEO = True
OPEN_INTERACTIVE_HTML = True
VIDEO_FPS = 10
ENVIRONMENT_VIDEO_SECONDS = 8.0
FUTURE_PAUSE_SECONDS = 2.0
ESCAPE_VIDEO_SECONDS = 16.0
MAX_DISPLAY_TUBES = 55
MAX_WEBSITE_TUBES = 24
TUBE_ANGLE_POINTS = 18
TUBE_TIME_STRIDE = 1

# Visual identity: the observed set is neutral gray, the initial point is blue
# and each escape branch receives a continuously sampled pastel color.
DETECTION_TUBE_COLOR = (0.66, 0.68, 0.71)
DETECTION_TUBE_COLOR_CSS = "rgb(168,173,181)"
DETECTION_CENTERLINE_COLOR = "#626871"
START_COLOR = "#397eb8"
ESCAPE_COLOR = "#df7f87"
ESCAPE_COLORS = [ESCAPE_COLOR]
ESCAPE_FAMILY_COLORS = [ESCAPE_COLOR]
WEBSITE_BACKGROUND_COLOR = "#dceff9"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def random_pastel_escape_color(rng: np.random.Generator) -> str:
    """Sample a continuous light color that remains visible on the background."""

    background_rgb = (0xDC / 255.0, 0xEF / 255.0, 0xF9 / 255.0)
    while True:
        hue = float(rng.uniform(0.0, 1.0))
        saturation = float(rng.uniform(0.40, 0.62))
        lightness = float(rng.uniform(0.68, 0.78))
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        distance_from_background = math.sqrt(
            sum(
                (channel - background_channel) ** 2
                for channel, background_channel in zip(rgb, background_rgb)
            )
        )
        if distance_from_background >= 0.24:
            return "#" + "".join(
                f"{int(round(255.0 * channel)):02x}" for channel in rgb
            )


def random_pastel_escape_colors(
    rng: np.random.Generator,
    count: int,
) -> list[str]:
    """Generate one random pastel color per path using the golden angle."""

    if count <= 0:
        return []
    background_rgb = (0xDC / 255.0, 0xEF / 255.0, 0xF9 / 255.0)
    golden_angle_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    initial_hue = float(rng.uniform(0.0, 1.0))
    colors: list[str] = []
    used_colors: set[str] = set()
    for color_index in range(count):
        hue = (initial_hue + color_index * golden_angle_fraction) % 1.0
        for _attempt in range(100):
            saturation = float(rng.uniform(0.42, 0.62))
            lightness = float(rng.uniform(0.68, 0.78))
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            distance_from_background = math.sqrt(
                sum(
                    (channel - background_channel) ** 2
                    for channel, background_channel in zip(
                        rgb, background_rgb
                    )
                )
            )
            candidate = "#" + "".join(
                f"{int(round(255.0 * channel)):02x}" for channel in rgb
            )
            if distance_from_background >= 0.22 and candidate not in used_colors:
                colors.append(candidate)
                used_colors.add(candidate)
                break
        else:
            # A deterministic fallback preserves uniqueness after rounding.
            saturation = 0.55
            lightness = 0.72 - 0.001 * (color_index % 20)
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            candidate = "#" + "".join(
                f"{int(round(255.0 * channel)):02x}" for channel in rgb
            )
            colors.append(candidate)
            used_colors.add(candidate)
    return colors


def simulate_environment(rng: np.random.Generator, times: np.ndarray):
    """Simulate the truncated marked Poisson process and Brownian motions."""

    if not 0.0 < RADIUS_LEVEL_TAIL < 1.0:
        raise ValueError("RADIUS_LEVEL_TAIL must lie strictly between 0 and 1.")
    if RADIUS_LEVEL_GROWTH <= 1.0:
        raise ValueError("RADIUS_LEVEL_GROWTH must be greater than 1.")
    if RADIUS_MOMENT_ORDER <= 0.0:
        raise ValueError("RADIUS_MOMENT_ORDER must be positive.")
    if (
        RADIUS_LEVEL_TAIL
        * RADIUS_LEVEL_GROWTH**RADIUS_MOMENT_ORDER
        >= 1.0
    ):
        raise ValueError(
            "The radius law must satisfy "
            "RADIUS_LEVEL_TAIL * "
            "RADIUS_LEVEL_GROWTH**RADIUS_MOMENT_ORDER < 1."
        )

    halo = BROWNIAN_HALO_SIGMAS * math.sqrt(TOTAL_TIME) + EXTRA_HALO
    poisson_half_width = SPACE_HALF_WIDTH + halo
    poisson_area = (2.0 * poisson_half_width) ** 2
    particle_count = rng.poisson(POISSON_INTENSITY * poisson_area)

    initial_positions = rng.uniform(
        -poisson_half_width,
        poisson_half_width,
        size=(particle_count, 2),
    )
    radius_levels = rng.geometric(
        p=1.0 - RADIUS_LEVEL_TAIL,
        size=particle_count,
    ) - 1
    radii = RADIUS_BASE * RADIUS_LEVEL_GROWTH**radius_levels

    dt = np.diff(times)
    increments = rng.normal(
        size=(len(times) - 1, particle_count, 2)
    ) * np.sqrt(dt)[:, None, None]

    centers = np.empty((len(times), particle_count, 2), dtype=float)
    centers[0] = initial_positions
    centers[1:] = initial_positions[None, :, :] + np.cumsum(
        increments, axis=0
    )
    return initial_positions, radii, centers, poisson_half_width


def endpoint_clearances(
    grid_points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Clearance from every grid point to the occupied set at every time."""

    time_count = len(centers)
    grid_shape = grid_points.shape[:2]
    flat_points = grid_points.reshape(-1, 2)
    result = np.empty((time_count, *grid_shape), dtype=np.float32)

    boundary_clearance = (
        SPACE_HALF_WIDTH - np.max(np.abs(flat_points), axis=1)
    )

    for time_index in range(time_count):
        if len(radii):
            difference = (
                flat_points[:, None, :] - centers[time_index][None, :, :]
            )
            obstacle_clearance = np.min(
                np.linalg.norm(difference, axis=2)
                - radii[None, :]
                - TARGET_RADIUS,
                axis=1,
            )
        else:
            obstacle_clearance = np.full(len(flat_points), np.inf)

        result[time_index] = np.minimum(
            obstacle_clearance, boundary_clearance
        ).reshape(grid_shape)

    return result


# ---------------------------------------------------------------------------
# Vacant bounded-speed crossing
# ---------------------------------------------------------------------------


def overlapping_slices(size: int, shift: int):
    """Slices for current_index = previous_index + shift."""

    if shift >= 0:
        previous = slice(0, size - shift)
        current = slice(shift, size)
    else:
        previous = slice(-shift, size)
        current = slice(0, size + shift)
    return previous, current


def transition_clearance(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    previous_centers: np.ndarray,
    current_centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Exact clearance for linearly interpolated target/particle segments.

    Brownian motion is represented by its standard piecewise-linear numerical
    interpolation.  The minimum relative distance on each whole time interval
    is computed analytically, rather than checked only at frame endpoints.
    """

    if not len(radii):
        return np.full(len(previous_points), np.inf)

    relative_start = (
        previous_points[:, None, :] - previous_centers[None, :, :]
    )
    relative_velocity = (
        (current_points - previous_points)[:, None, :]
        - (current_centers - previous_centers)[None, :, :]
    )

    denominator = np.sum(relative_velocity**2, axis=2)
    numerator = -np.sum(relative_start * relative_velocity, axis=2)
    closest_parameter = np.zeros_like(denominator)
    np.divide(
        numerator,
        denominator,
        out=closest_parameter,
        where=denominator > 1.0e-14,
    )
    np.clip(closest_parameter, 0.0, 1.0, out=closest_parameter)

    closest_relative_position = (
        relative_start
        + closest_parameter[:, :, None] * relative_velocity
    )
    return np.min(
        np.linalg.norm(closest_relative_position, axis=2)
        - radii[None, :]
        - TARGET_RADIUS,
        axis=1,
    )


def allowed_grid_shifts(spatial_step: float, dt: float):
    maximum_index_shift = int(
        math.floor(MAX_TARGET_SPEED * dt / spatial_step + 1.0e-12)
    )
    shifts = []
    for row_shift in range(-maximum_index_shift, maximum_index_shift + 1):
        for column_shift in range(
            -maximum_index_shift, maximum_index_shift + 1
        ):
            distance = spatial_step * math.hypot(row_shift, column_shift)
            if distance <= MAX_TARGET_SPEED * dt + 1.0e-12:
                shifts.append((row_shift, column_shift, distance))
    return shifts


def build_admissible_predecessor_tree(
    times: np.ndarray,
    grid_points: np.ndarray,
    clearances: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    start_grid_index: tuple[int, int],
):
    """Build all least-cost reachable states in the vacant fine-grid DAG.

    Validity is imposed first.  Among valid transitions, dynamic programming
    prefers large spatial steps and clearance from the Brownian sausages.
    No direction, winding pattern or target endpoint is prescribed.
    """

    grid_size = grid_points.shape[0]
    spatial_step = float(grid_points[0, 1, 0] - grid_points[0, 0, 0])
    dt = float(times[1] - times[0])
    shifts = allowed_grid_shifts(spatial_step, dt)
    if len(shifts) <= 1:
        raise ValueError(
            "The space-time grid is too fine for MAX_TARGET_SPEED; "
            "increase the speed or the number of time steps."
        )
    maximum_spatial_step = max(shift[2] for shift in shifts)

    flat_indices = np.arange(grid_size**2, dtype=np.int32).reshape(
        grid_size, grid_size
    )
    entrance_mask = np.zeros((grid_size, grid_size), dtype=bool)
    entrance_mask[start_grid_index] = True
    if clearances[0][start_grid_index] < PATH_CLEARANCE_FLOOR:
        raise ValueError("The selected initial point is not in the vacant set.")

    costs = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    costs[entrance_mask] = 0.0
    bottlenecks = np.full(
        (grid_size, grid_size), -np.inf, dtype=np.float32
    )
    bottlenecks[entrance_mask] = clearances[0][entrance_mask]
    back_pointers = np.full(
        (len(times), grid_size, grid_size), -1, dtype=np.int32
    )

    for time_index in range(1, len(times)):
        next_costs = np.full_like(costs, np.inf)
        next_bottlenecks = np.full_like(bottlenecks, -np.inf)
        next_back = back_pointers[time_index]

        for row_shift, column_shift, spatial_distance in shifts:
            previous_rows, current_rows = overlapping_slices(
                grid_size, row_shift
            )
            previous_columns, current_columns = overlapping_slices(
                grid_size, column_shift
            )

            previous_slice = (previous_rows, previous_columns)
            current_slice = (current_rows, current_columns)
            previous_costs = costs[previous_slice]
            previous_bottlenecks = bottlenecks[previous_slice]
            reachable = np.isfinite(previous_costs)
            if not np.any(reachable):
                continue

            previous_points = grid_points[previous_slice].reshape(-1, 2)
            current_points = grid_points[current_slice].reshape(-1, 2)
            interval_clearance = transition_clearance(
                previous_points,
                current_points,
                centers[time_index - 1],
                centers[time_index],
                radii,
            ).reshape(previous_costs.shape)

            local_clearance = np.minimum(
                clearances[time_index][current_slice],
                interval_clearance,
            )
            candidate_bottlenecks = np.minimum(
                previous_bottlenecks, local_clearance
            )
            candidate_is_valid = (
                reachable
                & (candidate_bottlenecks >= PATH_CLEARANCE_FLOOR)
            )

            clearance_deficit = np.maximum(
                0.0,
                PREFERRED_PATH_CLEARANCE - local_clearance,
            ) / PREFERRED_PATH_CLEARANCE
            continuous_clearance_cost = 1.0 / (
                0.15 + np.maximum(local_clearance, PATH_CLEARANCE_FLOOR)
            )
            candidate_costs = (
                previous_costs
                + MOVEMENT_SHORTFALL_COST_WEIGHT
                * (maximum_spatial_step - spatial_distance) ** 2
                + LOW_CLEARANCE_COST_WEIGHT * clearance_deficit**2
                + CLEARANCE_PATH_COST_WEIGHT * continuous_clearance_cost
            )

            current_cost_view = next_costs[current_slice]
            current_bottleneck_view = next_bottlenecks[current_slice]
            equal_cost = np.isclose(
                candidate_costs,
                current_cost_view,
                rtol=0.0,
                atol=1.0e-10,
            )
            better = candidate_is_valid & (
                (candidate_costs < current_cost_view)
                | (
                    equal_cost
                    & (candidate_bottlenecks > current_bottleneck_view)
                )
            )
            if np.any(better):
                current_cost_view[better] = candidate_costs[better]
                current_bottleneck_view[better] = candidate_bottlenecks[better]
                previous_index_view = flat_indices[previous_slice]
                next_back_view = next_back[current_slice]
                next_back_view[better] = previous_index_view[better]

        costs = next_costs
        bottlenecks = next_bottlenecks
        if not np.any(np.isfinite(costs)):
            return None, None, None

    return costs, bottlenecks, back_pointers


def reconstruct_path(
    endpoint_flat_index: int,
    back_pointers: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    """Reconstruct one root-to-exit path from the predecessor tree."""

    flat_grid_points = grid_points.reshape(-1, 2)
    path = np.empty((len(back_pointers), 2), dtype=float)
    path[-1] = flat_grid_points[endpoint_flat_index]
    current_flat_index = int(endpoint_flat_index)
    for time_index in range(len(back_pointers) - 1, 0, -1):
        current_flat_index = int(
            back_pointers[time_index].flat[current_flat_index]
        )
        if current_flat_index < 0:
            raise RuntimeError("Broken back-pointer in admissible path.")
        path[time_index - 1] = flat_grid_points[current_flat_index]
    return path


def path_separation(candidate: np.ndarray, selected: np.ndarray) -> float:
    """Measure sustained post-root separation between two crossings."""

    start = max(1, len(candidate) // 5)
    distances = np.linalg.norm(candidate[start:] - selected[start:], axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))
    terminal = float(np.linalg.norm(candidate[-1] - selected[-1]))
    return 0.75 * rms + 0.25 * terminal


def select_diverse_tree_children(
    children: list[dict],
    branch_factor: int,
    cost_low: float,
    cost_scale: float,
) -> list[dict]:
    """Choose well-separated viable children of one predecessor-tree node."""

    selected = [min(children, key=lambda child: child["minimum_cost"])]
    remaining = [child for child in children if child is not selected[0]]
    while len(selected) < branch_factor:
        best_child = None
        best_score = -np.inf
        for child in remaining:
            separation = min(
                float(
                    np.linalg.norm(
                        np.asarray(child["coordinate"])
                        - np.asarray(previous["coordinate"])
                    )
                )
                for previous in selected
            )
            normalized_cost = min(
                3.0,
                (child["minimum_cost"] - cost_low) / cost_scale,
            )
            score = BRANCH_DIVERSITY_WEIGHT * separation - normalized_cost
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            break
        selected.append(best_child)
        remaining.remove(best_child)
    return selected


def select_recursive_branching_crossings(
    costs: np.ndarray,
    bottlenecks: np.ndarray,
    back_pointers: np.ndarray,
    grid_points: np.ndarray,
    branch_factor: int,
    branching_levels: int,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[float],
    list[int],
    list[int],
]:
    """Extract a complete regular subtree from the predecessor tree.

    At each coarse time boundary every retained space-time box has exactly
    ``branch_factor`` viable child boxes.  This is the block-level notion used
    in the multiscale construction: crossings in one box need not coincide at
    one microscopic grid vertex.
    """

    finite_endpoints = np.flatnonzero(np.isfinite(costs.ravel()))
    if len(finite_endpoints) == 0:
        return [], [], [], [], []

    endpoint_costs = costs.ravel()[finite_endpoints]
    order = np.argsort(endpoint_costs)
    candidate_endpoints = finite_endpoints[
        order[: min(len(order), MAX_BRANCH_ENDPOINT_CANDIDATES)]
    ]
    candidate_paths = [
        reconstruct_path(int(endpoint), back_pointers, grid_points)
        for endpoint in candidate_endpoints
    ]
    candidate_costs = np.asarray(
        [float(costs.ravel()[endpoint]) for endpoint in candidate_endpoints]
    )
    cost_low = float(np.min(candidate_costs))
    cost_high = float(np.percentile(candidate_costs, 90.0))
    cost_scale = max(cost_high - cost_low, 1.0e-12)

    boundary_indices = np.linspace(
        0,
        len(back_pointers) - 1,
        branching_levels + 1,
    ).round().astype(int).tolist()

    def coarse_box(point: np.ndarray):
        box_width = 2.0 * SPACE_HALF_WIDTH / TESSELLATION_SPATIAL_DIVISIONS
        box_indices = np.floor(
            (np.asarray(point) + SPACE_HALF_WIDTH) / box_width
        ).astype(int)
        box_indices = np.clip(
            box_indices, 0, TESSELLATION_SPATIAL_DIVISIONS - 1
        )
        key = tuple(int(index) for index in box_indices)
        center = tuple(
            -SPACE_HALF_WIDTH + (index + 0.5) * box_width
            for index in box_indices
        )
        return key, center

    root = {
        "coordinate": tuple(candidate_paths[0][0]),
        "children": {},
        "indices": [],
    }
    for candidate_index, path in enumerate(candidate_paths):
        node = root
        for time_index in boundary_indices[1:]:
            box_key, box_center = coarse_box(path[time_index])
            node = node["children"].setdefault(
                box_key,
                {
                    "coordinate": box_center,
                    "children": {},
                    "indices": [],
                },
            )
        node["indices"].append(candidate_index)

    def prune_to_regular_subtree(node: dict, depth: int):
        if depth == branching_levels:
            if not node["indices"]:
                return None
            leaf_index = min(
                node["indices"], key=lambda index: candidate_costs[index]
            )
            return {
                "coordinate": node["coordinate"],
                "minimum_cost": float(candidate_costs[leaf_index]),
                "children": [],
                "leaf_index": int(leaf_index),
            }

        viable_children = []
        for child in node["children"].values():
            pruned_child = prune_to_regular_subtree(child, depth + 1)
            if pruned_child is not None:
                viable_children.append(pruned_child)
        if len(viable_children) < branch_factor:
            return None

        selected_children = select_diverse_tree_children(
            viable_children,
            branch_factor,
            cost_low,
            cost_scale,
        )
        return {
            "coordinate": node["coordinate"],
            "minimum_cost": min(
                child["minimum_cost"] for child in selected_children
            ),
            "children": selected_children,
            "leaf_index": None,
        }

    selected_tree = prune_to_regular_subtree(root, 0)
    if selected_tree is None:
        return [], [], [], [], boundary_indices

    selected_indices: list[int] = []
    family_indices: list[int] = []

    def collect_leaves(node: dict, family_index: int | None = None):
        if node["leaf_index"] is not None:
            selected_indices.append(int(node["leaf_index"]))
            family_indices.append(int(family_index))
            return
        for child_index, child in enumerate(node["children"]):
            next_family = child_index if family_index is None else family_index
            collect_leaves(child, next_family)

    collect_leaves(selected_tree)

    paths = [candidate_paths[index] for index in selected_indices]
    path_costs = [float(candidate_costs[index]) for index in selected_indices]
    path_bottlenecks = [
        float(bottlenecks.ravel()[candidate_endpoints[index]])
        for index in selected_indices
    ]
    return (
        paths,
        path_bottlenecks,
        path_costs,
        family_indices,
        boundary_indices,
    )


def find_admissible_crossings(
    times: np.ndarray,
    grid_points: np.ndarray,
    clearances: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    start_grid_index: tuple[int, int],
    branch_factor: int,
    branching_levels: int,
):
    """Find a regular block-level tree using one precomputed transition DAG."""

    grid_size = grid_points.shape[0]
    spatial_step = float(grid_points[0, 1, 0] - grid_points[0, 0, 0])
    dt = float(times[1] - times[0])
    shifts = allowed_grid_shifts(spatial_step, dt)
    maximum_spatial_step = max(shift[2] for shift in shifts)

    # Collision geometry is the expensive part.  Compute it exactly once for
    # every fine-grid transition, then reuse it for all descendants.
    transition_valid = np.zeros(
        (len(times) - 1, len(shifts), grid_size, grid_size), dtype=bool
    )
    transition_cost = np.full(
        (len(times) - 1, len(shifts), grid_size, grid_size),
        np.inf,
        dtype=np.float32,
    )
    for time_index in range(1, len(times)):
        for shift_index, (
            row_shift,
            column_shift,
            spatial_distance,
        ) in enumerate(shifts):
            previous_rows, current_rows = overlapping_slices(
                grid_size, row_shift
            )
            previous_columns, current_columns = overlapping_slices(
                grid_size, column_shift
            )
            previous_slice = (previous_rows, previous_columns)
            current_slice = (current_rows, current_columns)
            previous_points = grid_points[previous_slice].reshape(-1, 2)
            current_points = grid_points[current_slice].reshape(-1, 2)
            interval_clearance = transition_clearance(
                previous_points,
                current_points,
                centers[time_index - 1],
                centers[time_index],
                radii,
            ).reshape(grid_points[current_slice].shape[:2])
            local_clearance = np.minimum(
                clearances[time_index][current_slice], interval_clearance
            )
            valid = local_clearance >= PATH_CLEARANCE_FLOOR
            clearance_deficit = np.maximum(
                0.0, PREFERRED_PATH_CLEARANCE - local_clearance
            ) / PREFERRED_PATH_CLEARANCE
            continuous_clearance_cost = 1.0 / (
                0.15 + np.maximum(local_clearance, PATH_CLEARANCE_FLOOR)
            )
            increment = (
                MOVEMENT_SHORTFALL_COST_WEIGHT
                * (maximum_spatial_step - spatial_distance) ** 2
                + LOW_CLEARANCE_COST_WEIGHT * clearance_deficit**2
                + CLEARANCE_PATH_COST_WEIGHT * continuous_clearance_cost
            )
            transition_valid[
                time_index - 1, shift_index
            ][current_slice] = valid
            transition_cost[
                time_index - 1, shift_index
            ][current_slice] = increment.astype(np.float32)

    flat_indices = np.arange(grid_size**2, dtype=np.int32).reshape(
        grid_size, grid_size
    )

    def segment_tree(start_time: int, end_time: int, start_flat_index: int):
        costs = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
        costs.flat[start_flat_index] = 0.0
        back = np.full(
            (end_time - start_time + 1, grid_size, grid_size),
            -1,
            dtype=np.int32,
        )
        for current_time in range(start_time + 1, end_time + 1):
            next_costs = np.full_like(costs, np.inf)
            next_back = back[current_time - start_time]
            for shift_index, (
                row_shift,
                column_shift,
                _spatial_distance,
            ) in enumerate(shifts):
                previous_rows, current_rows = overlapping_slices(
                    grid_size, row_shift
                )
                previous_columns, current_columns = overlapping_slices(
                    grid_size, column_shift
                )
                previous_slice = (previous_rows, previous_columns)
                current_slice = (current_rows, current_columns)
                previous_costs = costs[previous_slice]
                increments = transition_cost[
                    current_time - 1, shift_index
                ][current_slice]
                valid = transition_valid[
                    current_time - 1, shift_index
                ][current_slice]
                candidate_costs = previous_costs + increments
                current_cost_view = next_costs[current_slice]
                better = (
                    np.isfinite(previous_costs)
                    & valid
                    & (candidate_costs < current_cost_view)
                )
                if np.any(better):
                    current_cost_view[better] = candidate_costs[better]
                    previous_index_view = flat_indices[previous_slice]
                    next_back_view = next_back[current_slice]
                    next_back_view[better] = previous_index_view[better]
            costs = next_costs
            if not np.any(np.isfinite(costs)):
                return None, None
        return costs, back

    def reconstruct_segment(
        endpoint_flat_index: int,
        back: np.ndarray,
    ) -> np.ndarray:
        flat_grid_points = grid_points.reshape(-1, 2)
        segment = np.empty((len(back), 2), dtype=float)
        current_flat_index = int(endpoint_flat_index)
        segment[-1] = flat_grid_points[current_flat_index]
        for local_time in range(len(back) - 1, 0, -1):
            current_flat_index = int(back[local_time].flat[current_flat_index])
            if current_flat_index < 0:
                raise RuntimeError("Broken back-pointer in branch segment.")
            segment[local_time - 1] = flat_grid_points[current_flat_index]
        return segment

    def coarse_box(point: np.ndarray):
        box_width = 2.0 * SPACE_HALF_WIDTH / TESSELLATION_SPATIAL_DIVISIONS
        indices = np.floor(
            (np.asarray(point) + SPACE_HALF_WIDTH) / box_width
        ).astype(int)
        indices = np.clip(indices, 0, TESSELLATION_SPATIAL_DIVISIONS - 1)
        key = tuple(int(index) for index in indices)
        center = tuple(
            -SPACE_HALF_WIDTH + (index + 0.5) * box_width
            for index in indices
        )
        return key, center

    def endpoint_boxes(costs: np.ndarray):
        grouped: dict[tuple[int, int], dict] = {}
        for flat_index in np.flatnonzero(np.isfinite(costs.ravel())):
            point = grid_points.reshape(-1, 2)[flat_index]
            key, center = coarse_box(point)
            group = grouped.setdefault(
                key,
                {"coordinate": center, "state_candidates": []},
            )
            group["state_candidates"].append(
                (float(costs.ravel()[flat_index]), int(flat_index))
            )
        for group in grouped.values():
            group["state_candidates"].sort(key=lambda item: item[0])
            group["state_candidates"] = group["state_candidates"][:4]
        return list(grouped.values())

    boundary_indices = np.linspace(
        0, len(times) - 1, branching_levels + 1
    ).round().astype(int).tolist()
    memo: dict[tuple[int, int], dict | None] = {}

    def build_regular_node(level: int, start_flat_index: int):
        memo_key = (level, start_flat_index)
        if memo_key in memo:
            return memo[memo_key]
        start_time = boundary_indices[level]
        end_time = boundary_indices[level + 1]
        segment_costs, back = segment_tree(
            start_time, end_time, start_flat_index
        )
        if segment_costs is None:
            memo[memo_key] = None
            return None

        successful_boxes = []
        for box in endpoint_boxes(segment_costs):
            successful_child = None
            for segment_cost, endpoint_flat_index in box["state_candidates"]:
                if level + 1 == branching_levels:
                    subtree = None
                    future_cost = 0.0
                else:
                    subtree = build_regular_node(level + 1, endpoint_flat_index)
                    if subtree is None:
                        continue
                    future_cost = float(subtree["minimum_cost"])
                successful_child = {
                    "coordinate": box["coordinate"],
                    "minimum_cost": float(segment_cost + future_cost),
                    "segment_cost": float(segment_cost),
                    "segment": reconstruct_segment(endpoint_flat_index, back),
                    "subtree": subtree,
                }
                break
            if successful_child is not None:
                successful_boxes.append(successful_child)

        if len(successful_boxes) < branch_factor:
            memo[memo_key] = None
            return None
        local_cost_low = min(
            child["minimum_cost"] for child in successful_boxes
        )
        local_cost_high = max(
            child["minimum_cost"] for child in successful_boxes
        )
        selected_children = select_diverse_tree_children(
            successful_boxes,
            branch_factor,
            local_cost_low,
            max(local_cost_high - local_cost_low, 1.0e-12),
        )
        node = {
            "minimum_cost": min(
                child["minimum_cost"] for child in selected_children
            ),
            "children": selected_children,
        }
        memo[memo_key] = node
        return node

    start_flat_index = int(flat_indices[start_grid_index])
    selected_tree = build_regular_node(0, start_flat_index)
    if selected_tree is None:
        return [], [], [], [], boundary_indices

    paths: list[np.ndarray] = []
    path_costs: list[float] = []
    family_indices: list[int] = []

    def collect_paths(
        node: dict,
        prefix: np.ndarray | None = None,
        accumulated_cost: float = 0.0,
        family_index: int | None = None,
    ):
        for child_index, child in enumerate(node["children"]):
            segment = child["segment"]
            full_prefix = (
                segment.copy()
                if prefix is None
                else np.vstack((prefix, segment[1:]))
            )
            next_cost = accumulated_cost + float(child["segment_cost"])
            next_family = child_index if family_index is None else family_index
            if child["subtree"] is None:
                paths.append(full_prefix)
                path_costs.append(next_cost)
                family_indices.append(int(next_family))
            else:
                collect_paths(
                    child["subtree"],
                    full_prefix,
                    next_cost,
                    next_family,
                )

    collect_paths(selected_tree)
    # Exact bottlenecks are computed by verify_path immediately afterwards.
    return (
        paths,
        [float("nan")] * len(paths),
        path_costs,
        family_indices,
        boundary_indices,
    )


def verify_path(
    path: np.ndarray,
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
):
    speeds = np.linalg.norm(np.diff(path, axis=0), axis=1) / np.diff(times)
    interval_clearances = np.empty(len(times) - 1)
    for time_index in range(1, len(times)):
        interval_clearances[time_index - 1] = transition_clearance(
            path[time_index - 1 : time_index],
            path[time_index : time_index + 1],
            centers[time_index - 1],
            centers[time_index],
            radii,
        )[0]

    maximum_speed = float(np.max(speeds))
    minimum_clearance = float(np.min(interval_clearances))
    if maximum_speed > MAX_TARGET_SPEED + 1.0e-9:
        raise RuntimeError("The computed path violates the speed bound.")
    if minimum_clearance < NUMERICAL_SAFETY_MARGIN - 1.0e-9:
        raise RuntimeError("The computed path intersects a detection tube.")
    return maximum_speed, minimum_clearance, speeds, interval_clearances


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def tube_mesh(
    particle_centers: np.ndarray,
    radius: float,
    times: np.ndarray,
    time_stride: int = 1,
    angle_points: int = TUBE_ANGLE_POINTS,
):
    time_indices = np.arange(0, len(times), time_stride)
    if time_indices[-1] != len(times) - 1:
        time_indices = np.append(time_indices, len(times) - 1)
    theta = np.linspace(0.0, 2.0 * np.pi, angle_points)
    selected_centers = particle_centers[time_indices]
    x = selected_centers[:, 0, None] + radius * np.cos(theta)[None, :]
    y = selected_centers[:, 1, None] + radius * np.sin(theta)[None, :]
    z = np.repeat(times[time_indices, None], angle_points, axis=1)
    return x, y, z


def choose_display_particles(
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray] | None,
):
    intersects_window = np.any(
        np.max(np.abs(centers), axis=2)
        <= SPACE_HALF_WIDTH + radii[None, :],
        axis=0,
    )
    candidates = np.flatnonzero(intersects_window)
    if len(candidates) <= MAX_DISPLAY_TUBES:
        return candidates

    if not paths:
        distance_to_focus = np.min(
            np.linalg.norm(centers[:, candidates, :], axis=2)
            - radii[candidates][None, :],
            axis=0,
        )
    else:
        path_array = np.stack(paths, axis=0)
        distance_to_focus = np.min(
            np.linalg.norm(
                centers[None, :, candidates, :]
                - path_array[:, :, None, :],
                axis=3,
            )
            - radii[candidates][None, None, :],
            axis=(0, 1),
        )
    order = np.argsort(distance_to_focus)
    return candidates[order[:MAX_DISPLAY_TUBES]]


def draw_space_time_box(ax):
    h = SPACE_HALF_WIDTH
    lower = 0.0
    upper = TOTAL_TIME
    corners = [
        (-h, -h),
        (h, -h),
        (h, h),
        (-h, h),
        (-h, -h),
    ]
    x = [point[0] for point in corners]
    y = [point[1] for point in corners]
    ax.plot(x, y, zs=lower, color="#9ca3af", linewidth=0.65, alpha=0.55)
    ax.plot(x, y, zs=upper, color="#9ca3af", linewidth=0.65, alpha=0.55)
    for corner_x, corner_y in corners[:4]:
        ax.plot(
            [corner_x, corner_x],
            [corner_y, corner_y],
            [lower, upper],
            color="#9ca3af",
            linewidth=0.65,
            alpha=0.55,
        )


def tessellation_lines():
    """Return line segments for a coarse rectangular space-time tessellation."""

    h = SPACE_HALF_WIDTH
    spatial_levels = np.linspace(
        -h, h, TESSELLATION_SPATIAL_DIVISIONS + 1
    )
    time_levels = np.linspace(0.0, TOTAL_TIME, TESSELLATION_TIME_DIVISIONS + 1)
    segments = []

    # Horizontal grids mark the scale-k space-time boxes.
    for time_level in time_levels:
        for coordinate in spatial_levels:
            segments.append(([-h, h], [coordinate, coordinate], [time_level] * 2))
            segments.append(([coordinate, coordinate], [-h, h], [time_level] * 2))

    # A sparse set of vertical edges gives depth without overwhelming the
    # Brownian tubes and the path tree.
    vertical_levels = spatial_levels[::2]
    if vertical_levels[-1] != spatial_levels[-1]:
        vertical_levels = np.append(vertical_levels, spatial_levels[-1])
    for x_coordinate in vertical_levels:
        for y_coordinate in vertical_levels:
            segments.append(
                ([x_coordinate] * 2, [y_coordinate] * 2, [0.0, TOTAL_TIME])
            )
    return segments


def draw_tessellation(ax, color: str = "#6f91a5", alpha: float = 0.13):
    for x, y, z in tessellation_lines():
        ax.plot(x, y, z, color=color, linewidth=0.42, alpha=alpha, zorder=0)


def plotly_tessellation_trace():
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for x, y, z in tessellation_lines():
        x_values.extend([float(x[0]), float(x[1]), None])
        y_values.extend([float(y[0]), float(y[1]), None])
        z_values.extend([float(z[0]), float(z[1]), None])
    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line=dict(color="rgba(92,126,148,0.24)", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="space-time tessellation",
    )


def style_axis(ax):
    ax.set_xlim(-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH)
    ax.set_ylim(-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH)
    ax.set_zlim(0.0, TOTAL_TIME)
    ax.set_xlabel(r"$x_1$", labelpad=8)
    ax.set_ylabel(r"$x_2$", labelpad=8)
    ax.set_zlabel(r"time $t$", labelpad=8)
    ax.set_box_aspect((1.0, 1.0, 1.25))
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.view_init(elev=23, azim=-55)


def save_static_figure(
    output_path: Path,
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray],
    escape_colors: list[str],
    display_particles: np.ndarray,
):
    figure = plt.figure(figsize=(10.5, 10.5), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")

    for particle_index in display_particles:
        x, y, z = tube_mesh(
            centers[:, particle_index],
            radii[particle_index],
            times,
            time_stride=TUBE_TIME_STRIDE,
        )
        axis.plot_surface(
            x,
            y,
            z,
            color=DETECTION_TUBE_COLOR,
            alpha=0.16,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        axis.plot(
            centers[:, particle_index, 0],
            centers[:, particle_index, 1],
            times,
            color=DETECTION_CENTERLINE_COLOR,
            linewidth=0.85,
            alpha=0.78,
        )

    draw_tessellation(axis)
    path_line_width = 0.52 if len(paths) >= 20 else 1.05
    endpoint_size = 10 if len(paths) >= 20 else 24
    for path, color in zip(paths, escape_colors):
        axis.plot(
            path[:, 0],
            path[:, 1],
            times,
            color=color,
            linewidth=path_line_width,
            zorder=20,
        )
        axis.scatter(
            [path[-1, 0]],
            [path[-1, 1]],
            [times[-1]],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=endpoint_size,
            depthshade=False,
            zorder=21,
        )
    axis.scatter(
        [paths[0][0, 0]],
        [paths[0][0, 1]],
        [times[0]],
        color=START_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=36,
        depthshade=False,
        zorder=21,
    )
    draw_space_time_box(axis)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def choose_website_particles(
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray],
    display_particles: np.ndarray,
) -> np.ndarray:
    """Mix path-near tubes with the largest visible radius levels."""

    if len(display_particles) <= MAX_WEBSITE_TUBES:
        return display_particles

    path_array = np.stack(paths, axis=0)
    distances = np.min(
        np.linalg.norm(
            centers[None, :, display_particles, :]
            - path_array[:, :, None, :],
            axis=3,
        )
        - radii[display_particles][None, None, :],
        axis=(0, 1),
    )
    near_count = int(round(0.65 * MAX_WEBSITE_TUBES))
    near_order = np.argsort(distances)
    near_particles = display_particles[near_order[:near_count]]

    remaining_mask = ~np.isin(display_particles, near_particles)
    remaining = display_particles[remaining_mask]
    large_order = np.argsort(radii[remaining])[::-1]
    large_particles = remaining[
        large_order[: MAX_WEBSITE_TUBES - len(near_particles)]
    ]
    selected = np.concatenate((near_particles, large_particles))
    if len(selected) < MAX_WEBSITE_TUBES:
        unused = display_particles[~np.isin(display_particles, selected)]
        selected = np.concatenate(
            (selected, unused[: MAX_WEBSITE_TUBES - len(selected)])
        )
    return selected


def save_website_figure(
    output_path: Path,
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray],
    escape_colors: list[str],
    display_particles: np.ndarray,
):
    """Save a clean 16:9 rendering intended for a research website."""

    background = WEBSITE_BACKGROUND_COLOR
    tube_color = "#557b92"
    centerline_color = "#385d73"
    path_glow = "#ffffff"
    website_particles = choose_website_particles(
        centers, radii, paths, display_particles
    )

    figure = plt.figure(figsize=(12.0, 6.75), facecolor=background)
    axis = figure.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
        facecolor=background,
    )

    # Draw the environment first.  Tubes farther from the path are omitted so
    # the composition remains readable at banner size.
    for rank, particle_index in enumerate(website_particles):
        x, y, z = tube_mesh(
            centers[:, particle_index],
            radii[particle_index],
            times,
            time_stride=1,
            angle_points=24,
        )
        distance_fraction = rank / max(1, len(website_particles) - 1)
        surface_alpha = 0.18 - 0.08 * distance_fraction
        line_alpha = 0.48 - 0.20 * distance_fraction
        axis.plot_surface(
            x,
            y,
            z,
            color=tube_color,
            alpha=surface_alpha,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            zorder=2,
        )
        axis.plot(
            centers[:, particle_index, 0],
            centers[:, particle_index, 1],
            times,
            color=centerline_color,
            linewidth=0.8,
            alpha=line_alpha,
            zorder=3,
        )

    draw_tessellation(axis, color="#4f7991", alpha=0.12)

    # Thin white under-strokes keep the branching paths readable without
    # making them resemble the detection tubes.
    path_line_width = 0.68 if len(paths) >= 20 else 1.15
    glow_line_width = 1.40 if len(paths) >= 20 else 2.4
    endpoint_size = 12 if len(paths) >= 20 else 28
    for path, path_color in zip(paths, escape_colors):
        axis.plot(
            path[:, 0],
            path[:, 1],
            times,
            color=path_glow,
            linewidth=glow_line_width,
            alpha=0.48,
            solid_capstyle="round",
            zorder=20,
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            times,
            color=path_color,
            linewidth=path_line_width,
            alpha=1.0,
            solid_capstyle="round",
            zorder=21,
        )
        axis.scatter(
            [path[-1, 0]],
            [path[-1, 1]],
            [times[-1]],
            color=path_color,
            edgecolor="#ffffff",
            linewidth=0.8,
            s=endpoint_size,
            depthshade=False,
            zorder=22,
        )
    axis.scatter(
        [paths[0][0, 0]],
        [paths[0][0, 1]],
        [times[0]],
        color=START_COLOR,
        edgecolor="#ffffff",
        linewidth=1.0,
        s=58,
        depthshade=False,
        zorder=23,
    )

    axis.set_xlim(-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH)
    axis.set_ylim(-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH)
    axis.set_zlim(0.0, TOTAL_TIME)
    axis.set_box_aspect((1.0, 1.0, 1.30))
    axis.view_init(elev=18, azim=-52)
    axis.set_proj_type("persp", focal_length=0.92)
    axis.set_axis_off()

    # Keep every terminal branch marker inside the banner.  The earlier
    # single-path composition could use aggressive negative margins, whereas
    # a branching crown can reach several different sides of the box.
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    figure.savefig(
        output_path,
        dpi=200,
        facecolor=background,
        edgecolor="none",
    )
    plt.close(figure)


def build_interactive_figure(
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray] | None,
    escape_colors: list[str],
    display_particles: np.ndarray,
    interval_clearances: list[np.ndarray] | None,
):
    figure = go.Figure()
    figure.add_trace(plotly_tessellation_trace())
    centerline_x = []
    centerline_y = []
    centerline_z = []

    for particle_index in display_particles:
        x, y, z = tube_mesh(
            centers[:, particle_index],
            radii[particle_index],
            times,
            time_stride=TUBE_TIME_STRIDE,
            angle_points=16,
        )
        figure.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=np.zeros_like(x),
                colorscale=[
                    [0.0, DETECTION_TUBE_COLOR_CSS],
                    [1.0, DETECTION_TUBE_COLOR_CSS],
                ],
                showscale=False,
                opacity=0.19,
                name=f"particle {particle_index}",
                hovertemplate=(
                    f"particle {particle_index}<br>"
                    f"radius = {radii[particle_index]:.3f}"
                    "<extra></extra>"
                ),
            )
        )
        centerline_x.extend(centers[:, particle_index, 0].tolist())
        centerline_y.extend(centers[:, particle_index, 1].tolist())
        centerline_z.extend(times.tolist())
        centerline_x.append(None)
        centerline_y.append(None)
        centerline_z.append(None)

    if len(display_particles):
        figure.add_trace(
            go.Scatter3d(
                x=centerline_x,
                y=centerline_y,
                z=centerline_z,
                mode="lines",
                line=dict(color=DETECTION_CENTERLINE_COLOR, width=2),
                opacity=0.78,
                name="Brownian particle centers",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if paths is not None and interval_clearances is not None:
        for branch_index, (path, color, clearances) in enumerate(
            zip(paths, escape_colors, interval_clearances), start=1
        ):
            path_clearance = np.r_[clearances[0], clearances]
            custom_data = np.column_stack((times, path_clearance))
            figure.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=times,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"admissible branch {branch_index}",
                    customdata=custom_data,
                    hovertemplate=(
                        f"branch {branch_index}<br>"
                        "t = %{customdata[0]:.3f}<br>"
                        "interval clearance = %{customdata[1]:.3f}"
                        "<extra></extra>"
                    ),
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[path[-1, 0]],
                    y=[path[-1, 1]],
                    z=[times[-1]],
                    mode="markers",
                    marker=dict(size=5, color=color),
                    text=[f"exit {branch_index}"],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter3d(
                x=[paths[0][0, 0]],
                y=[paths[0][0, 1]],
                z=[times[0]],
                mode="markers",
                marker=dict(size=7, color=START_COLOR),
                text=["common entrance"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    figure.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title="x1",
                range=[-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH],
                backgroundcolor="white",
                gridcolor="#e5e7eb",
            ),
            yaxis=dict(
                title="x2",
                range=[-SPACE_HALF_WIDTH, SPACE_HALF_WIDTH],
                backgroundcolor="white",
                gridcolor="#e5e7eb",
            ),
            zaxis=dict(
                title="time t",
                range=[0.0, TOTAL_TIME],
                backgroundcolor="white",
                gridcolor="#e5e7eb",
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.25),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=1.15)),
            dragmode="orbit",
        ),
    )
    return figure


def save_interactive_figure(
    output_path: Path,
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray] | None,
    escape_colors: list[str],
    display_particles: np.ndarray,
    interval_clearances: list[np.ndarray] | None,
):
    figure = build_interactive_figure(
        times,
        centers,
        radii,
        paths,
        escape_colors,
        display_particles,
        interval_clearances,
    )
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )


def save_video(
    output_path: Path,
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    paths: list[np.ndarray],
    escape_colors: list[str],
    display_particles: np.ndarray,
    interval_clearances: list[np.ndarray],
):
    if not animation.writers.is_available("ffmpeg"):
        raise SystemExit(
            "FFmpeg is required for MP4 output. On macOS: brew install ffmpeg"
        )

    figure = plt.figure(figsize=(9.5, 9.5), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")

    surface_artists = []
    environment_lines = []
    for particle_index in display_particles:
        x, y, z = tube_mesh(
            centers[:, particle_index],
            radii[particle_index],
            times,
            time_stride=max(TUBE_TIME_STRIDE, 3),
            angle_points=14,
        )
        surface = axis.plot_surface(
            x,
            y,
            z,
            color=DETECTION_TUBE_COLOR,
            alpha=0.10,
            linewidth=0.0,
            shade=False,
        )
        surface.set_visible(False)
        surface_artists.append(surface)
        centerline, = axis.plot(
            [],
            [],
            [],
            color=DETECTION_CENTERLINE_COLOR,
            linewidth=0.6,
            alpha=0.58,
        )
        environment_lines.append(centerline)

    draw_space_time_box(axis)
    draw_tessellation(axis, alpha=0.10)
    style_axis(axis)

    escape_lines = []
    current_targets = []
    video_path_width = 0.68 if len(paths) >= 20 else 1.25
    video_marker_size = 10 if len(paths) >= 20 else 27
    for color in escape_colors:
        escape_line, = axis.plot(
            [], [], [], color=color, linewidth=video_path_width
        )
        current_target = axis.scatter(
            [],
            [],
            [],
            color=color,
            edgecolor="white",
            s=video_marker_size,
            depthshade=False,
        )
        escape_lines.append(escape_line)
        current_targets.append(current_target)
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    cross_sections = []
    for _ in display_particles:
        line, = axis.plot(
            [],
            [],
            [],
            color=DETECTION_CENTERLINE_COLOR,
            linewidth=1.5,
            alpha=0.95,
        )
        cross_sections.append(line)

    slice_outline, = axis.plot([], [], [], color="#374151", linewidth=0.8, alpha=0.7)
    status_text = axis.text2D(
        0.03,
        0.96,
        "",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#374151",
    )

    environment_frame_count = max(
        2, int(round(VIDEO_FPS * ENVIRONMENT_VIDEO_SECONDS))
    )
    pause_frame_count = max(1, int(round(VIDEO_FPS * FUTURE_PAUSE_SECONDS)))
    escape_frame_count = max(
        2, int(round(VIDEO_FPS * ESCAPE_VIDEO_SECONDS))
    )
    environment_indices = np.linspace(
        0, len(times) - 1, environment_frame_count
    ).round().astype(int)
    escape_indices = np.linspace(
        0, len(times) - 1, escape_frame_count
    ).round().astype(int)
    total_frames = (
        environment_frame_count + pause_frame_count + escape_frame_count
    )

    def set_environment_history(time_index):
        for line, particle_index in zip(
            environment_lines, display_particles
        ):
            line.set_data_3d(
                centers[: time_index + 1, particle_index, 0],
                centers[: time_index + 1, particle_index, 1],
                times[: time_index + 1],
            )

    def set_cross_sections(time_index):
        current_time = times[time_index]
        for line, particle_index in zip(cross_sections, display_particles):
            center = centers[time_index, particle_index]
            radius = radii[particle_index]
            line.set_data_3d(
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                np.full_like(theta, current_time),
            )
        h = SPACE_HALF_WIDTH
        slice_outline.set_data_3d(
            [-h, h, h, -h, -h],
            [-h, -h, h, h, -h],
            np.full(5, current_time),
        )

    def hide_cross_sections():
        for line in cross_sections:
            line.set_data_3d([], [], [])
        slice_outline.set_data_3d([], [], [])

    def update(frame_number):
        if frame_number < environment_frame_count:
            # Phase 1: reveal the whole random environment.  The target is not
            # shown and no path choice is visualized yet.
            time_index = int(environment_indices[frame_number])
            set_environment_history(time_index)
            set_cross_sections(time_index)
            for surface in surface_artists:
                surface.set_visible(False)
            for escape_line, current_target in zip(
                escape_lines, current_targets
            ):
                escape_line.set_data_3d([], [], [])
                current_target._offsets3d = ([], [], [])
            status_text.set_text(
                "phase 1: drawing the environment\n"
                f"environment time: {times[time_index]:5.2f} / "
                f"{TOTAL_TIME:.2f}"
            )
        elif frame_number < environment_frame_count + pause_frame_count:
            # The complete future is now visible before the target starts.
            set_environment_history(len(times) - 1)
            hide_cross_sections()
            for surface in surface_artists:
                surface.set_visible(True)
            for escape_line, current_target in zip(
                escape_lines, current_targets
            ):
                escape_line.set_data_3d([], [], [])
                current_target._offsets3d = ([], [], [])
            status_text.set_text(
                "environment complete\n"
                "the full future is known before branching"
            )
        else:
            # Phase 2: the environment remains completely visible while the
            # target follows the path selected with full future knowledge.
            escape_frame = (
                frame_number - environment_frame_count - pause_frame_count
            )
            time_index = int(escape_indices[escape_frame])
            current_time = times[time_index]
            set_environment_history(len(times) - 1)
            set_cross_sections(time_index)
            for surface in surface_artists:
                surface.set_visible(True)
            current_clearances = []
            current_speeds = []
            for path, clearances, escape_line, current_target in zip(
                paths,
                interval_clearances,
                escape_lines,
                current_targets,
            ):
                escape_line.set_data_3d(
                    path[: time_index + 1, 0],
                    path[: time_index + 1, 1],
                    times[: time_index + 1],
                )
                current_target._offsets3d = (
                    [path[time_index, 0]],
                    [path[time_index, 1]],
                    [current_time],
                )
                current_clearances.append(
                    clearances[max(0, time_index - 1)]
                    if len(clearances)
                    else np.inf
                )
                current_speeds.append(
                    np.linalg.norm(path[time_index] - path[time_index - 1])
                    / (times[time_index] - times[time_index - 1])
                    if time_index > 0
                    else 0.0
                )
            status_text.set_text(
                "phase 2: branching escapes with the full future known\n"
                f"escape time: {current_time:5.2f} / {TOTAL_TIME:.2f}\n"
                f"branches: {len(paths)}\n"
                f"maximum branch speed: {max(current_speeds):.3f} / "
                f"{MAX_TARGET_SPEED:.3f}\n"
                f"minimum branch clearance: {min(current_clearances):.3f}"
            )

        axis.view_init(elev=23, azim=-55 + 0.10 * frame_number)
        return [
            *escape_lines,
            *current_targets,
            slice_outline,
            status_text,
            *surface_artists,
            *environment_lines,
            *cross_sections,
        ]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=total_frames,
        interval=1000 / VIDEO_FPS,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=VIDEO_FPS,
        bitrate=3200,
        metadata={"title": "Dynamic Brownian detection field"},
    )
    movie.save(output_path, writer=writer, dpi=125)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main():
    global ESCAPE_COLOR, ESCAPE_COLORS, ESCAPE_FAMILY_COLORS

    # Presentation colors are deliberately independent of SEED: the stochastic
    # environment remains reproducible while every rendering receives a fresh
    # random golden-angle pastel sequence.
    color_rng = np.random.default_rng()
    seed_sequence = np.random.SeedSequence(SEED)
    run_seed = int(seed_sequence.entropy)
    times = np.linspace(0.0, TOTAL_TIME, TIME_STEPS)

    grid_limit = SPACE_HALF_WIDTH - PATH_BOUNDARY_MARGIN
    spatial_axis = np.linspace(
        -grid_limit, grid_limit, NUM_SPATIAL_GRID_POINTS
    )
    grid_x, grid_y = np.meshgrid(spatial_axis, spatial_axis, indexing="xy")
    grid_points = np.stack((grid_x, grid_y), axis=2)

    # First sample one complete environment, independently of whether it will
    # admit a crossing.  Only after that environment has been fixed do we run
    # the deterministic path search.  In particular, failed environments are
    # never discarded and silently resampled.
    environment_seed, path_seed = seed_sequence.spawn(2)
    environment_rng = np.random.default_rng(environment_seed)
    initial_positions, radii, centers, poisson_half_width = (
        simulate_environment(environment_rng, times)
    )
    clearances = endpoint_clearances(grid_points, centers, radii)
    path_rng = np.random.default_rng(path_seed)
    available_start_mask = (
        np.linalg.norm(grid_points, axis=2) <= ENTRANCE_RADIUS
    ) & (clearances[0] >= PATH_CLEARANCE_FLOOR)
    available_starts = np.argwhere(available_start_mask)
    if len(available_starts) == 0:
        raise RuntimeError(
            "The generated environment has no vacant initial point in the "
            "chosen entrance disk. The environment was not resampled."
        )
    start_grid_index_array = available_starts[
        path_rng.integers(len(available_starts))
    ]
    start_grid_index = tuple(int(value) for value in start_grid_index_array)
    start_point = grid_points[start_grid_index].copy()
    (
        paths,
        bottleneck_clearances,
        path_costs,
        branch_family_indices,
        branching_boundary_indices,
    ) = find_admissible_crossings(
        times,
        grid_points,
        clearances,
        centers,
        radii,
        start_grid_index,
        BRANCHING_FACTOR,
        TESSELLATION_TIME_DIVISIONS,
    )

    if not paths:
        raise RuntimeError(
            "This fixed random environment does not contain a complete "
            f"{BRANCHING_FACTOR}-ary admissible tree with "
            f"{TESSELLATION_TIME_DIVISIONS} generations on the chosen finite "
            "grid. It was not resampled. Run again for a new environment or "
            "reduce the branching factor/number of time layers explicitly."
        )
    ESCAPE_COLORS = random_pastel_escape_colors(color_rng, len(paths))
    ESCAPE_FAMILY_COLORS = list(ESCAPE_COLORS)
    ESCAPE_COLOR = ESCAPE_COLORS[0]
    branch_verifications = [
        verify_path(path, times, centers, radii) for path in paths
    ]
    maximum_speeds = [verification[0] for verification in branch_verifications]
    minimum_clearances = [verification[1] for verification in branch_verifications]
    bottleneck_clearances = list(minimum_clearances)
    speeds = [verification[2] for verification in branch_verifications]
    interval_clearances = [
        verification[3] for verification in branch_verifications
    ]
    maximum_speed = float(max(maximum_speeds))
    minimum_clearance = float(min(minimum_clearances))
    spatial_path_lengths = [
        float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        for path in paths
    ]
    spatial_path_length = float(np.mean(spatial_path_lengths))
    signed_spatial_turns_by_branch = []
    for path in paths:
        unwrapped_angles = np.unwrap(np.arctan2(path[:, 1], path[:, 0]))
        signed_spatial_turns_by_branch.append(
            float(
                (unwrapped_angles[-1] - unwrapped_angles[0])
                / (2.0 * np.pi)
            )
        )
    signed_spatial_turns = float(np.mean(signed_spatial_turns_by_branch))
    continuous_speed_budget = MAX_TARGET_SPEED * (times[-1] - times[0])
    speed_budget_fractions = [
        path_length / continuous_speed_budget
        for path_length in spatial_path_lengths
    ]
    speed_budget_fraction = float(max(speed_budget_fractions))

    display_particles = choose_display_particles(centers, radii, paths)
    output_directory = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"escapist_unique_path_colors_v10_{run_id}"
    image_path = output_directory / f"{base_name}.png"
    website_image_path = (
        output_directory
        / f"{base_name}_website_25_unique_pastels.png"
    )
    interactive_path = output_directory / f"{base_name}_interactive.html"
    video_path = output_directory / f"{base_name}.mp4"

    print(f"run seed: {run_seed}")
    print(
        "unique golden-angle pastel path colors: "
        f"{', '.join(ESCAPE_COLORS)}"
    )
    print("random environments generated: 1")
    print(
        "random initial point selected after the environment: "
        f"({start_point[0]:.6f}, {start_point[1]:.6f})"
    )
    print(f"Poisson particles in truncated window: {len(radii)}")
    radius_levels = np.rint(
        np.log(radii / RADIUS_BASE) / np.log(RADIUS_LEVEL_GROWTH)
    ).astype(int)
    realized_levels, realized_counts = np.unique(
        radius_levels, return_counts=True
    )
    expected_requested_moment = (
        RADIUS_BASE**RADIUS_MOMENT_ORDER
        * (1.0 - RADIUS_LEVEL_TAIL)
        / (
            1.0
            - RADIUS_LEVEL_TAIL
            * RADIUS_LEVEL_GROWTH**RADIUS_MOMENT_ORDER
        )
    )
    print(
        "radius law: P(K=k)=(1-q)q^k, "
        f"R_k={RADIUS_BASE:.3f}*{RADIUS_LEVEL_GROWTH:.3f}^k, "
        f"q={RADIUS_LEVEL_TAIL:.3f}"
    )
    print(
        f"theoretical E[R^{RADIUS_MOMENT_ORDER:g}]: "
        f"{expected_requested_moment:.6f}"
    )
    print("radius levels in this realization:")
    for level, count in zip(realized_levels, realized_counts):
        level_probability = (
            (1.0 - RADIUS_LEVEL_TAIL) * RADIUS_LEVEL_TAIL**level
        )
        level_radius = RADIUS_BASE * RADIUS_LEVEL_GROWTH**level
        print(
            f"  k={level}: R_k={level_radius:.6f}, "
            f"p_k={level_probability:.6f}, count={count}"
        )
    print(f"displayed detection tubes: {len(display_particles)}")
    print(f"recursive branching factor: {BRANCHING_FACTOR}")
    print(f"branching generations: {TESSELLATION_TIME_DIVISIONS}")
    print(f"terminal admissible escape paths: {len(paths)}")
    print(
        "branching time indices: "
        + ", ".join(str(index) for index in branching_boundary_indices)
    )
    print(
        "space-time tessellation: "
        f"{TESSELLATION_SPATIAL_DIVISIONS} x "
        f"{TESSELLATION_SPATIAL_DIVISIONS} x "
        f"{TESSELLATION_TIME_DIVISIONS}"
    )
    branch_rows = list(zip(
        maximum_speeds,
        minimum_clearances,
        bottleneck_clearances,
        path_costs,
        spatial_path_lengths,
    ))
    rows_to_print = branch_rows if len(branch_rows) <= 20 else branch_rows[:10]
    for branch_index, (
        branch_maximum_speed,
        branch_minimum_clearance,
        branch_bottleneck,
        branch_cost,
        branch_length,
    ) in enumerate(
        rows_to_print,
        start=1,
    ):
        print(
            f"  branch {branch_index}: "
            f"max speed={branch_maximum_speed:.6f}, "
            f"min clearance={branch_minimum_clearance:.6f}, "
            f"bottleneck={branch_bottleneck:.6f}, "
            f"tree cost={branch_cost:.6f}, "
            f"length={branch_length:.6f}"
        )
    if len(branch_rows) > len(rows_to_print):
        print(
            f"  ... {len(branch_rows) - len(rows_to_print)} further "
            "validated branches are available in the returned statistics."
        )
    print(
        f"maximum speed over all branches: {maximum_speed:.6f} "
        f"<= {MAX_TARGET_SPEED:.6f}"
    )
    print(f"minimum clearance over all branches: {minimum_clearance:.6f}")
    print(f"mean spatial path length: {spatial_path_length:.6f}")
    print(
        "largest continuous speed budget used: "
        f"{100.0 * speed_budget_fraction:.2f}%"
    )
    print(f"initial Poisson half-width: {poisson_half_width:.3f}")

    save_static_figure(
        image_path,
        times,
        centers,
        radii,
        paths,
        ESCAPE_COLORS,
        display_particles,
    )
    print(f"image saved to: {image_path}")

    save_website_figure(
        website_image_path,
        times,
        centers,
        radii,
        paths,
        ESCAPE_COLORS,
        display_particles,
    )
    print(f"website image saved to: {website_image_path}")

    save_interactive_figure(
        interactive_path,
        times,
        centers,
        radii,
        paths,
        ESCAPE_COLORS,
        display_particles,
        interval_clearances,
    )
    print(f"interactive 3D figure saved to: {interactive_path}")
    if OPEN_INTERACTIVE_HTML:
        webbrowser.open(interactive_path.as_uri())

    if SAVE_VIDEO:
        save_video(
            video_path,
            times,
            centers,
            radii,
            paths,
            ESCAPE_COLORS,
            display_particles,
            interval_clearances,
        )
        print(f"video saved to: {video_path}")

    return {
        "run_seed": run_seed,
        "escape_color": ESCAPE_COLOR,
        "escape_colors": list(ESCAPE_COLORS),
        "escape_family_colors": list(ESCAPE_FAMILY_COLORS),
        "branching_factor": int(BRANCHING_FACTOR),
        "branching_generations": int(TESSELLATION_TIME_DIVISIONS),
        "branching_boundary_indices": list(branching_boundary_indices),
        "branch_count": int(len(paths)),
        "start_point": [float(start_point[0]), float(start_point[1])],
        "particle_count": int(len(radii)),
        "displayed_tube_count": int(len(display_particles)),
        "maximum_speed": maximum_speed,
        "minimum_clearance": minimum_clearance,
        "bottleneck_clearance": float(min(bottleneck_clearances)),
        "spatial_path_length": spatial_path_length,
        "speed_budget_fraction": speed_budget_fraction,
        "signed_spatial_turns": signed_spatial_turns,
        "tessellation": {
            "spatial_divisions": int(TESSELLATION_SPATIAL_DIVISIONS),
            "time_divisions": int(TESSELLATION_TIME_DIVISIONS),
        },
        "branch_statistics": [
            {
                "branch": branch_index,
                "primary_family": int(
                    branch_family_indices[branch_index - 1] + 1
                ),
                "color": ESCAPE_COLORS[branch_index - 1],
                "maximum_speed": float(maximum_speeds[branch_index - 1]),
                "minimum_clearance": float(
                    minimum_clearances[branch_index - 1]
                ),
                "bottleneck_clearance": float(
                    bottleneck_clearances[branch_index - 1]
                ),
                "tree_cost": float(path_costs[branch_index - 1]),
                "spatial_path_length": float(
                    spatial_path_lengths[branch_index - 1]
                ),
            }
            for branch_index in range(1, len(paths) + 1)
        ],
        "radius_moment_order": float(RADIUS_MOMENT_ORDER),
        "expected_requested_moment": float(expected_requested_moment),
        "radius_levels": [
            {
                "level": int(level),
                "radius": float(
                    RADIUS_BASE * RADIUS_LEVEL_GROWTH**int(level)
                ),
                "count": int(count),
            }
            for level, count in zip(realized_levels, realized_counts)
        ],
        "image_path": str(image_path),
        "website_image_path": str(website_image_path),
        "interactive_path": str(interactive_path),
        "video_path": str(video_path) if SAVE_VIDEO else None,
    }


if __name__ == "__main__":
    main()
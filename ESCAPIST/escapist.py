from __future__ import annotations

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
POISSON_INTENSITY = 0.065
RADIUS_BASE = 0.40
RADIUS_LEVEL_GROWTH = 1.35
RADIUS_LEVEL_TAIL = 0.35  # q

# The target is a point.  Increase TARGET_RADIUS to simulate a thick target.
TARGET_RADIUS = 0.0
MAX_TARGET_SPEED = 4.5
NUM_SPATIAL_GRID_POINTS = 61
PATH_BOUNDARY_MARGIN = 0.50
NUMERICAL_SAFETY_MARGIN = 0.025
# A slightly larger clearance floor makes separation from the translucent tube
# boundary visible in the rendered figure while remaining a valid vacant path.
PATH_CLEARANCE_FLOOR = 0.10

# The escape path is selected among valid paths by following a slow rotating
# spatial guide.  This gives the admissible curve visible spatial turns instead
# of the nearly direct maximum-clearance route.  The guide is only a preference:
# obstacles and the speed bound always take precedence.
WINDING_GUIDE_TURNS = 1.50
WINDING_GUIDE_RADIUS = 3.00
GUIDE_TRACKING_WEIGHT = 1.00
# The target prefers transitions close to the largest spatial displacement
# allowed by the discrete speed bound.  This is a soft preference, so it can
# slow down whenever maximum-speed motion would collide with the environment.
MOVEMENT_SHORTFALL_COST_WEIGHT = 18.0
PREFERRED_PATH_CLEARANCE = 0.30
LOW_CLEARANCE_COST_WEIGHT = 3.00

# The entrance A is a disk in the lower time face, as in Definition 2.4.
ENTRANCE_RADIUS = 1.40

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
TUBE_ANGLE_POINTS = 18
TUBE_TIME_STRIDE = 1

# Visual identity: the observed set is neutral gray and the escaping target is
# red.  All particles share the same color because they have the same role.
DETECTION_TUBE_COLOR = (0.66, 0.68, 0.71)
DETECTION_TUBE_COLOR_CSS = "rgb(168,173,181)"
DETECTION_CENTERLINE_COLOR = "#626871"
ESCAPE_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def simulate_environment(rng: np.random.Generator, times: np.ndarray):
    """Simulate the truncated marked Poisson process and Brownian motions."""

    if not 0.0 < RADIUS_LEVEL_TAIL < 1.0:
        raise ValueError("RADIUS_LEVEL_TAIL must lie strictly between 0 and 1.")
    if RADIUS_LEVEL_GROWTH <= 1.0:
        raise ValueError("RADIUS_LEVEL_GROWTH must be greater than 1.")
    if RADIUS_LEVEL_TAIL * RADIUS_LEVEL_GROWTH**2 >= 1.0:
        raise ValueError(
            "The radius law must satisfy "
            "RADIUS_LEVEL_TAIL * RADIUS_LEVEL_GROWTH**2 < 1."
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


def winding_guide(times: np.ndarray) -> np.ndarray:
    """A slow reference curve used only to choose among admissible paths."""

    normalized_time = (times - times[0]) / (times[-1] - times[0])
    radius = WINDING_GUIDE_RADIUS * (
        0.30 + 0.70 * (1.0 - np.exp(-3.0 * normalized_time))
    )
    angle = 2.0 * np.pi * WINDING_GUIDE_TURNS * normalized_time
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def find_admissible_crossing(
    times: np.ndarray,
    grid_points: np.ndarray,
    clearances: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
):
    """Find a winding bounded-speed path through the vacant set.

    Validity is imposed first.  Among valid transitions, dynamic programming
    selects a path close to ``winding_guide`` while penalizing low clearance.
    The guide cannot authorize a collision or a violation of the speed bound.
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
    entrance_mask = (
        np.linalg.norm(grid_points, axis=2) <= ENTRANCE_RADIUS
    ) & (clearances[0] >= PATH_CLEARANCE_FLOOR)

    guide = winding_guide(times)
    initial_guide_distance = np.sum(
        (grid_points - guide[0][None, None, :]) ** 2, axis=2
    )

    costs = np.full((grid_size, grid_size), np.inf, dtype=np.float64)
    costs[entrance_mask] = initial_guide_distance[entrance_mask]
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

            guide_distance = np.sum(
                (
                    current_points
                    - guide[time_index][None, :]
                )
                ** 2,
                axis=1,
            ).reshape(previous_costs.shape)
            clearance_deficit = np.maximum(
                0.0,
                PREFERRED_PATH_CLEARANCE - local_clearance,
            ) / PREFERRED_PATH_CLEARANCE
            candidate_costs = (
                previous_costs
                + GUIDE_TRACKING_WEIGHT * guide_distance
                + MOVEMENT_SHORTFALL_COST_WEIGHT
                * (maximum_spatial_step - spatial_distance) ** 2
                + LOW_CLEARANCE_COST_WEIGHT * clearance_deficit**2
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
            return None, None

    endpoint_flat_index = int(np.argmin(costs))
    bottleneck_clearance = float(
        bottlenecks.flat[endpoint_flat_index]
    )
    if not np.isfinite(costs.flat[endpoint_flat_index]):
        return None, None

    flat_grid_points = grid_points.reshape(-1, 2)
    path = np.empty((len(times), 2), dtype=float)
    path[-1] = flat_grid_points[endpoint_flat_index]
    current_flat_index = endpoint_flat_index
    for time_index in range(len(times) - 1, 0, -1):
        current_flat_index = int(
            back_pointers[time_index].flat[current_flat_index]
        )
        if current_flat_index < 0:
            raise RuntimeError("Broken back-pointer in admissible path.")
        path[time_index - 1] = flat_grid_points[current_flat_index]

    return path, bottleneck_clearance


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
    path: np.ndarray | None,
):
    intersects_window = np.any(
        np.max(np.abs(centers), axis=2)
        <= SPACE_HALF_WIDTH + radii[None, :],
        axis=0,
    )
    candidates = np.flatnonzero(intersects_window)
    if len(candidates) <= MAX_DISPLAY_TUBES:
        return candidates

    if path is None:
        distance_to_focus = np.min(
            np.linalg.norm(centers[:, candidates, :], axis=2)
            - radii[candidates][None, :],
            axis=0,
        )
    else:
        distance_to_focus = np.min(
            np.linalg.norm(
                centers[:, candidates, :] - path[:, None, :], axis=2
            )
            - radii[candidates][None, :],
            axis=0,
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
    path: np.ndarray,
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

    axis.plot(
        path[:, 0],
        path[:, 1],
        times,
        color=ESCAPE_COLOR,
        linewidth=3.2,
        zorder=20,
    )
    axis.scatter(
        [path[0, 0]],
        [path[0, 1]],
        [times[0]],
        color=ESCAPE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=48,
        depthshade=False,
        zorder=21,
    )
    axis.scatter(
        [path[-1, 0]],
        [path[-1, 1]],
        [times[-1]],
        color=ESCAPE_COLOR,
        edgecolor="white",
        linewidth=0.8,
        s=48,
        depthshade=False,
        zorder=21,
    )
    draw_space_time_box(axis)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def build_interactive_figure(
    times: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    path: np.ndarray | None,
    display_particles: np.ndarray,
    interval_clearances: np.ndarray | None,
):
    figure = go.Figure()
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

    if path is not None and interval_clearances is not None:
        path_clearance = np.r_[interval_clearances[0], interval_clearances]
        custom_data = np.column_stack((times, path_clearance))
        figure.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=times,
                mode="lines",
                line=dict(color=ESCAPE_COLOR, width=7),
                name="v-admissible escape path",
                customdata=custom_data,
                hovertemplate=(
                    "escape path<br>t = %{customdata[0]:.3f}<br>"
                    "interval clearance = %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[path[0, 0], path[-1, 0]],
                y=[path[0, 1], path[-1, 1]],
                z=[times[0], times[-1]],
                mode="markers",
                marker=dict(size=6, color=[ESCAPE_COLOR, ESCAPE_COLOR]),
                text=["entrance", "exit"],
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
    path: np.ndarray | None,
    display_particles: np.ndarray,
    interval_clearances: np.ndarray | None,
):
    figure = build_interactive_figure(
        times,
        centers,
        radii,
        path,
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
    path: np.ndarray,
    display_particles: np.ndarray,
    interval_clearances: np.ndarray,
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
    style_axis(axis)

    escape_line, = axis.plot([], [], [], color=ESCAPE_COLOR, linewidth=3.2)
    current_target = axis.scatter(
        [], [], [], color=ESCAPE_COLOR, edgecolor="white", s=52, depthshade=False
    )
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
            escape_line.set_data_3d([], [], [])
            current_target._offsets3d = ([], [], [])
            status_text.set_text(
                "environment complete\n"
                "the full future is known before escape"
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
            clearance = (
                interval_clearances[max(0, time_index - 1)]
                if len(interval_clearances)
                else np.inf
            )
            spatial_speed = (
                np.linalg.norm(path[time_index] - path[time_index - 1])
                / (times[time_index] - times[time_index - 1])
                if time_index > 0
                else 0.0
            )
            status_text.set_text(
                "phase 2: escape with the full future known\n"
                f"escape time: {current_time:5.2f} / {TOTAL_TIME:.2f}\n"
                f"spatial speed: {spatial_speed:.3f} / "
                f"{MAX_TARGET_SPEED:.3f}\n"
                f"target clearance: {clearance:.3f}"
            )

        axis.view_init(elev=23, azim=-55 + 0.10 * frame_number)
        return [
            escape_line,
            current_target,
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
    environment_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    initial_positions, radii, centers, poisson_half_width = (
        simulate_environment(environment_rng, times)
    )
    clearances = endpoint_clearances(grid_points, centers, radii)
    path, bottleneck_clearance = find_admissible_crossing(
        times, grid_points, clearances, centers, radii
    )

    if path is None:
        raise RuntimeError(
            "This fixed random environment has no admissible crossing on the "
            "chosen finite grid. It was not resampled. Run the script again "
            "for a new independent environment, or change the model/grid "
            "parameters explicitly."
        )
    (
        maximum_speed,
        minimum_clearance,
        speeds,
        interval_clearances,
    ) = verify_path(path, times, centers, radii)
    spatial_path_length = float(
        np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    )
    unwrapped_angles = np.unwrap(np.arctan2(path[:, 1], path[:, 0]))
    signed_spatial_turns = float(
        (unwrapped_angles[-1] - unwrapped_angles[0]) / (2.0 * np.pi)
    )
    continuous_speed_budget = MAX_TARGET_SPEED * (times[-1] - times[0])
    speed_budget_fraction = spatial_path_length / continuous_speed_budget

    display_particles = choose_display_particles(centers, radii, path)
    output_directory = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"escapist_space_time_{run_id}"
    image_path = output_directory / f"{base_name}.png"
    interactive_path = output_directory / f"{base_name}_interactive.html"
    video_path = output_directory / f"{base_name}.mp4"

    print(f"run seed: {run_seed}")
    print("random environments generated: 1")
    print(f"Poisson particles in truncated window: {len(radii)}")
    radius_levels = np.rint(
        np.log(radii / RADIUS_BASE) / np.log(RADIUS_LEVEL_GROWTH)
    ).astype(int)
    realized_levels, realized_counts = np.unique(
        radius_levels, return_counts=True
    )
    expected_radius_squared = (
        RADIUS_BASE**2
        * (1.0 - RADIUS_LEVEL_TAIL)
        / (1.0 - RADIUS_LEVEL_TAIL * RADIUS_LEVEL_GROWTH**2)
    )
    print(
        "radius law: P(K=k)=(1-q)q^k, "
        f"R_k={RADIUS_BASE:.3f}*{RADIUS_LEVEL_GROWTH:.3f}^k, "
        f"q={RADIUS_LEVEL_TAIL:.3f}"
    )
    print(f"theoretical E[R^2]: {expected_radius_squared:.6f}")
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
    print(f"maximum target speed: {maximum_speed:.6f} <= {MAX_TARGET_SPEED:.6f}")
    print(f"minimum tube clearance: {minimum_clearance:.6f}")
    print(f"path bottleneck score: {bottleneck_clearance:.6f}")
    print(f"spatial path length: {spatial_path_length:.6f}")
    print(f"continuous speed budget used: {100.0 * speed_budget_fraction:.2f}%")
    print(f"signed spatial turns: {signed_spatial_turns:.6f}")
    print(f"initial Poisson half-width: {poisson_half_width:.3f}")

    save_static_figure(
        image_path, times, centers, radii, path, display_particles
    )
    print(f"image saved to: {image_path}")

    save_interactive_figure(
        interactive_path,
        times,
        centers,
        radii,
        path,
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
            path,
            display_particles,
            interval_clearances,
        )
        print(f"video saved to: {video_path}")


if __name__ == "__main__":
    main()
import colorsys
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# With seed=None, every execution generates new regions, new starting points,
# and new walks.  Replace None by an integer to reproduce a particular run.
seed = None
seed_sequence = np.random.SeedSequence(seed)
geometry_seed, regular_seed, trap_seed, color_seed = seed_sequence.spawn(4)
rng_geometry = np.random.default_rng(geometry_seed)
rng_regular = np.random.default_rng(regular_seed)
rng_trap = np.random.default_rng(trap_seed)
rng_colors = np.random.default_rng(color_seed)
print(f"run seed: {seed_sequence.entropy}")


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
R0 = rng_geometry.uniform(0.96, 1.04)
modes = np.array([2, 3, 5, 7])
amps = np.array([0.12, 0.08, 0.05, 0.035]) * rng_geometry.uniform(
    0.72, 1.25, size=4
)
phases = rng_geometry.uniform(0.0, 2.0 * np.pi, size=4)


def r_boundary(theta):
    """Radial function of the regular region."""
    theta = np.asarray(theta)
    radius = R0 * np.ones_like(theta, dtype=float)
    for amplitude, mode, phase in zip(amps, modes, phases):
        radius += amplitude * np.cos(mode * theta + phase)
    return radius


def phi_regular(xy):
    """Negative inside the regular region and positive outside it."""
    xy = np.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]
    theta = np.arctan2(y, x)
    radius = np.hypot(x, y)
    return radius - r_boundary(theta)


def phi_capsule(xy, start, end, radius):
    """Signed distance to a capsule joining two points."""
    xy = np.asarray(xy)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    parameter = np.sum((xy - start) * segment, axis=-1) / np.dot(segment, segment)
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = start + parameter[..., None] * segment
    return np.linalg.norm(xy - closest, axis=-1) - radius


# The corridor is regenerated at every run while remaining attached to the
# right-hand side of the main region.
right_radius = float(r_boundary(0.0))
corridor_start = np.array(
    [right_radius - rng_geometry.uniform(0.14, 0.19), rng_geometry.uniform(-0.07, 0.07)]
)
corridor_length = rng_geometry.uniform(0.56, 0.70)
corridor_bend = rng_geometry.uniform(-0.085, 0.085)
corridor_nodes = np.array(
    [
        corridor_start,
        corridor_start
        + np.array([0.32 * corridor_length, 0.45 * corridor_bend]),
        corridor_start
        + np.array([0.68 * corridor_length, corridor_bend]),
        corridor_start
        + np.array([corridor_length, 0.70 * corridor_bend]),
    ]
)
corridor_radius = rng_geometry.uniform(0.058, 0.075)

chamber_base_radius = rng_geometry.uniform(0.48, 0.56)
chamber_modes = np.array([2, 3, 5])
chamber_amps = np.array([0.043, 0.028, 0.017]) * rng_geometry.uniform(
    0.70, 1.25, size=3
)
chamber_phases = rng_geometry.uniform(0.0, 2.0 * np.pi, size=3)
chamber_center = corridor_nodes[-1] + np.array(
    [rng_geometry.uniform(0.27, 0.33), rng_geometry.uniform(-0.035, 0.035)]
)


def r_chamber(theta):
    """A new mildly irregular trapping chamber at every execution."""
    theta = np.asarray(theta)
    radius = chamber_base_radius * np.ones_like(theta, dtype=float)
    for amplitude, mode, phase in zip(
        chamber_amps, chamber_modes, chamber_phases
    ):
        radius += amplitude * np.cos(mode * theta + phase)
    return radius


def phi_chamber(xy):
    """Negative inside the irregular trapping chamber."""
    xy = np.asarray(xy)
    displacement = xy - chamber_center
    theta = np.arctan2(displacement[..., 1], displacement[..., 0])
    radius = np.linalg.norm(displacement, axis=-1)
    return radius - r_chamber(theta)


def smooth_union(*level_sets, sharpness=34.0):
    """Smooth approximation of the minimum of several level sets."""
    result = level_sets[0]
    for value in level_sets[1:]:
        result = -np.logaddexp(-sharpness * result, -sharpness * value) / sharpness
    return result


def phi_trap(xy):
    """Regular region, curved narrow corridor, and trapping chamber."""
    main_region = phi_regular(xy)
    corridor_pieces = [
        phi_capsule(xy, start, end, corridor_radius)
        for start, end in zip(corridor_nodes[:-1], corridor_nodes[1:])
    ]

    return smooth_union(main_region, *corridor_pieces, phi_chamber(xy))


# -----------------------------------------------------------------------------
# Slow-membrane random walk
# -----------------------------------------------------------------------------
N = 1_000
step = 1.0 / N
K = 25
n_jumps = K * N**2
alpha = 1.0

# Simulation and plotting controls.  The chain still performs all KN^2 steps;
# only the stored curve is thinned before plotting.
simulation_block_size = 20_000
maximum_plotted_points = 350_000
plot_stride = max(1, int(np.ceil(n_jumps / maximum_plotted_points)))

# Video controls.  The full KN^2-step walk is simulated first; these values
# only control how densely that exact realization is rendered in the movie.
video_duration_seconds = 15
video_frames_per_second = 24
video_frame_count = video_duration_seconds * video_frames_per_second
maximum_video_points = 60_000

lattice_directions = np.array(
    [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ],
    dtype=np.int32,
)

def nearest_inside_lattice_point(phi, point, interior_point):
    """Round a starting point to the lattice while keeping it inside."""
    point = np.asarray(point, dtype=float)
    interior_point = np.asarray(interior_point, dtype=float)

    for _ in range(20):
        site = np.rint(N * point).astype(np.int32)
        if phi(site / N) <= 0:
            return site
        point = 0.9 * point + 0.1 * interior_point

    raise RuntimeError("Could not find an interior lattice starting point.")


def simulate_slow_membrane_walk(phi, start, n_jumps, rng):
    """Simulate the discrete-time chain from the draft.

    Each coordinate direction is proposed with probability 1/4.  A proposed
    crossing edge is accepted with probability alpha/N; if it is rejected,
    the chain stays at its current site for that discrete time step.
    """
    position = np.asarray(start, dtype=np.int64).copy()
    current_inside = bool(phi(position / N) <= 0)

    sampled_positions = [position.astype(float) / N]
    sampled_steps = [0]
    crossing_points = []
    crossing_steps = []
    holding_steps = []
    elapsed_steps = 0
    crossing_acceptance = min(alpha / N, 1.0)

    def store_samples(block_positions, block_start_step):
        """Store stride-aligned positions from an accepted block prefix."""
        block_length = len(block_positions)
        if block_length == 0:
            return

        first_offset = plot_stride - (block_start_step % plot_stride)
        offsets = np.arange(first_offset, block_length + 1, plot_stride)
        if len(offsets) == 0:
            return

        sampled_positions.extend(block_positions[offsets - 1] / N)
        sampled_steps.extend(block_start_step + offsets)

    while elapsed_steps < n_jumps:
        block_length = min(simulation_block_size, n_jumps - elapsed_steps)
        proposed_directions = rng.integers(0, 4, size=block_length)
        increments = lattice_directions[proposed_directions]
        proposed_positions = position + np.cumsum(increments, axis=0)

        proposed_inside = np.asarray(phi(proposed_positions / N) <= 0)
        previous_inside = np.empty(block_length, dtype=bool)
        previous_inside[0] = current_inside
        previous_inside[1:] = proposed_inside[:-1]
        crossing_attempts = previous_inside != proposed_inside

        if not np.any(crossing_attempts):
            store_samples(proposed_positions, elapsed_steps)
            position = proposed_positions[-1]
            current_inside = bool(proposed_inside[-1])
            elapsed_steps += block_length
            continue

        # Accept the homogeneous prefix preceding the first slow edge.
        crossing_offset = int(np.flatnonzero(crossing_attempts)[0])
        prefix = proposed_positions[:crossing_offset]
        store_samples(prefix, elapsed_steps)

        if crossing_offset > 0:
            position = prefix[-1]
            current_inside = bool(proposed_inside[crossing_offset - 1])
            elapsed_steps += crossing_offset

        # One discrete time step is now spent attempting the slow edge.
        old_position = position.copy()
        if rng.random() < crossing_acceptance:
            position = proposed_positions[crossing_offset]
            current_inside = bool(proposed_inside[crossing_offset])
            crossing_points.append((old_position + position) / (2.0 * N))
            crossing_steps.append(elapsed_steps + 1)
        else:
            # The slow edge was proposed but rejected: this discrete-time
            # step is spent holding at the site adjacent to the membrane.
            holding_steps.append(elapsed_steps + 1)

        elapsed_steps += 1
        if elapsed_steps % plot_stride == 0:
            sampled_positions.append(position.astype(float) / N)
            sampled_steps.append(elapsed_steps)

    if sampled_steps[-1] != n_jumps:
        sampled_positions.append(position.astype(float) / N)
        sampled_steps.append(n_jumps)

    sampled_positions = np.asarray(sampled_positions)
    sampled_steps = np.asarray(sampled_steps, dtype=np.int64)
    crossing_steps = np.asarray(crossing_steps, dtype=np.int64)
    crossing_indices = np.searchsorted(sampled_steps, crossing_steps, side="left")
    crossing_indices = np.clip(crossing_indices, 0, len(sampled_positions) - 1)

    return (
        sampled_positions,
        np.asarray(crossing_points),
        crossing_indices,
        np.asarray(holding_steps, dtype=np.int64),
    )


# The first starting point is chosen randomly near the regular boundary.
theta_start = rng_geometry.uniform(0.0, 2.0 * np.pi)
regular_radius = r_boundary(theta_start) - 1.5 / N
regular_start_point = regular_radius * np.array(
    [np.cos(theta_start), np.sin(theta_start)]
)
regular_start = nearest_inside_lattice_point(
    phi_regular,
    regular_start_point,
    interior_point=[0.0, 0.0],
)

# The second starting point varies along the middle part of the narrow corridor.
start_parameter = rng_geometry.uniform(0.15, 0.85)
middle_start = corridor_nodes[1]
middle_end = corridor_nodes[2]
corridor_direction = middle_end - middle_start
corridor_direction /= np.linalg.norm(corridor_direction)
corridor_normal = np.array([-corridor_direction[1], corridor_direction[0]])
corridor_centerline_point = (
    (1.0 - start_parameter) * middle_start + start_parameter * middle_end
)
chamber_start_point = corridor_centerline_point + corridor_normal * rng_geometry.uniform(
    -0.25 * corridor_radius, 0.25 * corridor_radius
)
trap_start = nearest_inside_lattice_point(
    phi_trap,
    chamber_start_point,
    interior_point=corridor_centerline_point,
)

(
    regular_walk,
    regular_crossings,
    regular_crossing_indices,
    regular_holding_steps,
) = simulate_slow_membrane_walk(
    phi_regular,
    start=regular_start,
    n_jumps=n_jumps,
    rng=rng_regular,
)

(
    trapped_walk,
    trapped_crossings,
    trapped_crossing_indices,
    trapped_holding_steps,
) = simulate_slow_membrane_walk(
    phi_trap,
    start=trap_start,
    n_jumps=n_jumps,
    rng=rng_trap,
)

print(f"regular geometry: {len(regular_crossings)} crossings")
print(f"narrow-corridor geometry: {len(trapped_crossings)} crossings")
print(f"regular geometry: {len(regular_holding_steps)} membrane holding steps")
print(
    "narrow-corridor geometry: "
    f"{len(trapped_holding_steps)} membrane holding steps"
)


# -----------------------------------------------------------------------------
# One clean image containing both simulations
# -----------------------------------------------------------------------------
regular_extent = R0 + np.sum(np.abs(amps)) + 0.12
chamber_extent = chamber_base_radius + np.sum(np.abs(chamber_amps)) + 0.12
trap_x_max = chamber_center[0] + chamber_extent
vertical_extent = max(
    regular_extent,
    abs(chamber_center[1]) + chamber_extent,
)


def expanded_bounds(base_low, base_high, values, maximum_factor=1.75):
    """Zoom out moderately when a trajectory leaves the geometric window."""
    values = np.asarray(values)
    robust_low, robust_high = np.quantile(values, [0.002, 0.998])
    desired_low = min(base_low, robust_low)
    desired_high = max(base_high, robust_high)

    base_center = 0.5 * (base_low + base_high)
    base_width = base_high - base_low
    maximum_width = maximum_factor * base_width

    if desired_high - desired_low > maximum_width:
        desired_low = base_center - 0.5 * maximum_width
        desired_high = base_center + 0.5 * maximum_width

    padding = 0.025 * (desired_high - desired_low)
    return desired_low - padding, desired_high + padding


regular_x_limits = expanded_bounds(
    -regular_extent,
    regular_extent,
    regular_walk[:, 0],
)
trap_x_limits = expanded_bounds(
    -regular_extent,
    trap_x_max,
    trapped_walk[:, 0],
)
common_y_limits = expanded_bounds(
    -vertical_extent,
    vertical_extent,
    np.concatenate([regular_walk[:, 1], trapped_walk[:, 1]]),
)

x_regular = np.linspace(-regular_extent, regular_extent, 620)
y_regular = np.linspace(-vertical_extent, vertical_extent, 620)
X_regular, Y_regular = np.meshgrid(x_regular, y_regular)
Z_regular = phi_regular(np.stack([X_regular, Y_regular], axis=-1))

x_trap = np.linspace(-regular_extent, trap_x_max, 850)
y_trap = np.linspace(-vertical_extent, vertical_extent, 620)
X_trap, Y_trap = np.meshgrid(x_trap, y_trap)
Z_trap = phi_trap(np.stack([X_trap, Y_trap], axis=-1))

regular_panel_width = regular_x_limits[1] - regular_x_limits[0]
trap_panel_width = trap_x_limits[1] - trap_x_limits[0]
fig, (ax_regular, ax_trap) = plt.subplots(
    1,
    2,
    figsize=(12, 4.7),
    sharey=True,
    gridspec_kw={
        "width_ratios": [regular_panel_width, trap_panel_width],
        "wspace": 0.035,
    },
)

boundary_color = "#111111"

# The entire pastel palette is generated anew at every execution.  Successive
# hues are separated by the golden angle, so nearby trajectory segments have
# visibly different colors without a short periodic cycle.
initial_hue = rng_colors.uniform(0.0, 1.0)
golden_angle = 0.618033988749895
start_marker_color = "#4F93D2"


def walk_color(index):
    """Generate the index-th pastel color by a golden-angle rotation."""
    hue = (initial_hue + index * golden_angle) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.76, 0.56)
    return "#{:02X}{:02X}{:02X}".format(
        round(255 * red),
        round(255 * green),
        round(255 * blue),
    )


def plot_walk_by_crossings(ax, trajectory, crossing_indices):
    """Change the trajectory color after every membrane crossing."""
    endpoints = np.concatenate(
        ([0], crossing_indices, [len(trajectory) - 1])
    )

    for segment_index in range(len(endpoints) - 1):
        start = int(endpoints[segment_index])
        stop = int(endpoints[segment_index + 1])
        color = walk_color(segment_index)

        ax.plot(
            trajectory[start : stop + 1, 0],
            trajectory[start : stop + 1, 1],
            color=color,
            linewidth=0.32,
            alpha=0.72,
            zorder=1,
            clip_on=True,
        )


plot_walk_by_crossings(
    ax_regular,
    regular_walk,
    regular_crossing_indices,
)

plot_walk_by_crossings(
    ax_trap,
    trapped_walk,
    trapped_crossing_indices,
)

# Small markers indicate the starting sites without adding a legend.
ax_regular.scatter(
    regular_walk[0, 0],
    regular_walk[0, 1],
    s=24,
    color=start_marker_color,
    linewidths=0,
    zorder=4,
    clip_on=True,
)

ax_trap.scatter(
    trapped_walk[0, 0],
    trapped_walk[0, 1],
    s=24,
    color=start_marker_color,
    linewidths=0,
    zorder=4,
    clip_on=True,
)

# Each crossing point has the color of the trajectory segment that starts
# immediately after that crossing.
if len(regular_crossings):
    regular_crossing_colors = [
        walk_color(index + 1)
        for index in range(len(regular_crossings))
    ]
    ax_regular.scatter(
        regular_crossings[:, 0],
        regular_crossings[:, 1],
        s=17,
        color=regular_crossing_colors,
        linewidths=0,
        zorder=4,
        clip_on=True,
    )

if len(trapped_crossings):
    trapped_crossing_colors = [
        walk_color(index + 1)
        for index in range(len(trapped_crossings))
    ]
    ax_trap.scatter(
        trapped_crossings[:, 0],
        trapped_crossings[:, 1],
        s=17,
        color=trapped_crossing_colors,
        linewidths=0,
        zorder=4,
        clip_on=True,
    )

ax_regular.contour(
    X_regular,
    Y_regular,
    Z_regular,
    levels=[0.0],
    colors=[boundary_color],
    linewidths=1.8,
    zorder=2,
)

ax_trap.contour(
    X_trap,
    Y_trap,
    Z_trap,
    levels=[0.0],
    colors=[boundary_color],
    linewidths=1.8,
    zorder=2,
)

ax_regular.set_xlim(*regular_x_limits)
ax_trap.set_xlim(*trap_x_limits)

for panel in (ax_regular, ax_trap):
    panel.set_ylim(*common_y_limits)
    panel.set_aspect("equal", adjustable="box")
    panel.axis("off")

fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.035)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_path = Path(__file__).with_name(f"snob_two_regions_{run_id}.png")
fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor="white",
    transparent=False,
)
print(f"image saved to: {output_path}")
plt.close(fig)


# -----------------------------------------------------------------------------
# Animation of the same two realizations
# -----------------------------------------------------------------------------
def decimate_for_video(trajectory, crossing_indices):
    """Thin a stored trajectory while retaining time and color changes."""
    video_stride = max(1, int(np.ceil(len(trajectory) / maximum_video_points)))
    retained_indices = np.arange(0, len(trajectory), video_stride, dtype=np.int64)
    retained_indices = np.unique(
        np.concatenate(
            [retained_indices, crossing_indices, [len(trajectory) - 1]]
        )
    )
    video_trajectory = trajectory[retained_indices]
    video_crossing_indices = np.searchsorted(retained_indices, crossing_indices)
    video_times = retained_indices / (len(trajectory) - 1)
    return video_trajectory, video_crossing_indices, video_times


(
    regular_video_walk,
    regular_video_crossing_indices,
    regular_video_times,
) = decimate_for_video(
    regular_walk,
    regular_crossing_indices,
)
(
    trapped_video_walk,
    trapped_video_crossing_indices,
    trapped_video_times,
) = decimate_for_video(
    trapped_walk,
    trapped_crossing_indices,
)

video_fig, (video_ax_regular, video_ax_trap) = plt.subplots(
    1,
    2,
    figsize=(12, 5.35),
    sharey=True,
    gridspec_kw={
        "width_ratios": [regular_panel_width, trap_panel_width],
        "wspace": 0.035,
    },
)

video_ax_regular.contour(
    X_regular,
    Y_regular,
    Z_regular,
    levels=[0.0],
    colors=[boundary_color],
    linewidths=1.8,
    zorder=2,
)
video_ax_trap.contour(
    X_trap,
    Y_trap,
    Z_trap,
    levels=[0.0],
    colors=[boundary_color],
    linewidths=1.8,
    zorder=2,
)

video_ax_regular.set_xlim(*regular_x_limits)
video_ax_trap.set_xlim(*trap_x_limits)
for panel in (video_ax_regular, video_ax_trap):
    panel.set_ylim(*common_y_limits)
    panel.set_aspect("equal", adjustable="box")
    panel.axis("off")

video_fig.subplots_adjust(
    left=0.005,
    right=0.995,
    bottom=0.15,
    top=0.995,
    wspace=0.035,
)

regular_panel_position = video_ax_regular.get_position()
trap_panel_position = video_ax_trap.get_position()
regular_panel_center = 0.5 * (
    regular_panel_position.x0 + regular_panel_position.x1
)
trap_panel_center = 0.5 * (trap_panel_position.x0 + trap_panel_position.x1)

counter_style = {
    "ha": "center",
    "va": "center",
    "fontsize": 10.5,
    "family": "monospace",
    "color": "#303030",
}
regular_counter = video_fig.text(
    regular_panel_center,
    0.062,
    "",
    **counter_style,
)
trap_counter = video_fig.text(
    trap_panel_center,
    0.062,
    "",
    **counter_style,
)


def make_animated_walk(
    ax,
    trajectory,
    crossing_indices,
    crossing_points,
    trajectory_times,
):
    """Create artists for a trajectory whose color changes at crossings."""
    segment_endpoints = np.concatenate(
        ([0], crossing_indices, [len(trajectory) - 1])
    ).astype(np.int64)
    line_artists = []
    for segment_index in range(len(segment_endpoints) - 1):
        line, = ax.plot(
            [],
            [],
            color=walk_color(segment_index),
            linewidth=0.42,
            alpha=0.76,
            zorder=1,
            clip_on=True,
        )
        line_artists.append(line)

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=24,
        color=start_marker_color,
        linewidths=0,
        zorder=4,
        clip_on=True,
    )

    crossing_artist = ax.scatter(
        [],
        [],
        s=17,
        linewidths=0,
        zorder=4,
        clip_on=True,
    )
    crossing_colors = [
        walk_color(index + 1) for index in range(len(crossing_points))
    ]
    current_position_artist = ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=31,
        color=walk_color(0),
        edgecolors="white",
        linewidths=0.75,
        zorder=5,
        clip_on=True,
    )

    def update(progress):
        current_index = min(
            len(trajectory) - 1,
            int(np.searchsorted(trajectory_times, progress, side="right") - 1),
        )

        for segment_index, line in enumerate(line_artists):
            segment_start = int(segment_endpoints[segment_index])
            segment_stop = min(
                current_index,
                int(segment_endpoints[segment_index + 1]),
            )
            if segment_stop < segment_start:
                line.set_data([], [])
            else:
                line.set_data(
                    trajectory[segment_start : segment_stop + 1, 0],
                    trajectory[segment_start : segment_stop + 1, 1],
                )

        visible_crossings = int(np.searchsorted(
            crossing_indices,
            current_index,
            side="right",
        ))
        if visible_crossings:
            crossing_artist.set_offsets(crossing_points[:visible_crossings])
            crossing_artist.set_facecolors(crossing_colors[:visible_crossings])
        else:
            crossing_artist.set_offsets(np.empty((0, 2)))

        current_segment = int(np.searchsorted(
            crossing_indices,
            current_index,
            side="right",
        ))
        current_position_artist.set_offsets(trajectory[current_index])
        current_position_artist.set_facecolors([walk_color(current_segment)])

        return [*line_artists, crossing_artist, current_position_artist]

    return update


update_regular_video = make_animated_walk(
    video_ax_regular,
    regular_video_walk,
    regular_video_crossing_indices,
    regular_crossings,
    regular_video_times,
)
update_trapped_video = make_animated_walk(
    video_ax_trap,
    trapped_video_walk,
    trapped_video_crossing_indices,
    trapped_crossings,
    trapped_video_times,
)


def update_video(frame_number):
    progress = frame_number / (video_frame_count - 1)
    elapsed_steps = min(n_jumps, int(round(progress * n_jumps)))
    regular_holding_count = int(np.searchsorted(
        regular_holding_steps,
        elapsed_steps,
        side="right",
    ))
    trapped_holding_count = int(np.searchsorted(
        trapped_holding_steps,
        elapsed_steps,
        side="right",
    ))

    formatted_steps = f"{elapsed_steps:,}".replace(",", ".")
    formatted_total = f"{n_jumps:,}".replace(",", ".")
    regular_counter.set_text(
        f"passos: {formatted_steps} / {formatted_total}\n"
        "retidos na membrana: "
        f"{regular_holding_count:,}".replace(",", ".")
    )
    trap_counter.set_text(
        f"passos: {formatted_steps} / {formatted_total}\n"
        "retidos na membrana: "
        f"{trapped_holding_count:,}".replace(",", ".")
    )

    return [
        *update_regular_video(progress),
        *update_trapped_video(progress),
        regular_counter,
        trap_counter,
    ]


movie = animation.FuncAnimation(
    video_fig,
    update_video,
    frames=video_frame_count,
    interval=1_000 / video_frames_per_second,
    blit=False,
)
video_path = Path(__file__).with_name(f"snob_two_regions_{run_id}.mp4")
video_writer = animation.FFMpegWriter(
    fps=video_frames_per_second,
    codec="libx264",
    bitrate=2_800,
    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
)
movie.save(video_path, writer=video_writer, dpi=160)
plt.close(video_fig)
print(f"video saved to: {video_path}")
import colorsys
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
        "Plotly is required for the interactive 3D figure. "
        "Install it with: python -m pip install plotly"
    ) from error


# With seed=None, the cluster, starting point, pastel colors, and walk are new
# at every execution. Set an integer here to reproduce one realization.
seed = None
seed_sequence = np.random.SeedSequence(seed)
geometry_seed, walk_seed, color_seed = seed_sequence.spawn(3)
rng_geometry = np.random.default_rng(geometry_seed)
rng_walk = np.random.default_rng(walk_seed)
rng_colors = np.random.default_rng(color_seed)
print(f"run seed: {seed_sequence.entropy}")


# -----------------------------------------------------------------------------
# Random cluster of disjoint, nearby spheres with different radii
# -----------------------------------------------------------------------------
def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


# K has a truncated geometric distribution on {4, ..., 15}:
#
#                 P(K = k) proportional to 0.60 ** (k - 4).
#
# The weights decay quickly, so small clusters dominate while every value up
# to 15 remains possible.  The corresponding untruncated law has finite mean.
minimum_sphere_count = 4
maximum_sphere_count = 15
sphere_count_decay = 0.60
possible_sphere_counts = np.arange(
    minimum_sphere_count,
    maximum_sphere_count + 1,
)
sphere_count_weights = sphere_count_decay ** (
    possible_sphere_counts - minimum_sphere_count
)
sphere_count_probabilities = sphere_count_weights / np.sum(
    sphere_count_weights
)
sphere_count = int(
    rng_geometry.choice(
        possible_sphere_counts,
        p=sphere_count_probabilities,
    )
)

central_radius = rng_geometry.uniform(0.92, 1.12)
sphere_centers = [np.zeros(3)]
sphere_radii = [central_radius]
sphere_depths = [0]

# Each new sphere is attached near an existing one.  Parents near the origin
# receive more weight, which keeps the cluster compact, while occasional outer
# parents create a genuinely random three-dimensional environment.
minimum_surface_separation = 0.055
maximum_cluster_center_distance = 3.4 + 0.15 * sphere_count

for _ in range(1, sphere_count):
    radius = rng_geometry.uniform(0.50, 1.12)
    placed = False

    for attempt in range(1_600):
        existing_centers = np.asarray(sphere_centers)
        existing_radii = np.asarray(sphere_radii)
        center_distances = np.linalg.norm(existing_centers, axis=1)
        parent_weights = np.exp(-0.62 * center_distances)
        parent_weights[0] *= 2.2
        parent_weights /= np.sum(parent_weights)
        parent_index = int(
            rng_geometry.choice(len(sphere_centers), p=parent_weights)
        )

        direction = normalized(rng_geometry.normal(size=3))
        attachment_gap = rng_geometry.uniform(0.075, 0.24)
        candidate_center = (
            existing_centers[parent_index]
            + (
                existing_radii[parent_index]
                + radius
                + attachment_gap
            )
            * direction
        )

        if (
            np.linalg.norm(candidate_center)
            > maximum_cluster_center_distance
        ):
            continue

        surface_separations = (
            np.linalg.norm(existing_centers - candidate_center, axis=1)
            - existing_radii
            - radius
        )
        if np.min(surface_separations) < minimum_surface_separation:
            if attempt in {500, 1_000}:
                radius = max(0.44, 0.90 * radius)
            continue

        sphere_centers.append(candidate_center)
        sphere_radii.append(radius)
        sphere_depths.append(sphere_depths[parent_index] + 1)
        placed = True
        break

    if not placed:
        raise RuntimeError("Could not place the random sphere cluster.")

sphere_centers = np.asarray(sphere_centers, dtype=float)
sphere_radii = np.asarray(sphere_radii, dtype=float)

for first_index in range(sphere_count):
    for second_index in range(first_index + 1, sphere_count):
        center_distance = np.linalg.norm(
            sphere_centers[first_index] - sphere_centers[second_index]
        )
        if center_distance <= (
            sphere_radii[first_index]
            + sphere_radii[second_index]
            + 0.99 * minimum_surface_separation
        ):
            raise RuntimeError("The generated spheres unexpectedly overlap.")


def phi_sphere_cluster(xyz):
    """Negative inside any sphere and positive outside all spheres."""
    xyz = np.asarray(xyz, dtype=float)
    displacements = xyz[..., None, :] - sphere_centers
    distances = np.linalg.norm(displacements, axis=-1)
    return np.min(distances - sphere_radii, axis=-1)


# -----------------------------------------------------------------------------
# Slow-membrane random walk on the three-dimensional lattice
# -----------------------------------------------------------------------------
N = 1_000
K = 25
n_steps = K * N**2
alpha = 1.0

simulation_block_size = 20_000
maximum_plotted_points = 180_000
plot_stride = max(1, int(np.ceil(n_steps / maximum_plotted_points)))

video_duration_seconds = 15
video_frames_per_second = 24
video_frame_count = video_duration_seconds * video_frames_per_second
maximum_video_points = 32_000

# Long excursions are clipped after this amount of zoom-out, keeping the
# sphere cluster large enough to read in both the image and the video.
maximum_zoom_out_factor = 1.48

lattice_directions = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=np.int32,
)


def nearest_inside_lattice_point(phi, point, interior_point):
    """Round a starting point to the lattice while keeping it inside."""
    point = np.asarray(point, dtype=float)
    interior_point = np.asarray(interior_point, dtype=float)

    for _ in range(30):
        site = np.rint(N * point).astype(np.int64)
        if phi(site / N) < 0:
            return site
        point = 0.88 * point + 0.12 * interior_point

    raise RuntimeError("Could not find an interior lattice starting point.")


def simulate_slow_membrane_walk(phi, start, rng):
    """Simulate the exact discrete-time chain with slow sphere membranes.

    One of the six coordinate directions is proposed uniformly at every step.
    Crossing the boundary of any sphere is accepted with probability alpha/N.
    A rejected proposal consumes one time step without changing the position.
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
        block_length = len(block_positions)
        if block_length == 0:
            return

        first_offset = plot_stride - (block_start_step % plot_stride)
        offsets = np.arange(first_offset, block_length + 1, plot_stride)
        if len(offsets) == 0:
            return

        sampled_positions.extend(block_positions[offsets - 1] / N)
        sampled_steps.extend(block_start_step + offsets)

    while elapsed_steps < n_steps:
        block_length = min(simulation_block_size, n_steps - elapsed_steps)
        direction_indices = rng.integers(0, 6, size=block_length)
        increments = lattice_directions[direction_indices]
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

        crossing_offset = int(np.flatnonzero(crossing_attempts)[0])
        prefix = proposed_positions[:crossing_offset]
        store_samples(prefix, elapsed_steps)

        if crossing_offset > 0:
            position = prefix[-1]
            current_inside = bool(proposed_inside[crossing_offset - 1])
            elapsed_steps += crossing_offset

        old_position = position.copy()
        if rng.random() < crossing_acceptance:
            position = proposed_positions[crossing_offset]
            current_inside = bool(proposed_inside[crossing_offset])
            crossing_points.append((old_position + position) / (2.0 * N))
            crossing_step = elapsed_steps + 1
            crossing_steps.append(crossing_step)

            # Force the two endpoints of every accepted crossing into the
            # plotted trajectory.  Consequently, a color transition occurs at
            # the actual membrane edge instead of at a later subsampled site.
            if sampled_steps[-1] < elapsed_steps:
                sampled_positions.append(old_position.astype(float) / N)
                sampled_steps.append(elapsed_steps)
            else:
                sampled_positions[-1] = old_position.astype(float) / N
            sampled_positions.append(position.astype(float) / N)
            sampled_steps.append(crossing_step)
        else:
            holding_steps.append(elapsed_steps + 1)

        elapsed_steps += 1
        if (
            elapsed_steps % plot_stride == 0
            and sampled_steps[-1] < elapsed_steps
        ):
            sampled_positions.append(position.astype(float) / N)
            sampled_steps.append(elapsed_steps)

    if sampled_steps[-1] != n_steps:
        sampled_positions.append(position.astype(float) / N)
        sampled_steps.append(n_steps)

    sampled_positions = np.asarray(sampled_positions)
    sampled_steps = np.asarray(sampled_steps, dtype=np.int64)
    crossing_steps = np.asarray(crossing_steps, dtype=np.int64)
    crossing_indices = np.searchsorted(sampled_steps, crossing_steps, side="left")
    crossing_indices = np.clip(
        crossing_indices,
        0,
        len(sampled_positions) - 1,
    )

    crossing_points = np.asarray(crossing_points, dtype=float)
    if not len(crossing_points):
        crossing_points = np.empty((0, 3), dtype=float)

    return (
        sampled_positions,
        sampled_steps,
        crossing_points,
        crossing_indices,
        np.asarray(holding_steps, dtype=np.int64),
    )


# The initial point changes every run but stays near the center of the central
# sphere, between 4% and 18% of its radius.
start_direction = normalized(rng_geometry.normal(size=3))
start_fraction = rng_geometry.uniform(0.04, 0.18)
start_point = start_fraction * central_radius * start_direction
start = nearest_inside_lattice_point(
    phi_sphere_cluster,
    start_point,
    interior_point=sphere_centers[0],
)

if np.linalg.norm(start / N - sphere_centers[0]) >= sphere_radii[0]:
    raise RuntimeError("The walk did not start inside the central sphere.")

(
    walk,
    walk_steps,
    crossings,
    crossing_indices,
    holding_steps,
) = simulate_slow_membrane_walk(phi_sphere_cluster, start, rng_walk)

crossing_labels = []
for crossing_number, crossing_index in enumerate(crossing_indices, start=1):
    before_inside = bool(phi_sphere_cluster(walk[crossing_index - 1]) <= 0)
    after_inside = bool(phi_sphere_cluster(walk[crossing_index]) <= 0)
    if before_inside == after_inside:
        raise RuntimeError("A stored color transition is not a true crossing.")
    direction_label = (
        "inside to outside" if before_inside else "outside to inside"
    )
    crossing_labels.append(
        f"accepted crossing {crossing_number}<br>{direction_label}"
    )

print(f"spheres: {sphere_count}")
print(f"accepted crossings: {len(crossings)}")
print(f"rejected boundary attempts: {len(holding_steps)}")
print(f"start fraction of central radius: {start_fraction:.3f}")


# -----------------------------------------------------------------------------
# Surfaces, colors, and camera windows
# -----------------------------------------------------------------------------
azimuth_values = np.linspace(0.0, 2.0 * np.pi, 64)
polar_values = np.linspace(0.0, np.pi, 34)
azimuth_grid, polar_grid = np.meshgrid(azimuth_values, polar_values)
unit_sphere_surface = np.stack(
    [
        np.sin(polar_grid) * np.cos(azimuth_grid),
        np.sin(polar_grid) * np.sin(azimuth_grid),
        np.cos(polar_grid),
    ],
    axis=-1,
)

sphere_surfaces = []
for center, radius in zip(sphere_centers, sphere_radii):
    points = center + radius * unit_sphere_surface
    sphere_surfaces.append((points[..., 0], points[..., 1], points[..., 2]))

initial_hue = rng_colors.uniform(0.0, 1.0)
golden_angle = 0.618033988749895
start_marker_color = "#3F88C5"
surface_color = "#91B7C7"
surface_wire_color = "#526A75"


def walk_color(index):
    """Pastel colors separated by the golden angle."""
    hue = (initial_hue + index * golden_angle) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.75, 0.58)
    return "#{:02X}{:02X}{:02X}".format(
        round(255 * red),
        round(255 * green),
        round(255 * blue),
    )


geometry_low = np.min(sphere_centers - sphere_radii[:, None], axis=0)
geometry_high = np.max(sphere_centers + sphere_radii[:, None], axis=0)
geometry_center = 0.5 * (geometry_low + geometry_high)
base_half_extent = 0.55 * float(np.max(geometry_high - geometry_low))


def centered_cube_limits(center, half_extent):
    center = np.asarray(center, dtype=float)
    return tuple(
        (coordinate - half_extent, coordinate + half_extent)
        for coordinate in center
    )


def static_cube_limits(trajectory):
    robust_low = np.quantile(trajectory, 0.002, axis=0)
    robust_high = np.quantile(trajectory, 0.998, axis=0)
    required_half_extent = max(
        base_half_extent,
        float(np.max(np.abs(robust_low - geometry_center))),
        float(np.max(np.abs(robust_high - geometry_center))),
    )
    half_extent = min(
        1.04 * required_half_extent,
        maximum_zoom_out_factor * base_half_extent,
    )
    return centered_cube_limits(geometry_center, half_extent)


def set_centered_cube_window(ax, half_extent):
    limits = centered_cube_limits(geometry_center, half_extent)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


def best_projection_azimuth(points, elevation=22.0):
    centered_points = points - np.mean(points, axis=0)
    best_azimuth = -57.0
    best_area = -np.inf
    elevation_radians = np.deg2rad(elevation)

    for azimuth in np.linspace(-180.0, 175.0, 72):
        azimuth_radians = np.deg2rad(azimuth)
        horizontal = np.array(
            [-np.sin(azimuth_radians), np.cos(azimuth_radians), 0.0]
        )
        viewing_direction = np.array(
            [
                np.cos(elevation_radians) * np.cos(azimuth_radians),
                np.cos(elevation_radians) * np.sin(azimuth_radians),
                np.sin(elevation_radians),
            ]
        )
        vertical = normalized(np.cross(viewing_direction, horizontal))
        horizontal_extent = np.ptp(centered_points @ horizontal)
        vertical_extent = np.ptp(centered_points @ vertical)
        projected_area = horizontal_extent * vertical_extent
        if projected_area > best_area:
            best_area = projected_area
            best_azimuth = float(azimuth)

    return best_azimuth


view_elevation = 22.0
view_azimuth = best_projection_azimuth(sphere_centers, view_elevation)


def draw_sphere_cluster(ax, coarse=False):
    for surface in sphere_surfaces:
        ax.plot_surface(
            *surface,
            color=surface_color,
            alpha=0.22,
            linewidth=0,
            antialiased=True,
            shade=False,
            rstride=3 if coarse else 1,
            cstride=4 if coarse else 1,
        )
        ax.plot_wireframe(
            *surface,
            rstride=7 if coarse else 5,
            cstride=9 if coarse else 6,
            color=surface_wire_color,
            linewidth=0.28,
            alpha=0.26,
        )


def configure_3d_axis(ax, limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=view_elevation, azim=view_azimuth)
    ax.set_axis_off()
    ax.set_facecolor("white")


def segment_endpoints_for(trajectory, indices):
    return np.concatenate(([0], indices, [len(trajectory) - 1])).astype(
        np.int64
    )


def draw_static_walk(ax):
    segment_endpoints = segment_endpoints_for(walk, crossing_indices)
    for segment_index in range(len(segment_endpoints) - 1):
        segment_start = int(segment_endpoints[segment_index])
        segment_stop = int(segment_endpoints[segment_index + 1])
        segment = walk[segment_start : segment_stop + 1]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=walk_color(segment_index),
            linewidth=0.46,
            alpha=0.75,
        )

    ax.scatter(
        walk[0, 0],
        walk[0, 1],
        walk[0, 2],
        s=28,
        color=start_marker_color,
        edgecolors="white",
        linewidths=0.55,
        depthshade=False,
    )

    if len(crossings):
        crossing_colors = [
            walk_color(index + 1) for index in range(len(crossings))
        ]
        ax.scatter(
            crossings[:, 0],
            crossings[:, 1],
            crossings[:, 2],
            s=23,
            color=crossing_colors,
            edgecolors="white",
            linewidths=0.4,
            depthshade=False,
        )


# -----------------------------------------------------------------------------
# Final image
# -----------------------------------------------------------------------------
figure = plt.figure(figsize=(8.4, 7.8))
axis = figure.add_subplot(111, projection="3d")
draw_sphere_cluster(axis, coarse=False)
draw_static_walk(axis)
configure_3d_axis(axis, static_cube_limits(walk))
figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
image_path = Path(__file__).with_name(
    f"snob_multiple_spheres_3d_{run_id}.png"
)
figure.savefig(
    image_path,
    dpi=240,
    bbox_inches="tight",
    pad_inches=0.02,
    facecolor="white",
    transparent=False,
)
plt.close(figure)
print(f"image saved to: {image_path}")


# -----------------------------------------------------------------------------
# Animation of the same realization
# -----------------------------------------------------------------------------
def decimate_for_video(trajectory, trajectory_steps, indices):
    stride = max(1, int(np.ceil(len(trajectory) / maximum_video_points)))
    retained_indices = np.arange(0, len(trajectory), stride, dtype=np.int64)
    pre_crossing_indices = np.maximum(indices - 1, 0)
    retained_indices = np.unique(
        np.concatenate(
            [
                retained_indices,
                pre_crossing_indices,
                indices,
                [len(trajectory) - 1],
            ]
        )
    )
    video_trajectory = trajectory[retained_indices]
    video_crossing_indices = np.searchsorted(retained_indices, indices)
    video_times = trajectory_steps[retained_indices] / n_steps
    return video_trajectory, video_crossing_indices, video_times


video_walk, video_crossing_indices, video_times = decimate_for_video(
    walk,
    walk_steps,
    crossing_indices,
)
video_cumulative_extent = np.maximum.accumulate(
    np.max(np.abs(video_walk - geometry_center), axis=1)
)


def trajectory_index_at_progress(times, progress):
    return min(
        len(times) - 1,
        max(
            0,
            int(np.searchsorted(times, progress, side="right") - 1),
        ),
    )


def dynamic_half_extent(current_index):
    required_half_extent = max(
        base_half_extent,
        1.04 * float(video_cumulative_extent[current_index]),
    )
    return min(
        required_half_extent,
        maximum_zoom_out_factor * base_half_extent,
    )


def wrapped_angle_difference(target, current):
    """Shortest signed difference between two angles in degrees."""
    return (target - current + 180.0) % 360.0 - 180.0


def build_camera_follow_schedule():
    """Create a smooth camera orbit that keeps the particle on the near side.

    Close to the center, the camera makes only a gentle idle rotation.  Once
    the particle approaches or leaves the central membrane, the target camera
    direction follows its position relative to the displayed cluster.  Angular
    speed limits prevent the lattice-scale fluctuations from shaking the view.
    """
    camera_azimuths = np.empty(video_frame_count, dtype=float)
    camera_elevations = np.empty(video_frame_count, dtype=float)
    camera_azimuths[0] = view_azimuth
    camera_elevations[0] = view_elevation
    follow_distance = 0.62 * sphere_radii[0]

    for frame_number in range(1, video_frame_count):
        progress = frame_number / (video_frame_count - 1)
        current_index = trajectory_index_at_progress(video_times, progress)
        current_point = video_walk[current_index]
        distance_from_central_sphere = np.linalg.norm(
            current_point - sphere_centers[0]
        )

        if distance_from_central_sphere < follow_distance:
            # A slow orbit avoids a frozen camera without reacting to every
            # small displacement near the center.
            target_azimuth = view_azimuth + 28.0 * progress
            target_elevation = view_elevation
        else:
            offset = current_point - geometry_center
            horizontal_distance = np.hypot(offset[0], offset[1])
            target_azimuth = np.degrees(np.arctan2(offset[1], offset[0]))
            target_elevation = np.clip(
                np.degrees(np.arctan2(offset[2], horizontal_distance)),
                -28.0,
                38.0,
            )

        azimuth_error = wrapped_angle_difference(
            target_azimuth,
            camera_azimuths[frame_number - 1],
        )
        elevation_error = (
            target_elevation - camera_elevations[frame_number - 1]
        )
        camera_azimuths[frame_number] = (
            camera_azimuths[frame_number - 1]
            + np.clip(0.16 * azimuth_error, -2.6, 2.6)
        )
        camera_elevations[frame_number] = (
            camera_elevations[frame_number - 1]
            + np.clip(0.13 * elevation_error, -1.15, 1.15)
        )

    return camera_azimuths, camera_elevations


camera_azimuths, camera_elevations = build_camera_follow_schedule()


video_segment_endpoints = segment_endpoints_for(
    video_walk,
    video_crossing_indices,
)
crossing_colors = [
    walk_color(index + 1) for index in range(len(crossings))
]


# -----------------------------------------------------------------------------
# Interactive 3D figure: drag to rotate, scroll to zoom, hover over crossings
# -----------------------------------------------------------------------------
interactive_figure = go.Figure()

for surface in sphere_surfaces:
    interactive_figure.add_trace(
        go.Surface(
            x=surface[0][::2, ::2],
            y=surface[1][::2, ::2],
            z=surface[2][::2, ::2],
            surfacecolor=np.zeros_like(surface[0][::2, ::2]),
            colorscale=[
                [0.0, surface_color],
                [1.0, surface_color],
            ],
            cmin=0.0,
            cmax=1.0,
            showscale=False,
            opacity=0.20,
            hoverinfo="skip",
            lighting={
                "ambient": 0.82,
                "diffuse": 0.45,
                "specular": 0.08,
                "roughness": 0.90,
            },
        )
    )

for segment_index in range(len(video_segment_endpoints) - 1):
    segment_start = int(video_segment_endpoints[segment_index])
    segment_stop = int(video_segment_endpoints[segment_index + 1])
    segment = video_walk[segment_start : segment_stop + 1]
    interactive_figure.add_trace(
        go.Scatter3d(
            x=segment[:, 0],
            y=segment[:, 1],
            z=segment[:, 2],
            mode="lines",
            line={"color": walk_color(segment_index), "width": 3.0},
            opacity=0.82,
            hoverinfo="skip",
            showlegend=False,
        )
    )

interactive_figure.add_trace(
    go.Scatter3d(
        x=[video_walk[0, 0]],
        y=[video_walk[0, 1]],
        z=[video_walk[0, 2]],
        mode="markers",
        marker={
            "size": 6.5,
            "color": start_marker_color,
            "line": {"color": "white", "width": 1.0},
        },
        text=["starting point"],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
)

if len(crossings):
    interactive_figure.add_trace(
        go.Scatter3d(
            x=crossings[:, 0],
            y=crossings[:, 1],
            z=crossings[:, 2],
            mode="markers",
            marker={
                "size": 6.0,
                "color": crossing_colors,
                "line": {"color": "white", "width": 1.0},
            },
            text=crossing_labels,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

interactive_limits = static_cube_limits(walk)
initial_azimuth_radians = np.deg2rad(view_azimuth)
initial_elevation_radians = np.deg2rad(view_elevation)
camera_radius = 1.75
interactive_camera = {
    "eye": {
        "x": camera_radius
        * np.cos(initial_elevation_radians)
        * np.cos(initial_azimuth_radians),
        "y": camera_radius
        * np.cos(initial_elevation_radians)
        * np.sin(initial_azimuth_radians),
        "z": camera_radius * np.sin(initial_elevation_radians),
    },
    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
}

interactive_figure.update_layout(
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin={"l": 0, "r": 0, "b": 0, "t": 0},
    scene={
        "xaxis": {
            "visible": False,
            "range": list(interactive_limits[0]),
        },
        "yaxis": {
            "visible": False,
            "range": list(interactive_limits[1]),
        },
        "zaxis": {
            "visible": False,
            "range": list(interactive_limits[2]),
        },
        "aspectmode": "cube",
        "camera": interactive_camera,
        "dragmode": "orbit",
        "bgcolor": "white",
    },
)

interactive_path = Path(__file__).with_name(
    f"snob_multiple_spheres_3d_{run_id}_interactive.html"
)
interactive_figure.write_html(
    interactive_path,
    include_plotlyjs=True,
    full_html=True,
    auto_open=False,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
    },
)
print(f"interactive 3D figure saved to: {interactive_path}")


video_figure = plt.figure(figsize=(8.4, 7.8))
video_axis = video_figure.add_subplot(111, projection="3d")
draw_sphere_cluster(video_axis, coarse=True)
configure_3d_axis(
    video_axis,
    centered_cube_limits(geometry_center, base_half_extent),
)
video_figure.subplots_adjust(left=0.0, right=1.0, bottom=0.13, top=1.0)

counter = video_figure.text(
    0.5,
    0.052,
    "",
    ha="center",
    va="center",
    fontsize=11.0,
    family="monospace",
    color="#303030",
)

line_artists = []
for segment_index in range(len(video_segment_endpoints) - 1):
    line, = video_axis.plot(
        [],
        [],
        [],
        color=walk_color(segment_index),
        linewidth=0.56,
        alpha=0.80,
    )
    line_artists.append(line)

video_axis.scatter(
    video_walk[0, 0],
    video_walk[0, 1],
    video_walk[0, 2],
    s=28,
    color=start_marker_color,
    edgecolors="white",
    linewidths=0.55,
    depthshade=False,
)

crossing_artist = video_axis.scatter(
    [],
    [],
    [],
    s=22,
    edgecolors="white",
    linewidths=0.4,
    depthshade=False,
)
current_position_artist = video_axis.scatter(
    video_walk[0, 0],
    video_walk[0, 1],
    video_walk[0, 2],
    s=36,
    color=walk_color(0),
    edgecolors="white",
    linewidths=0.75,
    depthshade=False,
)


def update_video(frame_number):
    progress = frame_number / (video_frame_count - 1)
    elapsed_steps = min(n_steps, int(round(progress * n_steps)))
    current_index = trajectory_index_at_progress(video_times, progress)

    set_centered_cube_window(
        video_axis,
        dynamic_half_extent(current_index),
    )
    video_axis.view_init(
        elev=camera_elevations[frame_number],
        azim=camera_azimuths[frame_number],
    )

    for segment_index, line in enumerate(line_artists):
        segment_start = int(video_segment_endpoints[segment_index])
        segment_stop = min(
            current_index,
            int(video_segment_endpoints[segment_index + 1]),
        )
        if segment_stop < segment_start:
            line.set_data_3d([], [], [])
        else:
            line.set_data_3d(
                video_walk[segment_start : segment_stop + 1, 0],
                video_walk[segment_start : segment_stop + 1, 1],
                video_walk[segment_start : segment_stop + 1, 2],
            )

    visible_crossings = int(
        np.searchsorted(
            video_crossing_indices,
            current_index,
            side="right",
        )
    )
    if visible_crossings:
        visible_points = crossings[:visible_crossings]
        crossing_artist._offsets3d = (
            visible_points[:, 0],
            visible_points[:, 1],
            visible_points[:, 2],
        )
        crossing_artist.set_facecolors(crossing_colors[:visible_crossings])
    else:
        empty = np.empty(0)
        crossing_artist._offsets3d = (empty, empty, empty)

    current_segment = int(
        np.searchsorted(
            video_crossing_indices,
            current_index,
            side="right",
        )
    )
    current_point = video_walk[current_index]
    current_position_artist._offsets3d = (
        [current_point[0]],
        [current_point[1]],
        [current_point[2]],
    )
    current_position_artist.set_facecolors([walk_color(current_segment)])

    holding_count = int(
        np.searchsorted(holding_steps, elapsed_steps, side="right")
    )
    counter.set_text(
        f"steps: {elapsed_steps:,} / {n_steps:,}\n"
        f"rejected boundary attempts: {holding_count:,}"
    )

    return [
        *line_artists,
        crossing_artist,
        current_position_artist,
        counter,
    ]


movie = animation.FuncAnimation(
    video_figure,
    update_video,
    frames=video_frame_count,
    interval=1_000 / video_frames_per_second,
    blit=False,
)
video_path = Path(__file__).with_name(
    f"snob_multiple_spheres_3d_{run_id}.mp4"
)
video_writer = animation.FFMpegWriter(
    fps=video_frames_per_second,
    codec="libx264",
    bitrate=3_600,
    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
)
movie.save(video_path, writer=video_writer, dpi=120)
plt.close(video_figure)
print(f"video saved to: {video_path}")
import colorsys
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# With seed=None, the irregular membrane, starting sites, pastel palette, and
# both walks are regenerated at every execution.  Set an integer to reproduce
# one particular realization.
seed = None
seed_sequence = np.random.SeedSequence(seed)
geometry_seed, sphere_seed, irregular_seed, color_seed = seed_sequence.spawn(4)
rng_geometry = np.random.default_rng(geometry_seed)
rng_sphere = np.random.default_rng(sphere_seed)
rng_irregular = np.random.default_rng(irregular_seed)
rng_colors = np.random.default_rng(color_seed)
print(f"run seed: {seed_sequence.entropy}")


# -----------------------------------------------------------------------------
# Three-dimensional membranes
# -----------------------------------------------------------------------------
sphere_radius = 1.0


def phi_sphere(xyz):
    """Negative inside the fixed unit sphere and positive outside it."""
    xyz = np.asarray(xyz)
    return np.linalg.norm(xyz, axis=-1) - sphere_radius


# The second membrane is sampled from a much richer family.  A lumpy main body
# is joined to a random finite tree of trapping chambers through curved tubes
# and overlapping connector regions.  All centers, directions, bends, radii,
# bottlenecks, spherical modes, and the number of chambers are regenerated
# independently on every execution.


def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def perpendicular_basis(direction):
    """Return two unit vectors perpendicular to a given direction."""
    direction = normalized(direction)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, reference)) > 0.86:
        reference = np.array([0.0, 1.0, 0.0])
    first = normalized(np.cross(direction, reference))
    second = normalized(np.cross(direction, first))
    return first, second


def legendre_polynomial(degree, values):
    """Evaluate a Legendre polynomial without an additional dependency."""
    values = np.asarray(values)
    if degree == 0:
        return np.ones_like(values)
    if degree == 1:
        return values
    previous_previous = np.ones_like(values)
    previous = values
    for index in range(2, degree + 1):
        current = (
            (2 * index - 1) * values * previous
            - (index - 1) * previous_previous
        ) / index
        previous_previous, previous = previous, current
    return previous


def random_lumpy_body(center, base_radius, mode_count=None):
    """Create a random star-shaped body from rotated spherical modes."""
    if mode_count is None:
        mode_count = int(rng_geometry.integers(4, 7))
    mode_directions = rng_geometry.normal(size=(mode_count, 3))
    mode_directions /= np.linalg.norm(mode_directions, axis=1)[:, None]
    mode_degrees = rng_geometry.integers(2, 7, size=mode_count)
    mode_amplitudes = rng_geometry.uniform(0.035, 0.075, size=mode_count)
    mode_amplitudes *= rng_geometry.choice([-1.0, 1.0], size=mode_count)
    return {
        "center": np.asarray(center, dtype=float),
        "base_radius": float(base_radius),
        "directions": mode_directions,
        "degrees": mode_degrees,
        "amplitudes": mode_amplitudes,
    }


def lumpy_body_radius(body, directions):
    directions = np.asarray(directions, dtype=float)
    modulation = np.ones(directions.shape[:-1], dtype=float)
    for axis, degree, amplitude in zip(
        body["directions"],
        body["degrees"],
        body["amplitudes"],
    ):
        cosine = np.sum(directions * axis, axis=-1)
        modulation += amplitude * legendre_polynomial(int(degree), cosine)
    modulation = np.maximum(modulation, 0.58)
    return body["base_radius"] * modulation


def phi_lumpy_body(xyz, body):
    xyz = np.asarray(xyz)
    displacement = xyz - body["center"]
    distance = np.linalg.norm(displacement, axis=-1)
    directions = displacement / np.maximum(distance[..., None], 1.0e-14)
    return distance - lumpy_body_radius(body, directions)


def phi_capsule(xyz, start, end, radius):
    """Signed distance to a three-dimensional capsule."""
    xyz = np.asarray(xyz)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    parameter = np.sum((xyz - start) * segment, axis=-1) / np.dot(
        segment,
        segment,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = start + parameter[..., None] * segment
    return np.linalg.norm(xyz - closest, axis=-1) - radius


def cubic_bezier(start, control_one, control_two, end, parameters):
    parameters = np.asarray(parameters)[:, None]
    complement = 1.0 - parameters
    return (
        complement**3 * start
        + 3.0 * complement**2 * parameters * control_one
        + 3.0 * complement * parameters**2 * control_two
        + parameters**3 * end
    )


main_center = np.array(
    [
        rng_geometry.uniform(-0.72, -0.48),
        rng_geometry.uniform(-0.18, 0.18),
        rng_geometry.uniform(-0.14, 0.14),
    ]
)
main_radius = rng_geometry.uniform(0.90, 1.08)
main_body = random_lumpy_body(main_center, main_radius)

# K has a truncated geometric distribution on {1, ..., 15}.  Its probabilities
# decrease with k, but every allowed number of trapping chambers remains
# possible.  The untruncated geometric law already has a finite first moment.
maximum_trap_chambers = 15
geometric_decay = 0.78
possible_trap_counts = np.arange(1, maximum_trap_chambers + 1)
trap_count_weights = geometric_decay ** (possible_trap_counts - 1)
trap_count_probabilities = trap_count_weights / np.sum(trap_count_weights)
trap_chamber_count = int(
    rng_geometry.choice(possible_trap_counts, p=trap_count_probabilities)
)

generated_bodies = [main_body]
trap_bodies = []
trap_depths = []
tube_paths = []
connector_region_count = 0


def choose_new_chamber_direction(parent_body, child_radius, connection_gap, first):
    """Choose a direction that tends to keep new chambers separated."""
    candidate_count = 28
    if first:
        candidates = rng_geometry.normal(size=(candidate_count, 3))
        candidates[:, 0] = np.abs(candidates[:, 0]) + 0.55
    else:
        candidates = rng_geometry.normal(size=(candidate_count, 3))
        outward = parent_body["center"] - main_center
        if np.linalg.norm(outward) > 1.0e-10:
            candidates += 0.45 * normalized(outward)
    candidates /= np.linalg.norm(candidates, axis=1)[:, None]

    parent_radius = parent_body["base_radius"]
    connection_distance = parent_radius + connection_gap + child_radius
    candidate_centers = (
        parent_body["center"] + connection_distance * candidates
    )
    existing_centers = np.stack(
        [body["center"] for body in generated_bodies]
    )
    separations = np.linalg.norm(
        candidate_centers[:, None, :] - existing_centers[None, :, :],
        axis=-1,
    )
    scores = np.min(separations, axis=1)
    return candidates[int(np.argmax(scores))]


for chamber_index in range(trap_chamber_count):
    # A chamber can grow from the main body or from a previously created
    # chamber, producing a random rooted tree rather than one common corridor.
    parent_candidates = [main_body, *trap_bodies]
    parent_depth_candidates = np.array([0, *trap_depths], dtype=int)
    admissible = parent_depth_candidates < 4
    parent_weights = np.where(
        admissible,
        1.0 / (1.0 + parent_depth_candidates) ** 1.15,
        0.0,
    )
    parent_weights /= np.sum(parent_weights)
    parent_index = int(
        rng_geometry.choice(len(parent_candidates), p=parent_weights)
    )
    parent_body = parent_candidates[parent_index]
    parent_depth = int(parent_depth_candidates[parent_index])

    if chamber_index == 0:
        chamber_radius = rng_geometry.uniform(0.46, 0.68)
    else:
        chamber_radius = rng_geometry.uniform(0.25, 0.58)

    # The first connection is always narrow.  Later connections may instead
    # be broad bridges made from overlapping random regions.
    connector_kind = (
        "tube"
        if chamber_index == 0 or rng_geometry.random() < 0.60
        else "region_bridge"
    )
    if connector_kind == "tube":
        connection_gap = rng_geometry.uniform(0.42, 0.92)
    else:
        connection_gap = rng_geometry.uniform(0.30, 0.72)

    chamber_direction = choose_new_chamber_direction(
        parent_body,
        chamber_radius,
        connection_gap,
        first=chamber_index == 0,
    )
    chamber_center = (
        parent_body["center"]
        + (
            parent_body["base_radius"]
            + connection_gap
            + chamber_radius
        )
        * chamber_direction
    )
    chamber_body = random_lumpy_body(
        chamber_center,
        chamber_radius,
        mode_count=int(rng_geometry.integers(3, 6)),
    )

    parent_attachment_radius = float(
        lumpy_body_radius(parent_body, chamber_direction)
    )
    chamber_attachment_radius = float(
        lumpy_body_radius(chamber_body, -chamber_direction)
    )
    connection_start = (
        parent_body["center"]
        + 0.76 * parent_attachment_radius * chamber_direction
    )
    connection_end = (
        chamber_center
        - 0.74 * chamber_attachment_radius * chamber_direction
    )

    connection_perpendicular_one, connection_perpendicular_two = (
        perpendicular_basis(chamber_direction)
    )
    bend_scale = rng_geometry.uniform(0.24, 0.68)
    control_one = (
        connection_start
        + 0.31 * (connection_end - connection_start)
        + bend_scale
        * rng_geometry.uniform(-1.0, 1.0)
        * connection_perpendicular_one
        + bend_scale
        * rng_geometry.uniform(-0.85, 0.85)
        * connection_perpendicular_two
    )
    control_two = (
        connection_start
        + 0.69 * (connection_end - connection_start)
        + bend_scale
        * rng_geometry.uniform(-1.0, 1.0)
        * connection_perpendicular_one
        + bend_scale
        * rng_geometry.uniform(-0.85, 0.85)
        * connection_perpendicular_two
    )

    if connector_kind == "tube":
        node_count = int(rng_geometry.integers(4, 7))
        connector_nodes = cubic_bezier(
            connection_start,
            control_one,
            control_two,
            connection_end,
            np.linspace(0.0, 1.0, node_count),
        )
        base_radius = rng_geometry.uniform(0.055, 0.125)
        connector_radii = base_radius * rng_geometry.uniform(
            0.76,
            1.24,
            size=node_count,
        )
        bottleneck_index = int(rng_geometry.integers(1, node_count - 1))
        connector_radii[bottleneck_index] *= rng_geometry.uniform(0.44, 0.70)
    else:
        bridge_region_count = int(rng_geometry.integers(1, 3))
        bridge_parameters = np.arange(1, bridge_region_count + 1) / (
            bridge_region_count + 1
        )
        bridge_centers = cubic_bezier(
            connection_start,
            control_one,
            control_two,
            connection_end,
            bridge_parameters,
        )
        connector_nodes = np.vstack(
            [connection_start, bridge_centers, connection_end]
        )

        # Overlapping lumpy bodies make the visible connection region-like.
        for bridge_index, bridge_center in enumerate(bridge_centers, start=1):
            previous_distance = np.linalg.norm(
                bridge_center - connector_nodes[bridge_index - 1]
            )
            next_distance = np.linalg.norm(
                connector_nodes[bridge_index + 1] - bridge_center
            )
            bridge_radius = np.clip(
                1.06 * max(previous_distance, next_distance),
                0.16,
                0.42,
            )
            generated_bodies.append(
                random_lumpy_body(bridge_center, bridge_radius, mode_count=3)
            )
            connector_region_count += 1

        broad_radius = rng_geometry.uniform(0.12, 0.22)
        connector_radii = broad_radius * rng_geometry.uniform(
            0.90,
            1.12,
            size=len(connector_nodes),
        )

    tube_paths.append(
        {
            "nodes": connector_nodes,
            "radii": connector_radii,
            "kind": connector_kind,
        }
    )
    trap_bodies.append(chamber_body)
    trap_depths.append(parent_depth + 1)
    generated_bodies.append(chamber_body)

tube_segments = []
for path_index, path in enumerate(tube_paths):
    for segment_index in range(len(path["nodes"]) - 1):
        tube_segments.append(
            {
                "start": path["nodes"][segment_index],
                "end": path["nodes"][segment_index + 1],
                "radius": float(
                    min(
                        path["radii"][segment_index],
                        path["radii"][segment_index + 1],
                    )
                ),
                "path_index": path_index,
                "segment_index": segment_index,
                "kind": path["kind"],
            }
        )


def phi_generated(xyz):
    """Level set of the random union of bodies, corridors, and pockets."""
    level_set = phi_lumpy_body(xyz, generated_bodies[0])
    for body in generated_bodies[1:]:
        level_set = np.minimum(level_set, phi_lumpy_body(xyz, body))
    for segment in tube_segments:
        level_set = np.minimum(
            level_set,
            phi_capsule(
                xyz,
                segment["start"],
                segment["end"],
                segment["radius"],
            ),
        )
    return level_set


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

# Keep the membranes visually prominent even when the walk makes a very long
# excursion.  Trajectory portions beyond this capped camera window are clipped.
maximum_zoom_out_factor = 1.55

# The irregular-region particle starts in the MAIN CENTRAL BODY, not in a
# terminal pocket.  The fraction is measured relative to that body's local
# radius; keeping it below 0.28 places the particle well away from the membrane.
generated_central_start_fraction_range = (0.05, 0.28)

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
        site = np.rint(N * point).astype(np.int32)
        if phi(site / N) < 0:
            return site
        point = 0.88 * point + 0.12 * interior_point

    raise RuntimeError("Could not find an interior lattice starting point.")


def simulate_slow_membrane_walk(phi, start, rng):
    """Simulate the exact discrete-time chain with a slow membrane.

    One of the six coordinate directions is proposed uniformly at every step.
    A crossing proposal is accepted with probability alpha/N.  A rejected
    proposal consumes one time step while leaving the position unchanged.
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
        proposed_direction_indices = rng.integers(0, 6, size=block_length)
        increments = lattice_directions[proposed_direction_indices]
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
            crossing_steps.append(elapsed_steps + 1)
        else:
            holding_steps.append(elapsed_steps + 1)

        elapsed_steps += 1
        if elapsed_steps % plot_stride == 0:
            sampled_positions.append(position.astype(float) / N)
            sampled_steps.append(elapsed_steps)

    if sampled_steps[-1] != n_steps:
        sampled_positions.append(position.astype(float) / N)
        sampled_steps.append(n_steps)

    sampled_positions = np.asarray(sampled_positions)
    sampled_steps = np.asarray(sampled_steps, dtype=np.int64)
    crossing_steps = np.asarray(crossing_steps, dtype=np.int64)
    crossing_indices = np.searchsorted(sampled_steps, crossing_steps, side="left")
    crossing_indices = np.clip(crossing_indices, 0, len(sampled_positions) - 1)

    crossing_points = np.asarray(crossing_points, dtype=float)
    if not len(crossing_points):
        crossing_points = np.empty((0, 3), dtype=float)

    return (
        sampled_positions,
        crossing_points,
        crossing_indices,
        np.asarray(holding_steps, dtype=np.int64),
    )


# Both walks start strictly inside their regions, at a visible distance from
# the membrane rather than one lattice spacing away from it.
sphere_start_direction = rng_geometry.normal(size=3)
sphere_start_direction /= np.linalg.norm(sphere_start_direction)
sphere_start_fraction = rng_geometry.uniform(0.25, 0.65)
sphere_start_point = (
    sphere_start_fraction * sphere_radius * sphere_start_direction
)
sphere_start = nearest_inside_lattice_point(
    phi_sphere,
    sphere_start_point,
    interior_point=np.zeros(3),
)

# The second walk starts deep inside the main, central body.  Its direction and
# radial fraction remain random, so executions do not reuse the same site.
generated_start_body = main_body
generated_start_direction = normalized(rng_geometry.normal(size=3))
generated_start_fraction = rng_geometry.uniform(
    *generated_central_start_fraction_range
)
generated_directional_radius = float(
    lumpy_body_radius(generated_start_body, generated_start_direction)
)
generated_interior_point = generated_start_body["center"]
generated_start_point = (
    generated_interior_point
    + generated_start_fraction
    * generated_directional_radius
    * generated_start_direction
)
generated_start = nearest_inside_lattice_point(
    phi_generated,
    generated_start_point,
    interior_point=generated_interior_point,
)

if phi_sphere(sphere_start / N) >= 0:
    raise RuntimeError("The sphere walk did not start strictly inside.")
if phi_generated(generated_start / N) >= 0:
    raise RuntimeError("The generated-region walk did not start strictly inside.")

(
    sphere_walk,
    sphere_crossings,
    sphere_crossing_indices,
    sphere_holding_steps,
) = simulate_slow_membrane_walk(phi_sphere, sphere_start, rng_sphere)

(
    generated_walk,
    generated_crossings,
    generated_crossing_indices,
    generated_holding_steps,
) = simulate_slow_membrane_walk(phi_generated, generated_start, rng_irregular)

print(f"sphere: {len(sphere_crossings)} crossings")
print(f"generated region: {len(generated_crossings)} crossings")
print(
    "generated trap geometry: "
    f"{trap_chamber_count} chambers, "
    f"{connector_region_count} connector regions, "
    f"{len(tube_paths)} connections"
)
print(f"sphere: {len(sphere_holding_steps)} membrane holding steps")
print(
    "generated region: "
    f"{len(generated_holding_steps)} membrane holding steps"
)


# -----------------------------------------------------------------------------
# Surface meshes, colors, and camera windows
# -----------------------------------------------------------------------------
sphere_azimuth = np.linspace(0.0, 2.0 * np.pi, 64)
sphere_polar = np.linspace(0.0, np.pi, 34)
sphere_azimuth_grid, sphere_polar_grid = np.meshgrid(
    sphere_azimuth,
    sphere_polar,
)
sphere_surface = (
    sphere_radius * np.sin(sphere_polar_grid) * np.cos(sphere_azimuth_grid),
    sphere_radius * np.sin(sphere_polar_grid) * np.sin(sphere_azimuth_grid),
    sphere_radius * np.cos(sphere_polar_grid),
)

unit_sphere_surface = np.stack(sphere_surface, axis=-1) / sphere_radius


def lumpy_body_surface(body):
    radii = lumpy_body_radius(body, unit_sphere_surface)
    points = body["center"] + radii[..., None] * unit_sphere_surface
    return points[..., 0], points[..., 1], points[..., 2]


def swept_tube_surface(path, angle_count=44):
    """Continuous visual mesh following a random polyline corridor."""
    nodes = path["nodes"]
    radii = path["radii"]
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count)
    points = np.empty((len(nodes), angle_count, 3), dtype=float)

    for node_index, (node, radius) in enumerate(zip(nodes, radii)):
        if node_index == 0:
            tangent = nodes[1] - nodes[0]
        elif node_index == len(nodes) - 1:
            tangent = nodes[-1] - nodes[-2]
        else:
            tangent = nodes[node_index + 1] - nodes[node_index - 1]
        first, second = perpendicular_basis(tangent)
        ring_directions = (
            np.cos(angles)[:, None] * first
            + np.sin(angles)[:, None] * second
        )
        points[node_index] = node + radius * ring_directions

    return points[..., 0], points[..., 1], points[..., 2]


generated_body_surfaces = [
    lumpy_body_surface(body) for body in generated_bodies
]
generated_tube_surfaces = [
    swept_tube_surface(path) for path in tube_paths
]
generated_geometry_points = np.concatenate(
    [
        np.column_stack([surface_axis.ravel() for surface_axis in surface])
        for surface in [*generated_body_surfaces, *generated_tube_surfaces]
    ],
    axis=0,
)


def best_projection_azimuth(points, elevation=22.0):
    """Choose a camera angle that exposes the largest projected branching."""
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


generated_camera_points = np.vstack(
    [
        *[body["center"] for body in generated_bodies],
        *[path["nodes"] for path in tube_paths],
    ]
)
generated_view_azimuth = best_projection_azimuth(generated_camera_points)

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


def cube_limits(
    geometry_center,
    base_half_extent,
    trajectory,
    maximum_factor=maximum_zoom_out_factor,
):
    """A common cubic window with a capped adaptive zoom-out."""
    geometry_center = np.asarray(geometry_center, dtype=float)
    robust_low = np.quantile(trajectory, 0.002, axis=0)
    robust_high = np.quantile(trajectory, 0.998, axis=0)
    required_half_extent = max(
        base_half_extent,
        float(np.max(np.abs(robust_low - geometry_center))),
        float(np.max(np.abs(robust_high - geometry_center))),
    )
    half_extent = min(
        1.04 * required_half_extent,
        maximum_factor * base_half_extent,
    )
    return tuple(
        (center - half_extent, center + half_extent)
        for center in geometry_center
    )


sphere_geometry_center = np.zeros(3)
sphere_base_half_extent = 1.12 * sphere_radius
sphere_limits = cube_limits(
    geometry_center=sphere_geometry_center,
    base_half_extent=sphere_base_half_extent,
    trajectory=sphere_walk,
)
generated_geometry_low = np.min(generated_geometry_points, axis=0)
generated_geometry_high = np.max(generated_geometry_points, axis=0)
generated_geometry_center = 0.5 * (
    generated_geometry_low + generated_geometry_high
)
generated_base_half_extent = 0.56 * float(
    np.max(generated_geometry_high - generated_geometry_low)
)
generated_limits = cube_limits(
    geometry_center=generated_geometry_center,
    base_half_extent=generated_base_half_extent,
    trajectory=generated_walk,
)


def centered_cube_limits(center, half_extent):
    """Return equal-size x, y, and z limits around one fixed center."""
    center = np.asarray(center, dtype=float)
    return tuple(
        (coordinate - half_extent, coordinate + half_extent)
        for coordinate in center
    )


def set_centered_cube_window(ax, center, half_extent):
    """Apply one cubic camera window without changing its center."""
    limits = centered_cube_limits(center, half_extent)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


def trajectory_index_at_progress(trajectory_times, progress):
    return min(
        len(trajectory_times) - 1,
        max(
            0,
            int(np.searchsorted(
                trajectory_times,
                progress,
                side="right",
            ) - 1),
        ),
    )


def dynamic_video_half_extent(
    cumulative_extent,
    current_index,
    base_half_extent,
):
    """Zoom out only as needed, but never beyond the prescribed cap."""
    required_half_extent = max(
        base_half_extent,
        1.04 * float(cumulative_extent[current_index]),
    )
    return min(
        required_half_extent,
        maximum_zoom_out_factor * base_half_extent,
    )


def draw_sphere_surface(ax, coarse=False):
    surface_row_stride = 3 if coarse else 1
    surface_column_stride = 4 if coarse else 1
    ax.plot_surface(
        *sphere_surface,
        color=surface_color,
        alpha=0.22,
        linewidth=0,
        antialiased=True,
        shade=False,
        rstride=surface_row_stride,
        cstride=surface_column_stride,
    )
    ax.plot_wireframe(
        *sphere_surface,
        rstride=7 if coarse else 5,
        cstride=9 if coarse else 6,
        color=surface_wire_color,
        linewidth=0.28,
        alpha=0.26,
    )


def draw_generated_surface(ax, coarse=False):
    body_row_stride = 3 if coarse else 1
    body_column_stride = 4 if coarse else 1
    for surface in generated_body_surfaces:
        ax.plot_surface(
            *surface,
            color=surface_color,
            alpha=0.22,
            linewidth=0,
            antialiased=True,
            shade=False,
            rstride=body_row_stride,
            cstride=body_column_stride,
        )
        ax.plot_wireframe(
            *surface,
            rstride=8 if coarse else 5,
            cstride=10 if coarse else 6,
            color=surface_wire_color,
            linewidth=0.26,
            alpha=0.24,
        )

    for surface in generated_tube_surfaces:
        ax.plot_surface(
            *surface,
            color=surface_color,
            alpha=0.24,
            linewidth=0,
            antialiased=True,
            shade=False,
            rstride=1,
            cstride=4 if coarse else 1,
        )
        ax.plot_wireframe(
            *surface,
            rstride=1,
            cstride=5,
            color=surface_wire_color,
            linewidth=0.25,
            alpha=0.22,
        )


def configure_3d_axis(ax, limits, azimuth=-57.0):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22.0, azim=azimuth)
    ax.set_axis_off()


def plot_walk_by_crossings(ax, trajectory, crossing_indices, linewidth=0.38):
    endpoints = np.concatenate(
        ([0], crossing_indices, [len(trajectory) - 1])
    )
    for segment_index in range(len(endpoints) - 1):
        start = int(endpoints[segment_index])
        stop = int(endpoints[segment_index + 1])
        ax.plot(
            trajectory[start : stop + 1, 0],
            trajectory[start : stop + 1, 1],
            trajectory[start : stop + 1, 2],
            color=walk_color(segment_index),
            linewidth=linewidth,
            alpha=0.76,
        )


def add_static_markers(ax, trajectory, crossing_points):
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        s=25,
        color=start_marker_color,
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
    )
    if len(crossing_points):
        crossing_colors = [
            walk_color(index + 1) for index in range(len(crossing_points))
        ]
        ax.scatter(
            crossing_points[:, 0],
            crossing_points[:, 1],
            crossing_points[:, 2],
            s=20,
            color=crossing_colors,
            edgecolors="white",
            linewidths=0.35,
            depthshade=False,
        )


# -----------------------------------------------------------------------------
# Final static image
# -----------------------------------------------------------------------------
figure, (axis_sphere, axis_generated) = plt.subplots(
    1,
    2,
    figsize=(14.0, 6.2),
    subplot_kw={"projection": "3d"},
)

draw_sphere_surface(axis_sphere)
draw_generated_surface(axis_generated)
plot_walk_by_crossings(axis_sphere, sphere_walk, sphere_crossing_indices)
plot_walk_by_crossings(
    axis_generated,
    generated_walk,
    generated_crossing_indices,
)
add_static_markers(axis_sphere, sphere_walk, sphere_crossings)
add_static_markers(axis_generated, generated_walk, generated_crossings)
configure_3d_axis(axis_sphere, sphere_limits)
configure_3d_axis(
    axis_generated,
    generated_limits,
    azimuth=generated_view_azimuth,
)
figure.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    wspace=0.015,
)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
image_path = Path(__file__).with_name(f"snob_two_regions_3d_{run_id}.png")
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
def decimate_for_video(trajectory, crossing_indices):
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
    sphere_video_walk,
    sphere_video_crossing_indices,
    sphere_video_times,
) = decimate_for_video(sphere_walk, sphere_crossing_indices)
(
    generated_video_walk,
    generated_video_crossing_indices,
    generated_video_times,
) = decimate_for_video(generated_walk, generated_crossing_indices)

sphere_video_cumulative_extent = np.maximum.accumulate(
    np.max(
        np.abs(sphere_video_walk - sphere_geometry_center),
        axis=1,
    )
)
generated_video_cumulative_extent = np.maximum.accumulate(
    np.max(
        np.abs(generated_video_walk - generated_geometry_center),
        axis=1,
    )
)
sphere_initial_video_limits = centered_cube_limits(
    sphere_geometry_center,
    sphere_base_half_extent,
)
generated_initial_video_limits = centered_cube_limits(
    generated_geometry_center,
    generated_base_half_extent,
)

video_figure, (video_axis_sphere, video_axis_generated) = plt.subplots(
    1,
    2,
    figsize=(14.0, 6.85),
    subplot_kw={"projection": "3d"},
)
draw_sphere_surface(video_axis_sphere, coarse=True)
draw_generated_surface(video_axis_generated, coarse=True)
configure_3d_axis(video_axis_sphere, sphere_initial_video_limits)
configure_3d_axis(
    video_axis_generated,
    generated_initial_video_limits,
    azimuth=generated_view_azimuth,
)
video_figure.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=0.125,
    top=1.0,
    wspace=0.015,
)

sphere_axis_position = video_axis_sphere.get_position()
generated_axis_position = video_axis_generated.get_position()
sphere_axis_center = 0.5 * (
    sphere_axis_position.x0 + sphere_axis_position.x1
)
generated_axis_center = 0.5 * (
    generated_axis_position.x0 + generated_axis_position.x1
)
counter_style = {
    "ha": "center",
    "va": "center",
    "fontsize": 11.0,
    "family": "monospace",
    "color": "#303030",
}
sphere_counter = video_figure.text(
    sphere_axis_center,
    0.052,
    "",
    **counter_style,
)
generated_counter = video_figure.text(
    generated_axis_center,
    0.052,
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
    segment_endpoints = np.concatenate(
        ([0], crossing_indices, [len(trajectory) - 1])
    ).astype(np.int64)
    line_artists = []
    for segment_index in range(len(segment_endpoints) - 1):
        line, = ax.plot(
            [],
            [],
            [],
            color=walk_color(segment_index),
            linewidth=0.52,
            alpha=0.78,
        )
        line_artists.append(line)

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        s=25,
        color=start_marker_color,
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
    )
    crossing_artist = ax.scatter(
        [],
        [],
        [],
        s=20,
        edgecolors="white",
        linewidths=0.35,
        depthshade=False,
    )
    crossing_colors = [
        walk_color(index + 1) for index in range(len(crossing_points))
    ]
    current_position_artist = ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        s=34,
        color=walk_color(0),
        edgecolors="white",
        linewidths=0.75,
        depthshade=False,
    )

    def update(progress):
        current_index = trajectory_index_at_progress(
            trajectory_times,
            progress,
        )

        for segment_index, line in enumerate(line_artists):
            segment_start = int(segment_endpoints[segment_index])
            segment_stop = min(
                current_index,
                int(segment_endpoints[segment_index + 1]),
            )
            if segment_stop < segment_start:
                line.set_data_3d([], [], [])
            else:
                line.set_data_3d(
                    trajectory[segment_start : segment_stop + 1, 0],
                    trajectory[segment_start : segment_stop + 1, 1],
                    trajectory[segment_start : segment_stop + 1, 2],
                )

        visible_crossings = int(np.searchsorted(
            crossing_indices,
            current_index,
            side="right",
        ))
        if visible_crossings:
            visible_points = crossing_points[:visible_crossings]
            crossing_artist._offsets3d = (
                visible_points[:, 0],
                visible_points[:, 1],
                visible_points[:, 2],
            )
            crossing_artist.set_facecolors(
                crossing_colors[:visible_crossings]
            )
        else:
            empty = np.empty(0)
            crossing_artist._offsets3d = (empty, empty, empty)

        current_segment = int(np.searchsorted(
            crossing_indices,
            current_index,
            side="right",
        ))
        current_point = trajectory[current_index]
        current_position_artist._offsets3d = (
            [current_point[0]],
            [current_point[1]],
            [current_point[2]],
        )
        current_position_artist.set_facecolors([walk_color(current_segment)])

        return [*line_artists, crossing_artist, current_position_artist]

    return update


update_sphere_video = make_animated_walk(
    video_axis_sphere,
    sphere_video_walk,
    sphere_video_crossing_indices,
    sphere_crossings,
    sphere_video_times,
)
update_generated_video = make_animated_walk(
    video_axis_generated,
    generated_video_walk,
    generated_video_crossing_indices,
    generated_crossings,
    generated_video_times,
)


def update_video(frame_number):
    progress = frame_number / (video_frame_count - 1)
    elapsed_steps = min(n_steps, int(round(progress * n_steps)))
    sphere_current_index = trajectory_index_at_progress(
        sphere_video_times,
        progress,
    )
    generated_current_index = trajectory_index_at_progress(
        generated_video_times,
        progress,
    )
    sphere_video_half_extent = dynamic_video_half_extent(
        sphere_video_cumulative_extent,
        sphere_current_index,
        sphere_base_half_extent,
    )
    generated_video_half_extent = dynamic_video_half_extent(
        generated_video_cumulative_extent,
        generated_current_index,
        generated_base_half_extent,
    )
    set_centered_cube_window(
        video_axis_sphere,
        sphere_geometry_center,
        sphere_video_half_extent,
    )
    set_centered_cube_window(
        video_axis_generated,
        generated_geometry_center,
        generated_video_half_extent,
    )
    sphere_holding_count = int(np.searchsorted(
        sphere_holding_steps,
        elapsed_steps,
        side="right",
    ))
    generated_holding_count = int(np.searchsorted(
        generated_holding_steps,
        elapsed_steps,
        side="right",
    ))

    formatted_steps = f"{elapsed_steps:,}"
    formatted_total = f"{n_steps:,}"
    sphere_counter.set_text(
        f"steps: {formatted_steps} / {formatted_total}\n"
        f"rejected boundary attempts: {sphere_holding_count:,}"
    )
    generated_counter.set_text(
        f"steps: {formatted_steps} / {formatted_total}\n"
        f"rejected boundary attempts: {generated_holding_count:,}"
    )

    return [
        *update_sphere_video(progress),
        *update_generated_video(progress),
        sphere_counter,
        generated_counter,
    ]


movie = animation.FuncAnimation(
    video_figure,
    update_video,
    frames=video_frame_count,
    interval=1_000 / video_frames_per_second,
    blit=False,
)
video_path = Path(__file__).with_name(f"snob_two_regions_3d_{run_id}.mp4")
video_writer = animation.FFMpegWriter(
    fps=video_frames_per_second,
    codec="libx264",
    bitrate=3_600,
    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
)
movie.save(video_path, writer=video_writer, dpi=120)
plt.close(video_figure)
print(f"video saved to: {video_path}")
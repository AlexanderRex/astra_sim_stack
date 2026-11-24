"""Metric localization of the Galbot workbench and known service objects from RGB."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .galbot import GalbotCalibration, read_sensors


TABLE_SIZE = (1.80, 1.20)
CHAMBER_SIZE = (.70, .722, .414)
CHAMBER_BOTTOM_TO_ORIGIN = .23174


@dataclass(frozen=True)
class Detection:
    name: str
    base_xyz_m: tuple[float, float, float]
    yaw_rad: float | None
    yaw_period_rad: float | None
    score: float
    camera: str
    outline_px: list


def _yaw(vector, period):
    return float(math.atan2(vector[1], vector[0]) % period)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _axes(yaw):
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    return np.array([forward, [-forward[1], forward[0]]])


def _inside_image(points, camera, border=0):
    return (np.isfinite(points).all(axis=-1)
            & (points[..., 0] >= border) & (points[..., 0] < camera.width - border)
            & (points[..., 1] >= border) & (points[..., 1] < camera.height - border))


def _mask_at(mask, points, camera):
    valid = _inside_image(points, camera, border=1)
    px = np.rint(points[valid]).astype(int)
    result = np.zeros(points.shape[:-1], dtype=bool)
    result[valid] = mask[px[:, 1], px[:, 0]]
    return result


def _rect(center, axes, length, width):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return center + (corners * [length / 2, width / 2]) @ axes


def _table_masks(rgb):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    bright = (np.minimum(np.minimum(red, green), blue) > 217) & (np.abs(blue - red) < 21)
    blue_floor = (blue - red > 16) & (blue - green > 6) & (red < 175)
    return bright, blue_floor


def _table_edges(rgb, camera, height, bright, floor):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detected = cv2.createLineSegmentDetector().detect(gray)[0]
    if detected is None:
        return []
    edges = []
    for line in detected.reshape(-1, 2, 2):
        pixel_dir = line[1] - line[0]
        if np.linalg.norm(pixel_dir) < 22:
            continue
        normal = _unit(np.array([-pixel_dir[1], pixel_dir[0]]))
        samples = line[0] + np.linspace(.12, .88, 9)[:, None] * pixel_dir
        scores = []
        for sign in (1, -1):
            inside = _mask_at(bright, samples + sign * 4 * normal, camera)
            outside = (_mask_at(floor, samples - sign * 4 * normal, camera)
                       | _mask_at(floor, samples - sign * 8 * normal, camera))
            scores.append(np.mean(inside & outside))
        if max(scores) < .50:
            continue
        xy = camera.pixels_on_plane(line, height)[:, :2]
        nudge = normal if scores[0] >= scores[1] else -normal
        inward = camera.pixels_on_plane([line.mean(axis=0) + 5 * nudge], height)[0, :2] - xy.mean(axis=0)
        if not np.isfinite(xy).all() or not np.isfinite(inward).all():
            continue
        length = np.linalg.norm(xy[1] - xy[0])
        if not .19 <= length <= 2.1:
            continue
        edges.append((xy, _unit(xy[1] - xy[0]), inward, length * max(scores)))
    return sorted(edges, key=lambda edge: edge[-1], reverse=True)[:16]


def _edge_support(corners, camera, height, bright, floor):
    other = np.roll(corners, -1, axis=0)
    samples = corners[:, None, :] + (other - corners)[:, None, :] * np.linspace(.055, .945, 27)[None, :, None]
    center = corners.mean(axis=0)
    inward = center - (corners + other) / 2
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    xy = np.stack([samples + .035 * inward[:, None, :], samples - .035 * inward[:, None, :]])
    uv = camera.project(np.c_[xy.reshape(-1, 2), np.full(xy.size // 2, height)]).reshape(xy.shape)
    return np.mean(_mask_at(bright, uv[0], camera) & _mask_at(floor, uv[1], camera), axis=1)


def _table(rgb, camera, height):
    bright, floor = _table_masks(rgb)
    edges = _table_edges(rgb, camera, height, bright, floor)
    best = None
    grid = np.stack(np.meshgrid(np.linspace(-.78, .78, 11), np.linspace(-.49, .49, 9)), axis=-1).reshape(-1, 2)

    def consider(center, long_direction, opposite=False):
        nonlocal best
        yaw = _yaw(long_direction, math.pi)
        corners = _rect(center, _axes(yaw), *TABLE_SIZE)
        supports = _edge_support(corners, camera, height, bright, floor)
        pair = np.roll(supports, 2 if opposite else 1)
        edge_score = float(np.mean(supports))
        if np.max(np.minimum(supports, pair)) < .16 or edge_score < .20:
            return
        uv_interior = camera.project(np.c_[center + grid @ _axes(yaw), np.full(len(grid), height)])
        visible = _inside_image(uv_interior, camera)
        if visible.sum() < 18:
            return
        background = np.mean(_mask_at(floor, uv_interior[visible], camera))
        if background > .20:
            return
        white = np.mean(_mask_at(bright, uv_interior[visible], camera))
        score = float(edge_score + .22 * white - .55 * background)
        if best is None or score > best.score:
            uv = camera.project(np.c_[corners, np.full(len(corners), height)])
            if np.isfinite(uv).all():
                best = Detection("table", (*map(float, center), float(height)), yaw, math.pi,
                                 score, camera.name, uv.tolist())

    for i, (xy, direction, inward, _) in enumerate(edges):
        side_normal = np.array([-direction[1], direction[0]])
        side_normal *= math.copysign(1, side_normal @ inward)
        edge_length = np.linalg.norm(xy[1] - xy[0])
        for span, normal_span in (TABLE_SIZE, TABLE_SIZE[::-1]):
            if abs(edge_length - span) < .16:
                center = xy.mean(axis=0) + .5 * normal_span * side_normal
                consider(center, direction if span > normal_span else side_normal)
        for other, other_dir, other_inward, _ in edges[i + 1:]:
            parallel = float(direction @ other_dir)
            if abs(parallel) > .99:
                u = _unit(direction + math.copysign(1, parallel) * other_dir)
                v = np.array([-u[1], u[0]])
                side, opposite_side = np.mean(xy @ v), np.mean(other @ v)
                width = abs(opposite_side - side)
                if (inward @ v) * (other_inward @ v) >= 0 or (inward @ v) * (opposite_side - side) <= 0:
                    continue
                along = np.r_[xy @ u, other @ u]
                for span, separation in (TABLE_SIZE, TABLE_SIZE[::-1]):
                    if abs(width - separation) > .12 or not span - .26 < np.ptp(along) < span + .20:
                        continue
                    middle = (along.min() + along.max()) / 2
                    offsets = np.linspace(-.06, .06, 5)
                    for offset in offsets:
                        center = (middle + offset) * u + (side + opposite_side) / 2 * v
                        consider(center, u if span > separation else v, opposite=True)
                continue
            if abs(parallel) > .24:
                continue
            along1, along2 = np.linalg.solve(np.column_stack([direction, -other_dir]), other[0] - xy[0])
            corner = (xy[0] + along1 * direction + other[0] + along2 * other_dir) / 2
            u = direction * (1 if direction @ other_inward > 0 else -1)
            v = np.array([-u[1], u[0]])
            if v @ inward < 0:
                v = -v
            # Both observed segments must run into the rectangle from the common corner.
            observed_u, observed_v = (xy - corner) @ u, (other - corner) @ v
            if min(observed_u.min(), observed_v.min()) < -.14:
                continue
            for span_u, span_v in (TABLE_SIZE, TABLE_SIZE[::-1]):
                if observed_u.max() > span_u + .19 or observed_v.max() > span_v + .19:
                    continue
                if max(observed_u.min(), observed_v.min()) > .75:
                    continue
                center = corner + .5 * (span_u * u + span_v * v)
                consider(center, u if span_u > span_v else v)
    return best


def _silhouette(camera, center, axes, plane, size):
    corners = _rect(center, axes, *size[:2])
    xyz = np.c_[np.tile(corners, (2, 1)), np.repeat([plane, plane + size[2]], 4)]
    uv = camera.project(xyz)
    if not np.isfinite(uv).all():
        return None
    return cv2.convexHull(np.rint(uv).astype(np.int32)).reshape(-1, 2)


def _silhouette_score(mask, edge_distance, hull):
    if hull is None:
        return 0.
    height, width = edge_distance.shape
    x, y, w, h = cv2.boundingRect(hull)
    margin_x, margin_y = round(.04 * width), round(.03 * height)
    if (w < 20 or h < 18 or w > width * .75 or h > height * .60
            or x < margin_x or y < margin_y or x + w >= width - margin_x or y + h >= height - margin_y):
        return 0.
    region = mask[max(y - 6, 0):min(y + h + 6, height), max(x - 6, 0):min(x + w + 6, width)]
    model = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(model, hull - [max(x - 6, 0), max(y - 6, 0)], 1)
    inside = model.astype(bool)
    if inside.sum() < 360:
        return 0.
    outside = (cv2.dilate(model, np.ones((9, 9), np.uint8)) > 0) & ~inside
    fill = np.mean(region[inside], axis=0)
    leak = np.mean(region[outside], axis=0)
    color_score = float(np.max(fill - .65 * leak))
    boundary_score = _line_support(edge_distance, hull)
    return float(.65 * color_score + .35 * boundary_score)


def _line_support(edge_distance, poly):
    vectors = np.roll(poly, -1, axis=0) - poly
    outline = poly[:, None, :] + vectors[:, None, :] * np.linspace(.07, .93, 16)[None, :, None]
    px = np.rint(outline).astype(int)
    support = np.exp(-edge_distance[px[..., 1], px[..., 0]] / 3.0).mean(axis=1)
    return float(np.average(support, weights=np.linalg.norm(vectors, axis=1)))


def _roof_material_support(mask, polygon):
    # Stay inside the projected roof, away from the bevel and raised handle.
    u, v = np.meshgrid(np.linspace(.15, .85, 7), np.linspace(.15, .85, 7))
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=-1)
    x, y = np.rint(weights @ polygon).astype(int).reshape(-1, 2).T
    return float(mask[y, x].mean())


def _chamber(rgb, camera, table):
    pixels = rgb.astype(np.int16)
    chroma = np.ptp(pixels, axis=2)
    value = np.mean(pixels, axis=2)
    neutral = (chroma < 27) & (value < 233)
    roof_material = (chroma < 26) & (value >= 140) & (value < 237)
    # A dark panel may have the same luminance as the surroundings. Use two
    # material hypotheses and require the image edges to support the projection.
    mask = np.stack([neutral & (value >= 24), neutral & (value >= 130)], axis=-1).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 65) | cv2.Canny(chroma.astype(np.uint8), 14, 40)
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    axes = _axes(table.yaw_rad)
    origin = np.array(table.base_xyz_m[:2])
    plane = table.base_xyz_m[2]
    roof = plane + CHAMBER_SIZE[2]
    dx, dy = np.array(TABLE_SIZE) / 2 - np.array(CHAMBER_SIZE[:2]) / 2 + .085
    best = (0., None, None, None)

    def consider(center, yaw):
        nonlocal best
        frame = _axes(yaw)
        hull = _silhouette(camera, center, frame, plane, CHAMBER_SIZE)
        score = _silhouette_score(mask, edge_distance, hull)
        if score < .36:
            return
        roof_poly = camera.project(np.c_[_rect(center, frame, *CHAMBER_SIZE[:2]), np.full(4, roof)])
        if not _inside_image(roof_poly, camera, border=2).all():
            return
        if _roof_material_support(roof_material, roof_poly) < .80:
            return
        score = .77 * score + .23 * _line_support(edge_distance, roof_poly)
        if score > best[0]:
            best = score, center, yaw, hull

    # Search the footprint and seed additional poses from visible roof edges.
    for x in np.arange(-dx, dx + .02, .045):
        for y in np.arange(-dy, dy + .02, .045):
            consider(origin + np.array([x, y]) @ axes, table.yaw_rad)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is not None:
        for line in lines.reshape(-1, 2, 2):
            if np.linalg.norm(line[1] - line[0]) < 25:
                continue
            metric = camera.pixels_on_plane(line, roof)[:, :2]
            length = np.linalg.norm(metric[1] - metric[0])
            if not .39 < length < .88:
                continue
            direction = (metric[1] - metric[0]) / length
            normal = np.array([-direction[1], direction[0]])
            yaw = _yaw(direction, math.pi / 2)
            for sign in (-1, 1):
                mid = metric.mean(axis=0) + sign * .35 * normal
                for offset in (-.09, 0, .09):
                    center = mid + offset * direction
                    if np.linalg.norm(center - origin) < 1.04:
                        consider(center, yaw)
    if best[0] < .46:
        return None
    _, initial_center, initial_yaw, _ = best
    for dx in (-.045, -.0225, 0., .0225, .045):
        for dy in (-.045, -.0225, 0., .0225, .045):
            center = initial_center + np.array([dx, dy]) @ axes
            for yaw in initial_yaw + np.linspace(-.12, .12, 5):
                consider(center, yaw)
    score, center, yaw, hull = best
    if score < .55:
        return None
    period = math.pi / 2
    return Detection("chamber", (*map(float, center), plane + CHAMBER_BOTTOM_TO_ORIGIN),
                     float(yaw % period), period, score, camera.name, hull.tolist())


def _sponge(rgb, camera, table):
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    mask = (green - red > 18) & (blue - red > 12) & (green > 85) & (blue - green < 9)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    plane = table.base_xyz_m[2] + .011
    table_center = np.array(table.base_xyz_m[:2])
    axes = _axes(table.yaw_rad)
    candidate = None
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not 18 <= area <= 1600 or h < 3 or w < 3 or area / (w * h) < .33:
            continue
        xyz = camera.pixels_on_plane([centroids[i]], plane)[0]
        if not np.isfinite(xyz).all() or not np.all(np.abs(axes @ (xyz[:2] - table_center)) < np.array(TABLE_SIZE) / 2):
            continue
        rim = [[int(px), int(py)] for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
        rect = camera.pixels_on_plane(rim, plane)
        extent = np.ptp(rect[:, :2], axis=0)
        if not .045 < np.max(extent) < .20 or np.min(extent) > .14:
            continue
        score = float(area / (w * h))
        if candidate is None or score > candidate.score:
            candidate = Detection("sponge", tuple(map(float, xyz)), None, None, score, camera.name, rim)
    return candidate


class WorkbenchLocalizer:
    """Recognize the table and service objects in Galbot base coordinates.

    Uses known physical dimensions and work-surface height from setup calibration.
    A target is omitted when the available image evidence is too weak.
    """

    def __init__(self, calibration=None, table_height_m=.72):
        self.calibration = GalbotCalibration() if calibration is None else calibration
        self.table_height_m = float(table_height_m)
        if not math.isfinite(self.table_height_m) or self.table_height_m <= 0:
            raise ValueError("Table height in the robot base frame must be positive and finite")

    def locate(self, images, encoder_positions):
        cameras = self.calibration.cameras(encoder_positions)
        for name, rgb in images.items():
            camera = cameras[name]
            if rgb.shape != (camera.height, camera.width, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid {name} RGB frame shape or dtype")
        tables = [table for name, rgb in images.items()
                  if (table := _table(rgb, cameras[name], self.table_height_m)) is not None]
        if not tables:
            return {}
        table = max(tables, key=lambda item: item.score)
        result = {"table": table}
        for detector in (_chamber, _sponge):
            candidates = [hit for name, rgb in images.items()
                          if (hit := detector(rgb, cameras[name], table)) is not None]
            if candidates:
                hit = max(candidates, key=lambda item: item.score)
                result[hit.name] = hit
        return result

    def from_observation(self, observation):
        return self.locate(*read_sensors(observation))

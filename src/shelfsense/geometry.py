"""Small dependency-free linear algebra and planar shelf coordinates."""

from math import isfinite, sqrt


def solve(matrix, rhs):
    """Solve a square system with partial pivoting; reject degenerate geometry."""
    n = len(rhs)
    rows = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs, strict=True)]
    if not n or any(len(row) != n + 1 for row in rows):
        raise ValueError("Expected a square linear system")
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(rows[i][col]))
        if abs(rows[pivot][col]) < 1e-10:
            raise ValueError("Degenerate or ill-conditioned calibration")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        factor = rows[col][col]
        rows[col] = [value / factor for value in rows[col]]
        for i in range(n):
            if i != col:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[col])]
    return [row[-1] for row in rows]


def least_squares(matrix, rhs):
    """QR least squares with reorthogonalization, avoiding normal equations."""
    columns = list(zip(*matrix))
    n = len(columns)
    if len(matrix) < n or not n:
        raise ValueError("Not enough independent calibration samples")
    q = []
    r = [[0.0] * n for _ in range(n)]
    for j, column in enumerate(columns):
        residual = list(column)
        for _ in range(2):
            for i, basis in enumerate(q):
                projection = sum(a * b for a, b in zip(basis, residual))
                r[i][j] += projection
                residual = [a - projection * b for a, b in zip(residual, basis)]
        norm = sqrt(sum(value * value for value in residual))
        if norm < 1e-8:
            raise ValueError("Calibration features are dependent; collect more varied samples")
        r[j][j] = norm
        q.append([value / norm for value in residual])
    return solve(r, [sum(a * b for a, b in zip(basis, rhs)) for basis in q])


def finite_vector(value, length, label):
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                   for v in value)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return tuple(float(v) for v in value)


class ShelfGeometry:
    """Map shelf pixels to [0,1]²; corners ordered TL, TR, BR, BL."""

    def __init__(self, image_size, corners):
        width, height = finite_vector(image_size, 2, "image_size")
        if width <= 0 or height <= 0 or width != int(width) or height != int(height):
            raise ValueError("image_size must be positive integer [width, height]")
        self.image_size = (int(width), int(height))
        if not isinstance(corners, (list, tuple)) or len(corners) != 4:
            raise ValueError("Shelf requires four ordered corners")
        self.corners = tuple(finite_vector(c, 2, "corner") for c in corners)
        points = [(x / width, y / height) for x, y in self.corners]
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
            raise ValueError("Shelf corners must lie inside the calibrated image")
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross <= 1e-8:
                raise ValueError("Corners must form a convex clockwise TL, TR, BR, BL polygon")
        matrix, rhs = [], []
        for (x, y), (u, v) in zip(points, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            matrix.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y],
                           [0, 0, 0, x, y, 1, -v*x, -v*y]])
            rhs.extend([u, v])
        self.h = solve(matrix, rhs)

    def to_shelf(self, point):
        x, y = finite_vector(point, 2, "target_image")
        x, y = x / self.image_size[0], y / self.image_size[1]
        a, b, c, d, e, f, g, h = self.h
        denominator = g*x + h*y + 1
        if abs(denominator) < 1e-10:
            raise ValueError("Point projects to infinity")
        return ((a*x + b*y + c) / denominator, (d*x + e*y + f) / denominator)


def validate_zones(zones):
    if not isinstance(zones, list) or not zones:
        raise ValueError("At least one shelf zone is required")
    result, ids = [], set()
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) != {"id", "bounds"}:
            raise ValueError("Each zone requires id and bounds")
        name = zone["id"]
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Zone IDs must be nonempty and unique")
        x1, y1, x2, y2 = finite_vector(zone["bounds"], 4, "zone bounds")
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Zone bounds must have positive area within [0,1]")
        for other in result:
            a, b, c, d = other["bounds"]
            if max(a, x1) < min(c, x2) and max(b, y1) < min(d, y2):
                raise ValueError("Shelf zones cannot overlap")
        result.append({"id": name, "bounds": (x1, y1, x2, y2)})
        ids.add(name)
    return result


def zone_at(point, zones):
    """Half-open edges; include the outer shelf boundary at exactly 1."""
    x, y = point
    for zone in zones:
        x1, y1, x2, y2 = zone["bounds"]
        if (x1 <= x and (x < x2 or x == x2 == 1)
                and y1 <= y and (y < y2 or y == y2 == 1)):
            return zone["id"]
    return None

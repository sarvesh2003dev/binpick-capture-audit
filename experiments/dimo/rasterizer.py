"""Small NumPy-only CAD rasterizer for BOP-format camera/object poses.

Distances are kept in input mesh/pose units (millimetres for DIMO). The depth
output is camera-axis Z, NOT Euclidean range and NOT measured sensor depth.
Masks are CAD projections derived from supplied poses, not manual annotations.
Pixel samples are at (column + .5, row + .5); no backface culling is performed.
Triangles crossing/behind the near plane are skipped, not clipped. DIMO objects
are wholly in front of the camera, so this restriction should not affect them.
"""

from pathlib import Path
import numpy as np


def read_ply(path):
    """Read ASCII PLY vertices and triangulate convex polygon faces.

    Extra scalar/list properties and elements are consumed and ignored. Only
    x/y/z and face vertex_indices (or vertex_index) are required. Binary PLY
    raises a clear error: the DIMO CAD models used here are ASCII PLY.
    """
    elements = []
    with Path(path).open("r", encoding="ascii") as stream:
        if stream.readline().strip() != "ply":
            raise ValueError("Not a PLY file")
        file_format = None
        for line in stream:
            words = line.strip().split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "format":
                file_format = words[1]
            elif words[0] == "element":
                elements.append((words[1], int(words[2]), []))
            elif words[0] == "property":
                if not elements:
                    raise ValueError("PLY property precedes element")
                elements[-1][2].append((words[-1], words[1] == "list"))
            elif words[0] == "end_header":
                break
        else:
            raise ValueError("Missing PLY end_header")
        if file_format != "ascii":
            raise ValueError(f"Only ASCII PLY supported, got {file_format!r}")
        vertices, faces = [], []
        for name, count, properties in elements:
            for _ in range(count):
                line = stream.readline()
                if not line:
                    raise ValueError("Truncated PLY element data")
                words, cursor, values = line.split(), 0, {}
                for prop_name, is_list in properties:
                    if cursor >= len(words):
                        raise ValueError("Truncated PLY property data")
                    if is_list:
                        length = int(words[cursor])
                        cursor += 1
                        if length < 0 or cursor + length > len(words):
                            raise ValueError("Invalid PLY list length")
                        values[prop_name] = words[cursor:cursor + length]
                        cursor += length
                    else:
                        values[prop_name] = words[cursor]
                        cursor += 1
                if name == "vertex":
                    vertices.append([float(values[c]) for c in ("x", "y", "z")])
                elif name == "face":
                    indices = values.get("vertex_indices", values.get("vertex_index"))
                    if indices is None:
                        raise ValueError("Face lacks vertex_indices property")
                    indices = [int(value) for value in indices]
                    if len(indices) < 3:
                        raise ValueError("Face has fewer than three vertices")
                    faces.extend((indices[0], indices[i], indices[i + 1])
                                 for i in range(1, len(indices) - 1))
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains nonfinite coordinates")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Face vertex index outside mesh")
    return vertices, faces


def render_scene(models, objects, K, width, height):
    """Return (depth_Z, instance_ids), shaped [height,width].

    models: {obj_id: (Nx3 vertices, Mx3 triangle indices)}.
    objects: BOP dictionaries containing obj_id, cam_R_m2c, cam_t_m2c.
    Instance IDs are one-based indices into objects, including carrier models;
    zero denotes background. Background depth is zero. No far-plane clipping.
    Equal-depth ties retain the first submitted instance. A full scene including
    carrier geometry is needed for correct CAD-to-CAD occlusion.
    """
    if width <= 0 or height <= 0 or int(width) != width or int(height) != height:
        raise ValueError("width and height must be positive integers")
    width, height = int(width), int(height)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("K must be a finite pinhole intrinsic matrix")
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    ids = np.zeros((height, width), dtype=np.int32)
    near = 1e-6
    for instance_id, obj in enumerate(objects, start=1):
        vertices, faces = models[int(obj["obj_id"])]
        vertices = np.asarray(vertices, dtype=np.float64)
        rotation = np.asarray(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(obj["cam_t_m2c"], dtype=np.float64).reshape(3)
        camera_vertices = vertices @ rotation.T + translation
        if not np.isfinite(camera_vertices).all():
            raise ValueError("Nonfinite camera-space coordinates")
        projected = camera_vertices @ K.T
        for face in np.asarray(faces, dtype=np.int64):
            triangle = camera_vertices[face]
            z = triangle[:, 2]
            if np.any(z <= near):
                continue
            xy = projected[face, :2] / z[:, None]
            x0, y0 = xy[0]
            x1, y1 = xy[1]
            x2, y2 = xy[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-12:
                continue
            xmin = max(0, int(np.floor(xy[:, 0].min())))
            xmax = min(width - 1, int(np.ceil(xy[:, 0].max())))
            ymin = max(0, int(np.floor(xy[:, 1].min())))
            ymax = min(height - 1, int(np.ceil(xy[:, 1].max())))
            if xmin > xmax or ymin > ymax:
                continue
            xx = np.arange(xmin, xmax + 1, dtype=np.float64)[None, :] + .5
            yy = np.arange(ymin, ymax + 1, dtype=np.float64)[:, None] + .5
            a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
            b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
            c = 1.0 - a - b
            inside = (a >= -1e-10) & (b >= -1e-10) & (c >= -1e-10)
            # Linear interpolation of Z would be wrong for tilted triangles.
            inverse_z = a / z[0] + b / z[1] + c / z[2]
            with np.errstate(divide="ignore", invalid="ignore"):
                sample_z = 1.0 / inverse_z
            region_z = zbuffer[ymin:ymax + 1, xmin:xmax + 1]
            region_ids = ids[ymin:ymax + 1, xmin:xmax + 1]
            update = inside & (inverse_z > 0) & (sample_z < region_z)
            region_z[update] = sample_z[update]
            region_ids[update] = instance_id
    depth = np.where(ids != 0, zbuffer, 0).astype(np.float32)
    return depth, ids

"""
File-based tissue point loading.

Supports CSV, NPY, PLY, and OBJ file formats.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, List
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_tissue_points_from_file(
    file_path: str,
    file_format: str = "auto",
    columns: Optional[List[str]] = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Load tissue points from a file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    file_format : str
        File format: "auto", "csv", "npy", "ply", "obj".
    columns : list of str, optional
        Column names for CSV files.
    scale : float
        Scale factor to apply to coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with tissue point coordinates.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tissue point file not found: {file_path}")

    if file_format == "auto":
        ext = path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".tsv": "csv",
            ".txt": "csv",
            ".npy": "npy",
            ".npz": "npy",
            ".ply": "ply",
            ".obj": "obj",
        }
        file_format = format_map.get(ext, "csv")

    loaders = {
        "csv": _load_csv,
        "npy": _load_npy,
        "ply": _load_ply,
        "obj": _load_obj,
    }

    loader = loaders.get(file_format)
    if loader is None:
        raise ValueError(f"Unsupported file format: {file_format}")

    points = loader(file_path, columns)

    if scale != 1.0:
        points = points * scale

    logger.info("Loaded %d tissue points from %s (format=%s, scale=%.4f)",
                len(points), file_path, file_format, scale)

    return points


def _load_csv(file_path: str, columns: Optional[List[str]] = None) -> np.ndarray:
    """Load from CSV/TSV/TXT."""
    path = Path(file_path)
    content = path.read_text().strip()

    if not content:
        return np.empty((0, 3))

    lines = content.split("\n")

    first_line = lines[0].strip()
    has_header = False
    for sep in [",", "\t", " "]:
        parts = first_line.split(sep)
        if len(parts) >= 3:
            try:
                float(parts[0])
            except ValueError:
                has_header = True
            break

    if has_header:
        header = lines[0].strip()
        data_lines = lines[1:]
    else:
        data_lines = lines

    sep = ","
    if "\t" in data_lines[0]:
        sep = "\t"
    elif "," not in data_lines[0] and " " in data_lines[0]:
        sep = None

    points = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        if sep is None:
            parts = line.split()
        else:
            parts = line.split(sep)

        if len(parts) < 3:
            continue

        if columns and has_header:
            header_parts = header.split(sep) if sep else header.split()
            col_map = {name.strip(): idx for idx, name in enumerate(header_parts)}
            try:
                x = float(parts[col_map.get(columns[0], 0)])
                y = float(parts[col_map.get(columns[1], 1)])
                z = float(parts[col_map.get(columns[2], 2)])
            except (KeyError, IndexError, ValueError):
                continue
        else:
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                z = float(parts[2].strip())
            except ValueError:
                continue

        points.append([x, y, z])

    return np.array(points, dtype=np.float64) if points else np.empty((0, 3))


def _load_npy(file_path: str, columns: Optional[List[str]] = None) -> np.ndarray:
    """Load from NPY/NPZ."""
    path = Path(file_path)
    if path.suffix.lower() == ".npz":
        data = np.load(file_path)
        if columns and columns[0] in data:
            return data[columns[0]].astype(np.float64)
        key = list(data.keys())[0]
        return data[key].astype(np.float64)
    else:
        data = np.load(file_path)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3].astype(np.float64)
        return data.astype(np.float64)


def _load_ply(file_path: str, columns: Optional[List[str]] = None) -> np.ndarray:
    """Load vertex positions from PLY file."""
    path = Path(file_path)
    content = path.read_bytes()

    is_binary = b"format binary" in content[:200]
    lines = content.split(b"\n")

    n_vertices = 0
    header_end = 0
    x_idx, y_idx, z_idx = 0, 1, 2
    prop_idx = 0
    in_vertex = False

    for i, line in enumerate(lines):
        line_str = line.decode("ascii", errors="ignore").strip()
        if line_str.startswith("element vertex"):
            n_vertices = int(line_str.split()[-1])
            in_vertex = True
            prop_idx = 0
        elif line_str.startswith("element") and in_vertex:
            in_vertex = False
        elif line_str.startswith("property") and in_vertex:
            parts = line_str.split()
            if len(parts) >= 3:
                name = parts[-1]
                if name == "x":
                    x_idx = prop_idx
                elif name == "y":
                    y_idx = prop_idx
                elif name == "z":
                    z_idx = prop_idx
            prop_idx += 1
        elif line_str == "end_header":
            header_end = i + 1
            break

    if is_binary:
        logger.warning("Binary PLY not fully supported; falling back to text parsing")
        return np.empty((0, 3))

    points = []
    for i in range(header_end, min(header_end + n_vertices, len(lines))):
        line_str = lines[i].decode("ascii", errors="ignore").strip()
        if not line_str:
            continue
        parts = line_str.split()
        if len(parts) > max(x_idx, y_idx, z_idx):
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                points.append([x, y, z])
            except ValueError:
                continue

    return np.array(points, dtype=np.float64) if points else np.empty((0, 3))


def _load_obj(file_path: str, columns: Optional[List[str]] = None) -> np.ndarray:
    """Load vertex positions from OBJ file."""
    points = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        points.append([x, y, z])
                    except ValueError:
                        continue

    return np.array(points, dtype=np.float64) if points else np.empty((0, 3))

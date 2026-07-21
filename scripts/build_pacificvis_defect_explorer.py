"""Build a compact browser explorer for the PacificVis 8x8x8 defect dataset.

The source CT is 6.9 GB and the OBJ meshes total more than 2.6 million faces.
This builder memory-maps/samples those assets into a small, deterministic payload
that remains responsive in a WebGL viewer.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/PacificVis Datasets/8x8x8 octet lattice with defects"
CT_PATH = DATASET / "five_defects_1200_xray_recon.npy"
REFERENCE_PATH = DATASET / "octet_truss_8x8x8_no_defects.obj"
DEFECT_PATH = DATASET / "octet_truss_8x8x8_with_defects.obj"
TEMPLATE = ROOT / "scripts/templates/pacificvis-defect-explorer.fragment.html"
OUT = (
    ROOT
    / ".codex/visualizations/2026/07/21/pacificvis-defects"
    / "octet-defect-explorer.html"
)

CT_STRIDE = 8
CT_FLOOR = 0.008
CT_LIMIT = 60_000
REFERENCE_LIMIT = 25_000
DEFECT_LIMIT = 40_000


def _bounded_even_sample(array: np.ndarray, limit: int) -> np.ndarray:
    if len(array) <= limit:
        return array
    indices = np.linspace(0, len(array) - 1, limit, dtype=np.int64)
    return array[indices]


def _load_ct_points() -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    volume = np.load(CT_PATH, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, found {volume.shape}")

    sampled = np.asarray(volume[::CT_STRIDE, ::CT_STRIDE, ::CT_STRIDE])
    occupied = np.argwhere(sampled >= CT_FLOOR)
    occupied = _bounded_even_sample(occupied, CT_LIMIT)
    density = sampled[tuple(occupied.T)].astype("<f4")

    # np.argwhere returns z, y, x. WebGL uses x, y, z centered on the origin.
    xyz = occupied[:, ::-1].astype(np.float32) * CT_STRIDE
    scale = np.asarray(volume.shape[::-1], dtype=np.float32) - 1
    xyz = xyz / scale - 0.5
    packed_xyz = np.rint(xyz * 32767).astype("<i2")
    return packed_xyz, density, tuple(int(value) for value in volume.shape)


def _load_obj_points(path: Path, limit: int) -> tuple[np.ndarray, int]:
    vertices: list[tuple[float, float, float]] = []
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            count += 1
            fields = line.split()
            if len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))

    xyz = _bounded_even_sample(np.asarray(vertices, dtype=np.float32), limit)
    # Both supplied models share approximately [-0.1, 16.1] bounds.
    xyz = (xyz - 8.0) / 16.2
    return np.rint(xyz * 32767).astype("<i2"), count


def _payload(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode("ascii")


def main() -> None:
    for path in (CT_PATH, REFERENCE_PATH, DEFECT_PATH, TEMPLATE):
        if not path.is_file():
            raise FileNotFoundError(path)

    ct_xyz, ct_density, shape = _load_ct_points()
    reference_xyz, reference_total = _load_obj_points(REFERENCE_PATH, REFERENCE_LIMIT)
    defect_xyz, defect_total = _load_obj_points(DEFECT_PATH, DEFECT_LIMIT)

    replacements = {
        "__CT_XYZ__": _payload(ct_xyz),
        "__CT_DENSITY__": _payload(ct_density),
        "__REFERENCE_XYZ__": _payload(reference_xyz),
        "__DEFECT_XYZ__": _payload(defect_xyz),
        "__CT_COUNT__": f"{len(ct_xyz):,}",
        "__REFERENCE_COUNT__": f"{len(reference_xyz):,}",
        "__DEFECT_COUNT__": f"{len(defect_xyz):,}",
        "__REFERENCE_TOTAL__": f"{reference_total:,}",
        "__DEFECT_TOTAL__": f"{defect_total:,}",
        "__VOLUME_SHAPE__": " x ".join(str(value) for value in shape),
    }
    fragment = TEMPLATE.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        fragment = fragment.replace(marker, value)
    unresolved = [part for part in replacements if part in fragment]
    if unresolved:
        raise RuntimeError(f"Unresolved template markers: {unresolved}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fragment, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(
        f"CT {shape}: {len(ct_xyz):,} samples; "
        f"reference {len(reference_xyz):,}/{reference_total:,} vertices; "
        f"defective {len(defect_xyz):,}/{defect_total:,} vertices"
    )


if __name__ == "__main__":
    main()

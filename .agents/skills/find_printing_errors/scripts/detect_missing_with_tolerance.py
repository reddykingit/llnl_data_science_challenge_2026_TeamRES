#!/usr/bin/env python3
"""Detect missing material in scanned volume relative to ideal geometry,
with a configurable spatial tolerance (distance radius) for matching nearby voxels.
"""

import argparse
import numpy as np
import tifffile as tiff
from pathlib import Path
from scipy.ndimage import binary_dilation

def make_spherical_structuring_element(radius: float) -> np.ndarray:
    """Create a 3D boolean spherical structuring element for binary dilation."""
    r = int(np.ceil(radius))
    z, y, x = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (x**2 + y**2 + z**2) <= radius**2

def detect_missing_with_tolerance(ideal_path: str, scan_path: str, output_path: str, radius: float):
    print(f"Loading ideal volume: {ideal_path}")
    ideal = tiff.imread(ideal_path) > 0

    print(f"Loading scan volume: {scan_path}")
    scan = tiff.imread(scan_path) > 0

    if ideal.shape != scan.shape:
        raise ValueError(f"Shape mismatch: ideal shape {ideal.shape} != scan shape {scan.shape}")

    if radius > 0:
        print(f"Applying spatial tolerance (radius = {radius} voxels)...")
        struct_elem = make_spherical_structuring_element(radius)
        scan_dilated = binary_dilation(scan, structure=struct_elem)
    else:
        print("Radius is 0: Performing exact voxel matching...")
        scan_dilated = scan

    print("Computing missing material (ideal AND NOT scan_dilated)...")
    missing = (ideal & (~scan_dilated)).astype(np.uint8) * 255

    num_missing_voxels = np.count_nonzero(missing)
    num_ideal_voxels = np.count_nonzero(ideal)
    print(f"Total ideal voxels: {num_ideal_voxels:,}")
    print(f"Missing voxels detected: {num_missing_voxels:,} ({num_missing_voxels / max(1, num_ideal_voxels) * 100:.2f}%)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving missing material volume to {output_path}...")
    tiff.imwrite(output_path, missing)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Detect missing material with neighborhood distance threshold.")
    parser.add_argument("-i", "--ideal", required=True, help="Path to ideal rasterized TIFF")
    parser.add_argument("-s", "--scan", required=True, help="Path to scanned segmentation TIFF")
    parser.add_argument("-o", "--output", default="missing_with_tolerance.tif", help="Output TIFF path")
    parser.add_argument("-r", "--radius", type=float, default=4.0, help="Tolerance search radius in voxels (default: 2.0)")

    args = parser.parse_args()
    detect_missing_with_tolerance(args.ideal, args.scan, args.output, args.radius)

if __name__ == "__main__":
    main()

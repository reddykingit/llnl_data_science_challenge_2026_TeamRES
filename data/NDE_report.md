# Non-Destructive Evaluation Report

## Dataset

This report analyzes the compatible NumPy volume artifacts found under `data/unitcell`:

- Original volume: `unitcell.npy`
- Segmentation mask: `unitcell_mask.npy` (intensity threshold `>= 0.01`)
- Skeleton: `unitcell_skeleton.npy`

All three arrays have shape `256 × 256 × 256` and are spatially compatible. No voxel spacing metadata was supplied, so volumes and lengths are reported in voxel units.

## Summary metrics

| Artifact | Metric | Value |
|---|---:|---:|
| Original volume | Shape | `256 × 256 × 256` |
| Original volume | Total voxels | 16,777,216 |
| Original volume | Mean intensity | 0.000539067 |
| Original volume | Intensity standard deviation | 0.002418240 |
| Original volume | Intensity range | -0.003128750 to 0.015257692 |
| Segmentation mask | Threshold | `>= 0.01` |
| Segmentation mask | Foreground volume | 622,182 voxels |
| Segmentation mask | Foreground fraction | 3.708494% |
| Segmentation mask | Connected components (26-connectivity) | 1 |
| Segmentation mask | Mean foreground intensity | 0.012151320 |
| Segmentation mask | Mean background intensity | 0.000091842 |
| Skeleton | Skeleton voxels | 3,126 |
| Skeleton | Fraction of mask volume | 0.502425% |
| Skeleton | Connected components (26-connectivity) | 1 |
| Skeleton | Endpoint voxels | 26 |
| Skeleton | Branch-point voxels | 142 |
| Skeleton | Voxels outside mask | 0 |

Endpoint and branch-point counts use each skeleton voxel's 26-neighborhood: endpoints have one neighbor and branch-point voxels have at least three. Adjacent branch-point voxels are counted individually, so the branch-point value is a voxel-level complexity measure rather than a count of consolidated junction objects.

## 3D visual gallery

The translucent surface represents the `0.01` segmentation boundary and the red points show the extracted skeleton. The provided renderer downsamples the data by a factor of two. Its displayed normalized isosurface level, `0.720248569`, is the normalized equivalent of the original intensity threshold `0.01`.

### View A — elevation 30°, azimuth 45°

![NDE 3D view at elevation 30 degrees and azimuth 45 degrees](unitcell/nde_view_a_e30_a45.png)

### View B — elevation 60°, azimuth 45°

![NDE 3D view at elevation 60 degrees and azimuth 45 degrees](unitcell/nde_view_b_e60_a45.png)

## Analysis

The mask-to-volume alignment is internally consistent. Every foreground voxel meets the `0.01` intensity criterion, the foreground mean intensity is substantially higher than the background mean, and the mask forms one connected component. The skeleton is also a single connected component and all 3,126 skeleton voxels lie inside the segmented region, indicating that the centerline extraction remains aligned with the mask.

The two 3D perspectives show the skeleton tracking the centers of the lattice struts and passing through their junctions. The 26 endpoint voxels and 142 branch-point voxels reflect a connected, junction-rich lattice topology. Because the images use a two-voxel rendering downsample and no physical voxel spacing is available, they support qualitative morphology and alignment assessment but not calibrated dimensional measurements.

# CT Lattice Segmentation Report

## Inputs and outputs

- Input: `data\9x9x9_octet_lattice\9x9x9_octet_lattice.tif`
- Binary mask: `data\9x9x9_octet_lattice\segmentation\segmented_mask.tif`
- Evaluation image: `data\9x9x9_octet_lattice\segmentation\slice_380.png`
- Volume shape: `(761, 815, 837)`
- Input dtype: `uint16`
- Output dtype: `uint8` (background 0, foreground 1)

## Method

The scan was segmented page-by-page to avoid loading the roughly 1 GB volume
into memory. To accommodate slice-wise baseline drift, the cutoff for each page
was its median intensity plus `9000`. Connected foreground components
smaller than `20` pixels were removed independently on each page
(8-connectivity). No resampling was performed.

## Statistics

- Foreground voxels: `20973726`
- Background voxels: `498146229`
- Total voxels: `519119955`
- Foreground fraction: `0.04040247`
- Background fraction: `0.95959753`
- Per-slice threshold minimum: `39971.000`
- Per-slice threshold median: `41267.000`
- Per-slice threshold maximum: `62404.000`
- Evaluation slice: `380` along axis 0

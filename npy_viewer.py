"""Self contained viewer for .npy files. No external imports beyond numpy and matplotlib."""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

PATH = r".\analysis_outputs/pattern_irregularities_all_groups_20260729_100236/all_groups_pattern_irregularities_viewer_rgb.npy"



def describe(arr, path):
    size_mb = arr.nbytes / (1024 ** 2)
    print(f"file     : {path}")
    print(f"shape    : {arr.shape}")
    print(f"dtype    : {arr.dtype}")
    print(f"size    : {arr.size} elements ({size_mb:.2f} MB)")
    if np.issubdtype(arr.dtype, np.number):
        values = np.asarray(arr).reshape(-1)
        sampled = arr.size > 5_000_000
        if sampled:
            stride = int(np.ceil(arr.size / 5_000_000))
            values = values[::stride]
        finite = np.isfinite(values)
        n_bad = values.size - int(finite.sum())
        if finite.any():
            vals = values[finite]
            print(f"min / max : {vals.min():.6g} / {vals.max():.6g}")
            print(f"mean / std: {vals.mean():.6g} / {vals.std():.6g}")
            if sampled:
                print(f"statistics: sampled every {stride} values")
        if n_bad:
            label = "sampled non finite" if sampled else "non finite"
            print(f"{label}: {n_bad} (NaN or inf)")
    print("\npreview:")
    with np.printoptions(threshold=50, edgeitems=3, precision=4, suppress=True):
        print(arr)



def show_1d(arr, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.asarray(arr).ravel(), lw=1)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()



def show_2d(arr, title, cmap):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.asarray(arr), cmap=cmap, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    plt.tight_layout()



def show_rgb(arr, title):
    img = np.asarray(arr)
    if img.dtype.kind == "f":
        if img.max() > 1.0:
            img = img / img.max()
        img = np.clip(img, 0, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


def show_rgb_volume(arr, title, axis=0):
    vol = np.moveaxis(np.asarray(arr), axis, 0)
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(vol[0])
    ax.set_title(f"{title} | slice 0 / {vol.shape[0] - 1}")
    ax.axis("off")

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        "slice",
        0,
        vol.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        k = int(slider.val)
        im.set_data(vol[k])
        ax.set_title(f"{title} | slice {k} / {vol.shape[0] - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider
    return fig



def show_volume(arr, title, cmap, axis=0):
    vol = np.moveaxis(np.asarray(arr), axis, 0)
    vmin, vmax = np.nanmin(vol), np.nanmax(vol)

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.15)
    im = ax.imshow(vol[0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(f"{title} | slice 0 / {vol.shape[0] - 1}")

    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, "slice", 0, vol.shape[0] - 1, valinit=0, valstep=1)

    def update(val):
        k = int(slider.val)
        im.set_data(vol[k])
        ax.set_title(f"{title}  |  slice {k} / {vol.shape[0] - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    fig._slider = slider # keep a reference so the widget stays responsive
    return fig



def view(path, cmap="viridis", axis=0, no_plot=False, max_load_mb=128):
    size_mb = os.path.getsize(path) / (1024 ** 2)
    mode = "r" if size_mb > max_load_mb else None
    arr = np.load(path, mmap_mode=mode, allow_pickle=False)
    if mode:
        print(f"note: file is {size_mb:.1f} MB, opened with memory mapping\n")

    describe(arr, path)
    if no_plot:
        return arr

    title = os.path.basename(path)
    if arr.ndim == 0:
        print("\nscalar array, nothing to plot")
    elif arr.ndim == 1:
        show_1d(arr, title)
    elif arr.ndim == 2:
        show_2d(arr, title, cmap)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        show_rgb(arr, title)
    elif arr.ndim == 3:
        show_volume(arr, title, cmap, axis=axis)
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        show_rgb_volume(arr, title, axis=axis)
    else:
        print(f"\n{arr.ndim}D array: plotting the first 2D slice")
        show_2d(np.asarray(arr[(0,) * (arr.ndim - 2)]), title, cmap)
    return arr



def run(path, cmap="viridis", axis=0, no_plot=False):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npy")))
        if not files:
            print(f"no .npy files found in {path}")
            print("directory contents:", os.listdir(path))
            return
        print(f"found {len(files)} file(s) in {path}\n")
        for f in files:
            view(f, cmap=cmap, axis=axis, no_plot=no_plot)
            print("-" * 60)
    elif os.path.isfile(path):
        view(path, cmap=cmap, axis=axis, no_plot=no_plot)
    else:
        print(f"path does not exist: {path}")
        return
    plt.show() # a single blocking call after every figure has been built



if __name__ == "__main__":
    run(PATH)

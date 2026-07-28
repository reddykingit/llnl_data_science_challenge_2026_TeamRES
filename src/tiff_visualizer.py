import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import tifffile as tiff

class TiffVisualizer:
    """Interactive visualizer for 2D and 3D TIFF images."""
    def __init__(self, tiff_path):
        self.tiff_path = tiff_path
        self.volume = tiff.imread(tiff_path)
        
        # Ensure data dimension handling
        print(f"Loaded TIFF shape: {self.volume.shape}, dtype: {self.volume.dtype}")
        if self.volume.ndim == 2:
            self.show_2d()
        elif self.volume.ndim == 3:
            self.init_interactive_stack()
        elif self.volume.ndim == 4:
            # If multi-channel or time-series, collapse or select first frame
            print("4D volume detected. Displaying first sequence/channel.")
            self.volume = self.volume[0]
            self.init_interactive_stack()
        else:
            raise ValueError(f"Unsupported array dimensions: {self.volume.ndim}")

    def show_2d(self):
        """Simple display for 2D images."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.volume, cmap='gray')
        plt.colorbar(label='Pixel Intensity')
        plt.title(f"2D Image: {self.tiff_path}")
        plt.axis('off')
        plt.show()

    def init_interactive_stack(self):
        """Creates an interactive slider and scroll-wheel viewer for 3D TIFF stacks."""
        self.current_slice = self.volume.shape[0] // 2
        
        fig, ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(bottom=0.15)  # Leave room for the slider
        
        # Display initial slice
        im = ax.imshow(self.volume[self.current_slice], cmap='gray')
        cbar = fig.colorbar(im, ax=ax, label='Intensity')
        ax.set_title(f"Slice {self.current_slice + 1}/{self.volume.shape[0]}")
        ax.axis('off')

        # Add Slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label='Z Slice ',
            valmin=0,
            valmax=self.volume.shape[0] - 1,
            valfmt='%d',
            valinit=self.current_slice,
            valstep=1
        )

        # Update functions
        def update(val):
            slice_idx = int(slider.val)
            im.set_data(self.volume[slice_idx])
            # Auto-scale intensity for slice
            im.set_clim(vmin=self.volume[slice_idx].min(), vmax=self.volume[slice_idx].max())
            ax.set_title(f"Slice {slice_idx + 1}/{self.volume.shape[0]}")
            fig.canvas.draw_idle()

        def on_scroll(event):
            """Enable mouse-wheel scrolling through slices."""
            if event.button == 'up':
                new_val = min(slider.val + 1, self.volume.shape[0] - 1)
            elif event.button == 'down':
                new_val = max(slider.val - 1, 0)
            else:
                return
            slider.set_val(new_val)

        slider.on_changed(update)
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        plt.show()

def orthoslice_viewer(tiff_path):
    """
    Displays synchronized Orthogonal Cuts (XY, XZ, YZ planes) across the volume.
    """
    volume = tiff.imread(tiff_path)
    if volume.ndim != 3:
        print("Orthoslice mode requires a 3D volume stack.")
        return

    nz, ny, nx = volume.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.3)

    # XY Plane
    im_xy = axes[0].imshow(volume[cz, :, :], cmap='gray', aspect='equal')
    axes[0].set_title(f"XY Plane (Z={cz})")
    
    # XZ Plane
    im_xz = axes[1].imshow(volume[:, cy, :], cmap='gray', aspect='auto')
    axes[1].set_title(f"XZ Plane (Y={cy})")

    # YZ Plane
    im_yz = axes[2].imshow(volume[:, :, cx].T, cmap='gray', aspect='auto')
    axes[2].set_title(f"YZ Plane (X={cx})")

    fig.suptitle("Orthogonal Slice Viewer", fontsize=14)
    plt.show()

# --- Run Visualizer ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively visualize a TIFF image or stack.")
    parser.add_argument("FILE_PATH", help="Path to the TIFF image or stack to visualize")
    args = parser.parse_args()

    # 1. Interactive Slice-by-Slice Viewer
    TiffVisualizer(args.FILE_PATH)

    # 2. Uncomment to view 3D Orthogonal planes simultaneously:
    # orthoslice_viewer(args.FILE_PATH)

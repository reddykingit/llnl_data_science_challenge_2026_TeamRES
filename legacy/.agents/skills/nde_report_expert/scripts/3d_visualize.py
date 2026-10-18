import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os

def visualize_3d(file_path, output_path, threshold=0.5, downsample_factor=2, elev=30, azim=45):
    """
    Renders a 3D visualization of a numpy dataset using marching cubes.
    
    Args:
        file_path (str): Path to the .npy file containing the 3D data.
        threshold (float): Threshold value for extracting the isosurface (0.0 to 1.0).
        downsample_factor (int): Factor to downsample the array to speed up computation.
        elev (float): Elevation angle for the 3D view.
        azim (float): Azimuth angle for the 3D view.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    print(f"Original data shape: {data.shape}")
    
    # Downsample to speed up rendering
    if downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled data shape: {data.shape}")
    
    print("Normalizing data...")
    # Normalize data to 0-1 range to make thresholding predictable
    d_min, d_max = data.min(), data.max()
    if d_min != d_max:
        data = (data - d_min) / (d_max - d_min)

    print(f"Extracting surface using marching cubes at threshold {threshold}...")
    try:
        # Use marching cubes to obtain the surface mesh
        verts, faces, normals, values = measure.marching_cubes(data, level=threshold)
    except ValueError as e:
        print(f"Error extracting isosurface: {e}")
        print("Try adjusting the threshold value.")
        return
        
    print(f"Extracted {len(verts)} vertices and {len(faces)} faces.")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Rendering view (elevation={elev}, azimuth={azim})...")
    
    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap='viridis', lw=0.1, edgecolor='none')
                    
    ax.set_title(f"3D Isosurface (threshold={threshold})")
    ax.view_init(elev=elev, azim=azim)
    
    # Remove axis for better visualization
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D visualization to: {output_path}")
    plt.close()

def visualize_3d_with_skeleton(file_path, skeleton_path, output_path, threshold=0.5, downsample_factor=2, elev=30, azim=45):
    """
    Renders a 3D visualization of a numpy dataset using marching cubes, and overlays a skeleton.
    
    Args:
        file_path (str): Path to the .npy file containing the 3D data.
        skeleton_path (str): Path to the .npy file containing the skeleton.
        output_path (str): Path to save the image.
        threshold (float): Threshold value for extracting the isosurface (0.0 to 1.0).
        downsample_factor (int): Factor to downsample the array to speed up computation.
        elev (float): Elevation angle for the 3D view.
        azim (float): Azimuth angle for the 3D view.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    if not os.path.exists(skeleton_path):
        print(f"Error: Skeleton file not found at {skeleton_path}")
        return

    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    print(f"Original data shape: {data.shape}")
    
    print(f"Loading skeleton from {skeleton_path}...")
    skeleton = np.load(skeleton_path)
    print(f"Original skeleton shape: {skeleton.shape}")
    
    # Downsample to speed up rendering
    if downsample_factor > 1:
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled data shape: {data.shape}")
    
    print("Normalizing data...")
    # Normalize data to 0-1 range to make thresholding predictable
    d_min, d_max = data.min(), data.max()
    if d_min != d_max:
        data = (data - d_min) / (d_max - d_min)

    print(f"Extracting surface using marching cubes at threshold {threshold}...")
    try:
        # Use marching cubes to obtain the surface mesh
        verts, faces, normals, values = measure.marching_cubes(data, level=threshold)
    except ValueError as e:
        print(f"Error extracting isosurface: {e}")
        print("Try adjusting the threshold value.")
        return
        
    print(f"Extracted {len(verts)} vertices and {len(faces)} faces.")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Rendering view (elevation={elev}, azimuth={azim})...")
    
    # Plot the surface
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap='viridis', lw=0.1, edgecolor='none', alpha=0.3)
                    
    # Plot the skeleton
    print("Extracting and scaling skeleton coordinates...")
    dim0, dim1, dim2 = np.where(skeleton > 0)
    dim0 = dim0 / downsample_factor
    dim1 = dim1 / downsample_factor
    dim2 = dim2 / downsample_factor
    
    ax.scatter(dim0, dim1, dim2, color='red', s=1.0, alpha=0.8, label='Skeleton')
                    
    ax.set_title(f"3D Isosurface with Skeleton\n(threshold={threshold})")
    ax.view_init(elev=elev, azim=azim)
    
    # Remove axis for better visualization
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D visualization to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Hardcoded parameters for testing
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "octet_truss_unit_cell_no_defects_0256_xray_recon.npy"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "render_output.png"))
    threshold = 0.5
    downsample_factor = 2
    elev = 60.0
    azim = 45.0
    
    visualize_3d(
        file_path=file_path, 
        output_path=output_path,
        threshold=threshold, 
        downsample_factor=downsample_factor, 
        elev=elev, 
        azim=azim
    )
    
    skeleton_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "octet_truss_unit_cell_skeleton.npy"))
    output_path_skel = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "render_output_with_skeleton.png"))
    
    visualize_3d_with_skeleton(
        file_path=file_path,
        skeleton_path=skeleton_path,
        output_path=output_path_skel,
        threshold=threshold, 
        downsample_factor=downsample_factor, 
        elev=elev, 
        azim=azim
    )


import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

def plot_3d_motion(
    save_path, 
    kinematic_tree, 
    joints, 
    title, 
    dataset, 
    figsize=(3, 3), 
    fps=120, 
    radius=3,
    vis_mode='default', 
    gt_frames=[],
    other_joints=None
):
    """
    Plots a 3D motion animation and saves it as a GIF.

    Parameters:
    - save_path (str): Path to save the resulting GIF.
    - kinematic_tree (list): Structure defining joint connections.
    - joints (numpy.ndarray): Joint positions (seq_len, joints_num, 3).
    - title (str): Title for the plot.
    - dataset (str): Dataset identifier to adjust visualization scale.
    - figsize (tuple): Figure size in inches (width, height).
    - fps (int): Frames per second for the animation.
    - radius (float): Spatial radius for the plot bounds.
    - vis_mode (str): Visualization mode (e.g., 'default', 'gt', 'upper_body').
    - gt_frames (list): Indices of ground-truth frames.

    Returns:
    - None
    """
    # Switch the backend to avoid GUI rendering issues.
    print("Other joints?", other_joints is not None)
    matplotlib.use('Agg')

    # Wrap long titles for better display.
    title = '\n'.join(wrap(title, 20))

    def init():
        """Initialize the 3D plot axes."""
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xz_plane(minx, maxx, miny, minz, maxz):
        """Plots an XZ plane in the 3D space."""
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def plot_xy_box(minx, maxx, miny, maxy, z):
        """Plots an outline of an XY box in the 3D space."""
        # Define the vertices of the box
        verts = [
            [minx, miny, z],
            [minx, maxy, z],
            [maxx, maxy, z],
            [maxx, miny, z],
            [minx, miny, z]  # Close the loop
        ]

        # Create a 3D line collection
        xy_box = Line3DCollection([verts], colors="black")
        ax.add_collection3d(xy_box)

    # Reshape and scale joint data based on dataset type.
    data = joints.copy().reshape(len(joints), -1, 3)
    if dataset == 'kit':
        data *= 0.003  # Scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # Scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5  # Reverse axes and scale for visualization
    
    if other_joints is not None:
        other_data = other_joints.copy().reshape(len(joints), -1, 3)
        if dataset == 'kit':
            other_data *= 0.003  # Scale for visualization
        elif dataset == 'humanml':
            other_data *= 1.3  # Scale for visualization
        elif dataset in ['humanact12', 'uestc']:
            other_data *= -1.5  # Reverse axes and scale for visualization

    # Create the figure and 3D axis.
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = Axes3D(fig)
    
    init()

    # Determine data bounds.
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    # Define colors for different visualization modes.
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange if vis_mode != 'gt' else colors_blue

    if vis_mode == 'upper_body':
        colors[:2] = colors_blue[:2]

    frame_number = data.shape[0]

    # Adjust heights and trajectories for visualization.
    height_offset = mins[1]
    data[:, :, 1] -= height_offset
    if other_joints is not None:
        other_data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    #data[..., 0] -= data[:, 0:1, 0]
    #data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        """Update the plot for a specific frame."""
        ax.lines = []
        ax.collections = []
        
        # Plot XZ plane at the current frame.
        if maxs[2] - mins[2] > 0.001:
            ax.view_init(elev=90, azim=-90)
            ax.dist = 7.5
            plot_xz_plane(mins[0], maxs[0], 0, mins[2], maxs[2])
        else:
            ax.view_init(elev=90, azim=-90)
            ax.dist = 7.5
            plot_xy_box(-1, 1, -1, 1, maxs[2])

        # Determine colors for the current frame.
        used_colors = colors_blue if index in gt_frames else colors

        # Plot the kinematic tree connections.
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color
            )
        if other_joints is not None:
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors_blue)):
                linewidth = 4.0 if i < 5 else 2.0
                ax.plot3D(
                    other_data[index, chain, 0],
                    other_data[index, chain, 1],
                    other_data[index, chain, 2],
                    linewidth=linewidth,
                    color=color
                )

        # Remove axis labels for a cleaner visualization.
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Create the animation.
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # Save the animation as a GIF.
    writergif = matplotlib.animation.PillowWriter(fps=fps)
    ani.save(f"{save_path}.gif", writer=writergif)

    # Close the plot to free resources.
    plt.close()

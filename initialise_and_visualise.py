import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

def initialise_cell(s):
    """
    Initialise a cell with given size, s.
    """
    #random.seed(42) # For reproducibility
    np.random.seed(42)

    cell = np.zeros((s, s), dtype=int)
    mt_number = random.randint(50, 70)

    for _ in range(mt_number):
        x, y = random.randint(0, s-1), random.randint(0, s-1)
        cell[x, y] = 1
    return cell

def visualise_step(cell, individual_mt, ax1, ax2, ax3, ax4, step, overall_ros, overall_atp, resources):
    ax1.clear()
    color_map = np.ones((*cell.shape, 3))            
    healthy_count = sick_count = mitophagy_count = 0
    for (x, y), mt in individual_mt.items():
        if mt.mtdna < 50:
            color_map[x, y] = [1, 0, 0]  # red
            mitophagy_count += 1
        elif mt.mtdna < 100:
            color_map[x, y] = [1, 0.65, 0]  # orange
            sick_count += 1
        else:
            color_map[x, y] = [0, 1, 0]  # green
            healthy_count += 1
    ax1.imshow(color_map, origin="lower")
    ax1.set_title(f"Step {step}: Cell state, Count = {mitophagy_count + sick_count + healthy_count}")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Healthy'),
        Patch(facecolor='orange', label='Sick'),
        Patch(facecolor='red', label='Mitophagy flag'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.75, 1))
    ax1.axis('off')
    
    # ROS plot
    ax2.clear()
    ax2.plot(overall_ros, 'r-', marker='o', markersize=3)
    ax2.set_title('ROS Over Time')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('ROS')
    ax2.grid(True, alpha=0.3)
    
    # ATP plot
    ax3.clear()
    ax3.plot(overall_atp, 'b-', marker='s', markersize=3)
    ax3.set_title('ATP Over Time')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('ATP')
    ax3.grid(True, alpha=0.3)
    
    # Resource Panel
    ax4.clear()
    steps = range(1, len(resources.get('ca', [])) + 1)
    if len(steps) > 0:
        ax4.plot(steps, resources['ca'], marker='.')
        ax4.plot(steps, resources['camp'], marker='.')
        ax4.plot(steps, resources['glucose'], marker='.')
        ax4.legend(['Ca$^{2+}$', 'cAMP', 'Glucose'], loc='lower left', fontsize=8, frameon=True)
    ax4.set_title('Resource Pools')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Pool level')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.5)

def _build_color_map(cell, individual_mt):
    """
    Create the RGB map and counts by health state.
    """
    color_map = np.ones((*cell.shape, 3))  # start white
    healthy_count = sick_count = mitophagy_count = 0
    for (x, y), mt in individual_mt.items():
        if hasattr(mt, 'mtdna'):
            if mt.mtdna < 50:
                color_map[x, y] = [1, 0, 0]       # red
                mitophagy_count += 1
            elif mt.mtdna < 100:
                color_map[x, y] = [1, 0.65, 0]    # orange
                sick_count += 1
            else:
                color_map[x, y] = [0, 1, 0]       # green
                healthy_count += 1
        else:
            color_map[x, y] = [1, 1 ,1]     # background white
    return color_map, healthy_count, sick_count, mitophagy_count


def draw_cell(ax, cell, individual_mt, title=None):
    """
    Draw the cell heatmap onto the provided axes (no plt.show).
    Keeps the style consistent with visualise/visualise_with_stats.
    """
    color_map, h, s, a = _build_color_map(cell, individual_mt)
    ax.clear()
    ax.imshow(color_map, origin="lower")
    if title is None:
        title = f"Mitochondria State (Healthy:{h}, Sick:{s})"
    ax.set_title(title)
    ax.axis("off")


def visualise_event(cell,
                    individual_mt,
                    event_type,
                    step=None,
                    ax=None):
    """
    Overlay a visual 'flash' for fusion/fission events onto an axes or a new fig.

    Args:
        cell, individual_mt: current state
        event_type: "fusion" or "fission"
        coords_a: sequence of (x,y) pixels for group A (pre-op)
        coords_b: sequence of (x,y) pixels for group B (fusion partner),
                  or a singleton [(x_new,y_new)] for the split piece (fission)
        step: optional step index (for the title)
        ax: optional matplotlib Axes. If None, a new figure is created.
        flash_times: how many pulses to draw
        flash_pause: seconds between pulses
    """
    # Prepare where to draw
    created_fig = False
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        created_fig = True

    # Base drawing (no blocking)
    base_title = f"Step {step}: " if step is not None else ""
    if event_type.lower() == "fusion":
        base_title += "Fusion event"
    elif event_type.lower() == "fission":
        base_title += "Fission event"
    else:
        base_title += "Event"

    draw_cell(ax, cell, individual_mt, title=base_title)
    if created_fig:
        plt.show(block=False)
        plt.pause(0.05)
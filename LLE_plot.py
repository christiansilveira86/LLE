import matplotlib.pyplot as plt
import ternary
import numpy as np
import ternary

def LLE_plot():
    # Read the data from your .txt file (replace 'output.txt' with the actual file path)
    with open('output1.txt', 'r') as file:
        lines = file.readlines()

    # Parse the data
    Z_points = np.array([list(map(float, line.split()[:3])) for line in lines])
    xI_points = np.array([list(map(float, line.split()[3:6])) for line in lines])
    xII_points = np.array([list(map(float, line.split()[6:])) for line in lines])

    # Filter out rows with NaN values in either xI_points or xII_points
    valid_indices = ~(np.isnan(xI_points).any(axis=1) | np.isnan(xII_points).any(axis=1))
    # Filter out rows with -999.00000 values in either xI_points or xII_points
    # valid_indices = ~(np.all(xI_points==-999.00000,axis=1) | np.all(xII_points==-999.00000,axis=1))
    xI_points_filtered = xI_points[valid_indices]
    xII_points_filtered = xII_points[valid_indices]
    Z_points_filtered = Z_points[valid_indices]
    # Filtered data
    xI = xI_points_filtered
    xII = xII_points_filtered
    Z = Z_points_filtered

    # Create a ternary plot
    figure, tax = ternary.figure(scale=1)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color=(97/255,97/255,97/255), multiple=0.1, linewidth=0.7)

    # Set axis ticks
    tax.ticks(axis='lbr', linewidth=0.7, multiple=0.05, fontsize=8, tick_formats="%.2f")

    # Plot each phase separately and assign labels
    tax.scatter(xI, marker='o', color=(116/255, 157/255, 161/255), label='Phase-I')
    tax.scatter(xII, marker='o', color=(180/255, 190/255, 137/255), label='Phase-II')
    tax.scatter(Z, marker='o', color=(235/255, 145/255, 145/255), label='Feed')

    # Draw lines connecting each xI to xII point
    for i in range(len(xI)):
        tax.line(xI[i], xII[i], linestyle='--', color='gray')

    # Customize labels and title
    tax.set_title("Ternary Liquid-Liquid Equilibrium Diagram", fontsize=16)
    tax.left_axis_label("Component C", fontsize=12)
    tax.right_axis_label("Component B", fontsize=12)
    tax.bottom_axis_label("Component A", fontsize=12, offset=-0.08)

    # Add legend
    tax.legend(fontsize=12)
    # Remove gray background
    tax.clear_matplotlib_ticks()

    # Show the plot
    plt.show()
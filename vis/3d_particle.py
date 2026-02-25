import ptrReader
import numpy as np 
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import struct
from tqdm import tqdm

plt.style.use('dark_background')

re = 6378137.0
files = sys.argv[1::]
x_coords = []
y_coords = []
z_coords = []

# Data Extraction
for file in tqdm(files):
    a, b, c, vx, vy, vz = ptrReader.read_ptr2_file(file)
    # ptrReader likely returns arrays; extend the list to keep it flat
    x_coords.extend(a / re)
    y_coords.extend(b / re)
    z_coords.extend(c / re)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trace
ax.plot(x_coords, y_coords, z_coords, color='cyan', lw=2.7, label='Particle Trace')

# Optional: Add a sphere to represent Earth for scale
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
# ax.plot_wireframe(xs, ys, zs, color="white", alpha=0.2, linewidths=0.5)

# Formatting
ax.set_xlabel('X ($R_e$)')
ax.set_ylabel('Y ($R_e$)')
ax.set_zlabel('Z ($R_e$)')
ax.set_title('3D Particle Trace')
ax.legend()

# Equalize aspect ratio (crucial for orbital/geospatial plots)
# Note: set_box_aspect is available in Matplotlib 3.3.0+
# ax.set_box_aspect([1,1,1]) 

plt.show()

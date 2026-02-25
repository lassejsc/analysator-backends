import sys
import ptrReader
import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from tqdm import tqdm
from multiprocessing import Pool
import subprocess
import matplotlib
import vlsvrs
from matplotlib.colors import LogNorm
import glob

files_AID = np.sort(glob.glob("/wrk-vakka/users/souhadah/vlsvrs/state.*.ptr"))

# for idx, file in enumerate(tqdm(files_AID)):
for idx, file in zip([150], [files_AID[150]]):


    re = 6378137.0
    km2m = 1e3
    # var = "proton/vg_rho"



    plane = "XY"
    vlsv_file = f"/wrk-vakka/group/spacephysics/vlasiator/2D/AID/bulk/bulk.{610+idx:07d}.vlsv"
    # files = sys.argv[3:]

    f = vlsvrs.VlsvFile(vlsv_file)

    # density = f.read_variable_f32("proton/vg_rho", op = 0).squeeze()
    # velocity = f.read_variable_f32("proton/vg_v", op = 0).squeeze()
    Efield = f.read_variable_f32("vg_e_vol", op = 0).squeeze()
    # magfield = f.read_variable_f32("vg_b_vol", op = 0).squeeze()

    # JdotE = JdotE_function((density, velocity, Efield))

    ext = f.get_spatial_mesh_extents()
    x_min, y_min, z_min, x_max, y_max, z_max = ext

    if plane == "XY":
        extent = [x_min/re, x_max/re, y_min/re, y_max/re]
    elif plane == "XZ":
        extent = [x_min/re, x_max/re, z_min/re, z_max/re]
    else:
        raise ValueError("Plane must be XY or XZ")


    x, y, z, vx, vy, vz = ptrReader.read_ptr2_file(file)
    x /= re
    y /= re
    z /= re

    EV_TO_J   = 1.602176634e-19

    PROTON_MASS = 1.67262192369e-27

    particle_energy = 0.5 * PROTON_MASS * (vx**2 + vz**2) / EV_TO_J * 1e-3 # in keV

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))


    im = ax.imshow(
    Efield[:,:,1].T*1e3,
    origin="lower",
    extent=extent,
    cmap="RdBu",
    alpha=1,
    vmin = -3, vmax = 3
    )
    cbar = fig.colorbar(im, label = "$J \\cdot E$ (nW/m$^3$)")

    if plane == "XY":
        real = ax.scatter(x, y, s=0.1, c=particle_energy, cmap = "nipy_spectral", vmin = 0, vmax = 6)

        cbar1 = fig.colorbar(real, label='Relative energy change')
        ax.set_xlabel("X [RE]")
        ax.set_ylabel("Y [RE]")


    ax.set_title(f"time = {(610+idx)/2}s")

    # plt.savefig(f"AID/particle_tracing/ants_everywhere/pop_{idx:07d}.png", dpi=300)
    # plt.close()


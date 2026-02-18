
# fmt: off
import os  # noqa
import gc

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from desc import set_device

if True:
    set_device("gpu", 0)

import jax
try:
    print(jax.devices())
    jax.config.update("jax_compilation_cache_dir", "../jax-caches")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
except Exception as e:
    print(e)
# fmt: on


from qsc import Qsc
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.io import hdf5Writer
from desc.optimize import Optimizer
from desc.plotting import plot_boozer_surface, plot_3d
from desc.objectives import (
    ObjectiveFunction, QuasisymmetryTwoTerm, FixBoundaryZ,
    QuasisymmetryBoozer, AspectRatio, ForceBalance,
    FixBoundaryR, FixPressure, FixCurrent, FixPsi, Volume, RotationalTransform
)

from desc.plotting import LinearGrid
import numpy as np

test_results = []

# TODO: Vary: Size (as discussed below), nfp, radius, etabar
# NOTE:
#       - rc[0]: Gives major radius
#       - zs[0]: Shifts the vertical axis up and down
#       - rc[1]: Gives vertical width of the spiral  <---------|
#       - zs[1]: Gives Vertical height ofthe spiral (I think I should keep these the same but opposite)
#       - Varying rc[1], zs[1], rc[0] should give ability to tune size

IOTAs = [i / np.pi for i in range(1, 4)]
ASPECT_RATIOs = [4, 5, 8]
NFPs = [2, 3, 4]

for iota in IOTAs:
    for aspectRatio in ASPECT_RATIOs:
        for NFP in NFPs:
            # stel = Qsc.from_paper("2022 QH nfp3 vacuum", nfp=NFP)

            # ntheta = 50
            # eq_fixed_bdry = Equilibrium.from_near_axis(
            #     stel,
            #     r=0.10,
            #     L=3,
            #     M=4,
            #     N=4,
            #     ntheta=ntheta
            # )

            surf = FourierRZToroidalSurface(
                R_lmn=[1, 0.125, 0.1],
                Z_lmn=[-0.125, -0.1],
                modes_R=[[0, 0], [1, 0], [0, 1]],
                modes_Z=[[-1, 0], [0, -1]],
                M=4,
                N=4,
                sym=True,
                NFP=4
            )

            eq_fixed_bdry = Equilibrium(M=4, N=4, Psi=0.04, surface=surf)

            grid = LinearGrid(
                rho=1,
                theta=np.linspace(0, 2 * np.pi, 100),
                zeta=np.linspace(0, 2 * np.pi, 100),
                axis=True
            )

            VOLUME = eq_fixed_bdry.compute("V")["V"]
            IOTAs = [i / np.pi for i in range(1, 4)]
            ASPECT_RATIOs = []

            grid = LinearGrid(
                L=3, N=4,
                M=4,
                NFP=eq_fixed_bdry.NFP,
                sym=eq_fixed_bdry.sym
            )

            grid_boozer = LinearGrid(
                L=3,
                N=4,
                M=4,
                NFP=eq_fixed_bdry.NFP,
                sym=False
            )

            eq = eq_fixed_bdry
            for kp in range(1, 5):

                R_modes = np.vstack((
                    [0, 0, 0],
                    eq.surface.R_basis.modes[
                        np.max(np.abs(eq.surface.R_basis.modes), 1) > kp, :
                    ])
                )

                Z_modes = eq.surface.Z_basis.modes[
                    np.max(np.abs(eq.surface.Z_basis.modes), 1) > kp, :
                ]

                constraints = (
                    ForceBalance(eq=eq, grid=grid),
                    FixPressure(eq=eq),
                    FixCurrent(eq=eq),
                    FixPsi(eq=eq),
                    FixBoundaryR(eq=eq, modes=R_modes),
                    FixBoundaryZ(eq=eq, modes=Z_modes),
                )

                optimizer = Optimizer('proximal-lsq-exact')
                eq_fixed_bdry.optimize(
                    optimizer=optimizer,
                    verbose=3,
                    ftol=1e-5,
                    objective=ObjectiveFunction((

                        AspectRatio(eq=eq, target=aspectRatio),
                        QuasisymmetryBoozer(
                            eq=eq, helicity=(1, NFP), grid=grid_boozer,
                            normalize=True, weight=10.0),

                        RotationalTransform(
                            eq=eq_fixed_bdry, target=iota, weight=10,
                            normalize=True)

                    ),
                    ),
                    maxiter=20,
                    xtol=1e-5,
                    constraints=constraints
                )

            filename = f"solved_nfp{NFP}"
            hdf5Writer(f"./data/{filename}.h5", 'w').write_obj(eq)

            test_results.append({
                "nfp": NFP,
                "iota": eq.iota,
                "eq_object": eq
            })
            grid = LinearGrid(
                rho=1,
                theta=np.linspace(0, 2 * np.pi, 100),
                zeta=np.linspace(0, 2 * np.pi, 100),
                axis=True
            )

            fig = plot_3d(eq, "|F|_normalized", log=True, grid=grid)
            fig.write_image(f"./image/{filename}.png")
            fig, ax = plot_boozer_surface(eq)
            fig.savefig(f"./image/{filename}_boozer.png")

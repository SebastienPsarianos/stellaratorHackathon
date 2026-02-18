
# fmt: off
import os  # noqa
import gc

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]


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
from desc.io import hdf5Writer
from desc.optimize import Optimizer
from desc.plotting import plot_boozer_surface, plot_3d
from desc.objectives import (
    ObjectiveFunction, QuasisymmetryTwoTerm, FixBoundaryZ,
    QuasisymmetryBoozer, AspectRatio, ForceBalance,
    FixBoundaryR, FixPressure, FixCurrent, FixPsi,
)

from desc.plotting import LinearGrid
import numpy as np

NFPs = [2, 3, 4]
scalingFactors = [0.03, 0.045, 0.06]
r_vals = [0.25]  # 0.3, 0.35]

# TODO: Vary: Size (as discussed below), nfp, radius, etabar
# NOTE:
#       - rc[0]: Gives major radius
#       - zs[0]: Shifts the vertical axis up and down
#       - rc[1]: Gives vertical width of the spiral  <---------|
#       - zs[1]: Gives Vertical height ofthe spiral (I think I should keep these the same but opposite)
#       - Varying rc[1], zs[1], rc[0] should give ability to tune size


test_results = []

for NFP in NFPs:
    for scalingFactor in scalingFactors:
        for r_val in r_vals:
            stel = Qsc(rc=[1, scalingFactor], zs=[
                0, -scalingFactor], nfp=NFP, etabar=-0.9)

            ntheta = 75
            r = r_val
            eq_fixed_bdry = Equilibrium.from_near_axis(
                stel,
                r=r,
                L=3,
                M=6,
                N=6,
                ntheta=ntheta
            )

            eq = eq_fixed_bdry.copy()
            print("Z-Basis Modes:", eq_fixed_bdry.surface.Z_basis.modes)

            ASPECT_RATIO = eq_fixed_bdry.compute("R0/a")["R0/a"]

            try:
                vol = eq_fixed_bdry.compute("V")["V"]
                if vol < 0 or np.isnan(vol):
                    print(f"Skipping Invalid Geometry: NFP={
                        NFP}, r={r_val}")
                    continue  # Skip to next iteration
            except:
                continue

            grid = LinearGrid(
                L=6, N=12,
                M=12,
                NFP=eq_fixed_bdry.NFP,
                sym=eq_fixed_bdry.sym
            )

            grid_boozer = LinearGrid(
                L=12,
                N=24,
                M=24,
                NFP=eq_fixed_bdry.NFP,
                sym=False
            )

            fixed_modes_R = np.array([
                [0, 0],
                [1, 0]
            ])

            fixed_modes_Z = np.array([
                [-1, 0]
            ])

            constraints = (
                FixPressure(eq=eq_fixed_bdry),
                FixCurrent(eq=eq_fixed_bdry),
                FixPsi(eq=eq_fixed_bdry),
                FixBoundaryR(eq=eq_fixed_bdry, modes=fixed_modes_R),
                FixBoundaryZ(eq=eq_fixed_bdry, modes=fixed_modes_Z)
            )

            optimizer = Optimizer('proximal-lsq-exact')
            eq_fixed_bdry.optimize(
                optimizer=optimizer,
                verbose=3,
                ftol=1e-3,
                objective=ObjectiveFunction((
                    AspectRatio(eq=eq_fixed_bdry, target=ASPECT_RATIO,
                                grid=grid,  normalize=True, weight=10.0),
                    QuasisymmetryTwoTerm(
                        eq=eq_fixed_bdry, helicity=(1, NFP),  grid=grid,
                        normalize=True, weight=20.0),
                    ForceBalance(eq=eq_fixed_bdry, grid=grid,
                                 normalize=True, weight=1.0)
                )
                ),
                maxiter=500,
                xtol=1e-4,
                constraints=constraints
            )

            eq_fixed_bdry.optimize(
                optimizer=optimizer,
                verbose=3,
                ftol=1e-5,
                objective=ObjectiveFunction((
                    AspectRatio(eq=eq_fixed_bdry, target=ASPECT_RATIO,
                                grid=grid,  normalize=True, weight=10.0),
                    QuasisymmetryBoozer(
                        eq=eq_fixed_bdry, helicity=(1, NFP), grid=grid_boozer,
                        normalize=True, weight=2.0),

                    ForceBalance(eq=eq_fixed_bdry, grid=grid,
                                 normalize=True, weight=10.0)
                )
                ),
                maxiter=500,
                xtol=1e-5,
                constraints=constraints
            )

            filename = f"solved_nfp{NFP}_r{
                r_val:.2f}_scale{scalingFactor:.3f}"
            hdf5Writer(f"./data/{filename}.h5", 'w').write_obj(eq_fixed_bdry)

            test_results.append({
                "nfp": NFP,
                "excursion": scalingFactor,
                "r": r_val,
                "iota": eq_fixed_bdry.iota,
                "aspect_ratio": ASPECT_RATIO,
                "eq_object": eq_fixed_bdry
            })
            grid = LinearGrid(
                rho=1,
                theta=np.linspace(0, 2 * np.pi, 100),
                zeta=np.linspace(0, 2 * np.pi, 100),
                axis=True
            )

            fig = plot_3d(eq_fixed_bdry, "|F|_normalized", log=True, grid=grid)
            fig.write_image(f"./image/{filename}.png")
            fig, ax = plot_boozer_surface(eq_fixed_bdry)
            fig.savefig(f"./image/{filename}_boozer.png")

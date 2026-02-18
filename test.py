
# fmt: off
import os  # noqa

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


from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.optimize import Optimizer
from desc.plotting import plot_boozer_surface, plot_3d, plot_boundary
from desc.objectives import (
    ObjectiveFunction, QuasisymmetryTwoTerm, AspectRatio, ForceBalance,
    FixBoundaryR, FixBoundaryZ, FixPressure, FixCurrent, FixPsi)

from desc.plotting import LinearGrid
import numpy as np

NFPs = [2, 3, 4, 5, 6]
scalingFactor = [1, 2, 3, 4, 5, 6]
aspectRatio = []


surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, 0.1],
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    M=20,
    N=20,
    sym=True,
    NFP=4
)

# create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# default is zero pressure and zero current
eq = Equilibrium(M=5, N=5, Psi=0.04, surface=surf)
fig, ax = plot_boundary(eq)
fig.savefig("TEST.png")

grid = LinearGrid(
    rho=0.8,
    theta=np.linspace(0, 2 * np.pi, 100),
    zeta=np.linspace(0, 2 * np.pi, 100),
    axis=True
)

fig = plot_3d(eq, "|F|", log=True, grid=grid)
fig.write_image("preoptimization.png")

# this is usually all you need to solve a fixed boundary equilibrium
eq0 = solve_continuation_automatic(eq, verbose=0)[-1]
fig = plot_3d(eq0, "|F|_normalized", log=True, grid=grid)
fig.write_image("postptimization.png")

exit()


def run_qh_optimization(k, eq):
    grid = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP,
        rho=np.array([0.6, 0.8, 1.0]),
        sym=True
    )

    objective = ObjectiveFunction((
        QuasisymmetryTwoTerm(eq=eq, helicity=(
            1, eq.NFP), grid=grid, weight=20),

        AspectRatio(eq=eq, target=8, weight=100))
    )

    # NOTE: Values greater than k are given as constraints
    R_modes = np.vstack((
        [0, 0, 0],
        eq.surface.R_basis.modes[
            np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
        ],)
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]

    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq)
    )

    optimizer = Optimizer('proximal-lsq-exact')

    maxIter = 5 * (i + 1)

    eq_new, history = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=maxIter,
        verbose=3,
        copy=True,
        options={
            "initial_trust_ratio": 0.1,
        }
    )

    return eq_new


grid = LinearGrid(
    rho=1,
    theta=np.linspace(0, 2 * np.pi, 100),
    zeta=np.linspace(0, 2 * np.pi, 100),
    axis=True
)
eq_test = eq
for i in range(1, 20):
    eq_test = run_qh_optimization(i, eq_test)
    fig = plot_3d(eq_test, "|F|", log=True, grid=grid)
    fig.write_image(f"improvement_{i}.png")
    fig, ax = plot_boozer_surface(eq_test)
    fig.savefig(f"improvement_{i}_boozer.png")

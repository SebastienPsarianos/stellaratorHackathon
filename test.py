from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.plotting import plot_boozer_surface, plot_3d

surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, 0.1],
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=4,
)

# create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# default is zero pressure and zero current
eq = Equilibrium(M=4, N=4, Psi=0.04, surface=surf)
plot_boozer_surface(eq)
plot_3d(eq)

# this is usually all you need to solve a fixed boundary equilibrium
# eq0 = solve_continuation_automatic(eq, verbose=0)[-1]

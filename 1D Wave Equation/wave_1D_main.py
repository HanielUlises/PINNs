import numpy as np
from sympy import Symbol, sin

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key

from wave_eq_main import WaveEquation1D


@modulus.sym.main(config_path="./", config_name="config_main")
def run(cfg: ModulusConfig) -> None:

    # PDE
    we = WaveEquation1D(c=1.0)

    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]

    x_symbol = Symbol("x")
    t_symbol = Symbol("t")

    L = float(np.pi)

    geometry = Line1D(0, L)

    time_range = {
        t_symbol: (0, 2 * L)
    }

    domain = Domain()

    # Initial Condition
    # u(x,0)=sin(x)
    # ut(x,0)=sin(x)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={
            "u": sin(x_symbol),
            "u__t": sin(x_symbol),
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "u": 1.0,
            "u__t": 1.0,
        },
        parameterization={
            t_symbol: 0.0
        },
    )

    domain.add_constraint(IC, "IC")

    # Boundary Conditions
    # u(0,t)=0
    # u(pi,t)=0
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={
            "u": 0,
        },
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )

    domain.add_constraint(BC, "BC")

    # PDE Residual
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={
            "wave_equation": 0,
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )

    domain.add_constraint(interior, "interior")

    # Validation Data
    # Exact solution:
    # u(x,t)=sin(x)(cos(t)+sin(t))
    delta_x = 0.01
    delta_t = 0.01

    x = np.arange(0, L + delta_x, delta_x)
    t = np.arange(0, 2 * L + delta_t, delta_t)

    X, T = np.meshgrid(x, t)

    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    U = np.sin(X_flat) * (np.cos(T_flat) + np.sin(T_flat))

    invar = {
        "x": X_flat,
        "t": T_flat,
    }

    outvar = {
        "u": U,
    }

    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar,
        true_outvar=outvar,
        batch_size=128,
    )

    domain.add_validator(validator)

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
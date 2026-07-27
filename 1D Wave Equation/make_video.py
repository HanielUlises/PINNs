"""Render an animation of the trained 1D wave-equation PINN.

Loads the FullyConnected network trained by ``wave_1D_main.py`` from its
checkpoint, evaluates it over the (x, t) domain, and writes an MP4 comparing
the PINN prediction against the exact solution

    u(x, t) = sin(x) * (cos(t) + sin(t)).
"""

import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.key import Key


@modulus.sym.main(config_path="./", config_name="config_main")
def run(cfg: ModulusConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hydra changes the working directory at runtime, so anchor paths to the
    # directory that holds this script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "outputs/wave_1D_main/wave_network.0.pth")
    out_path = os.path.join(base_dir, "wave_1D.mp4")

    # Rebuild the same architecture used for training and load the checkpoint.
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    wave_net.load_state_dict(ckpt)
    wave_net.to(device).eval()

    # Evaluation grid.
    L = float(np.pi)
    x = np.linspace(0, L, 256)
    t = np.linspace(0, 2 * L, 240)

    def pinn_at(tv):
        xv = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
        tt = torch.full_like(xv, float(tv))
        with torch.no_grad():
            out = wave_net({"x": xv, "t": tt})["u"]
        return out.cpu().numpy().flatten()

    def exact_at(tv):
        return np.sin(x) * (np.cos(tv) + np.sin(tv))

    # Figure setup.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (line_pinn,) = ax.plot([], [], color="#4C78A8", lw=2.5, label="PINN")
    (line_exact,) = ax.plot(
        [], [], color="#E45756", lw=1.8, ls="--", label="Exact"
    )
    ax.set_xlim(0, L)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    title = ax.set_title("")
    fig.tight_layout()

    def update(i):
        tv = t[i]
        line_pinn.set_data(x, pinn_at(tv))
        line_exact.set_data(x, exact_at(tv))
        title.set_text(f"1D Wave Equation  (c = 1)   t = {tv:.2f}")
        return line_pinn, line_exact, title

    anim = FuncAnimation(fig, update, frames=len(t), blit=True)
    writer = FFMpegWriter(fps=30, bitrate=2400)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()

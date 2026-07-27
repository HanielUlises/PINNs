"""Instructor-style post-processing animation for the 1D wave-equation PINN.

Reuses the visualization style from ``post_processing_1dwave_main.ipynb`` --
rainbow ``plot_trisurf`` surfaces of u over the (x, t) plane -- and turns it
into a rotating video.

The data is read straight from the PhysicsNeMo validator output
(``outputs/wave_1D_main/validators/validator.vtp``), which carries the same
fields as the notebook's ``results_valid.csv`` (Points_0, t, pred_u, true_u).
The CSV files the notebook expects are also (re)exported here so the notebook
stays runnable.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import vtk
from vtk.util.numpy_support import vtk_to_numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "wave_1D_main")


def load_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    pts = vtk_to_numpy(data.GetPoints().GetData())
    pd_ = data.GetPointData()
    fields = {
        pd_.GetArrayName(i): vtk_to_numpy(pd_.GetArray(i))
        for i in range(pd_.GetNumberOfArrays())
    }
    fields["Points_0"] = pts[:, 0]
    fields["Points_1"] = pts[:, 1]
    fields["Points_2"] = pts[:, 2]
    return pd.DataFrame(fields)


def main():
    # --- Validator field: predicted vs. exact solution over (x, t) ----------
    df = load_vtp(os.path.join(OUT_DIR, "validators", "validator.vtp"))
    df["Error"] = df["pred_u"] - df["true_u"]
    df.to_csv(os.path.join(BASE_DIR, "results_valid.csv"), index=False)

    # Also export the interior-points CSV the notebook's first half uses.
    df_int = load_vtp(os.path.join(OUT_DIR, "constraints", "interior.vtp"))
    df_int.to_csv(os.path.join(BASE_DIR, "results_int_points.csv"), index=False)

    # Subsample the ~200k validation points to keep per-frame rendering fast;
    # plot_trisurf triangulates the scattered (x, t) points, so a random
    # subset still yields a faithful surface.
    rng = np.random.default_rng(0)
    n_sub = min(4000, len(df))
    idx = rng.choice(len(df), size=n_sub, replace=False)
    dfs = df.iloc[idx]

    x, t = dfs["Points_0"].to_numpy(), dfs["t"].to_numpy()
    panels = [
        ("Predicted  u(x, t)", dfs["pred_u"].to_numpy(), "rainbow"),
        ("Exact  u(x, t)", dfs["true_u"].to_numpy(), "rainbow"),
        ("Error  (pred - exact)", dfs["Error"].to_numpy(), "rainbow"),
    ]

    # Shared axis limits for the two solution panels so they read together.
    u_min = min(df["pred_u"].min(), df["true_u"].min())
    u_max = max(df["pred_u"].max(), df["true_u"].max())

    fig = plt.figure(figsize=(16, 6))
    axes = []
    for i, (title, z, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        surf = ax.plot_trisurf(x, t, z, cmap=cmap, linewidth=0, antialiased=False)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u")
        ax.set_title(title, pad=12)
        if i < 2:
            ax.set_zlim(u_min, u_max)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        axes.append(ax)

    fig.suptitle(
        "1D Wave Equation — PINN post-processing  (c = 1)",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.15)

    # --- Rotate the surfaces for the video ----------------------------------
    n_frames = 72

    def update(i):
        azim = -60 + 360.0 * i / n_frames
        for ax in axes:
            ax.view_init(elev=28, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    mp4_path = os.path.join(BASE_DIR, "wave_1D_postproc.mp4")
    writer = FFMpegWriter(fps=24, bitrate=3200)
    anim.save(mp4_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"Wrote {mp4_path}")

    # Companion GIF (renders inline in the README on GitHub).
    gif_path = os.path.join(BASE_DIR, "wave_1D_postproc.gif")
    os.system(
        f'ffmpeg -y -i "{mp4_path}" '
        f'-vf "fps=16,scale=760:-1:flags=lanczos" "{gif_path}" -loglevel error'
    )
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()

"""Standalone waveform PNGs and a local HTML index; no external plotting service."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import numpy as np

from .train import read_json


def visualize(root, limit=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    if (root / "single/result.json").is_file():
        from .visualize_single import visualize_single
        return visualize_single(root, limit)
    recipe = read_json(root / "recipe.json")
    seed = 42 if 42 in recipe["options"]["seeds"] else min(recipe["options"]["seeds"])
    output = root / "visualizations"
    output.mkdir(exist_ok=True)
    paths = sorted((root / "jobs").glob(f"seed{seed}_fold*/predictions/full/*/traj_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    links = []
    for path in paths:
        domain = path.parent.name
        job = path.parents[3].name
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        n = len(arrays["time"])
        own_time = arrays["time"] - arrays["time"][0]
        real_time = arrays["real_time"] - arrays["real_time"][0]
        fig, axes = plt.subplots(3, 3, figsize=(16, 9), squeeze=False)
        for axis, name in enumerate(("Fx", "Fy", "Fz")):
            ax = axes[axis, 0]
            ax.plot(own_time, arrays["input_force"][:, axis], color="black", lw=1, label=f"{domain} measured")
            ax.plot(own_time, arrays["history"][:, -1, axis], lw=1, label="decoded from latent")
            ax.set_title(f"{name}: input-domain reconstruction")
            ax.set_ylabel("N")
            for column, step in enumerate((1, 5), 1):
                ax = axes[axis, column]
                indices = np.arange(n)
                ii = indices[(indices + step < arrays["ends"]) & (indices + step < arrays["real_ends"])]
                ax.plot(real_time, arrays["real_force"][:, axis], color="black", lw=1, label="Real target")
                ax.plot(real_time[ii + step], arrays["future"][ii, step - 1, axis], lw=1, label="force-only forecast")
                ax.plot(real_time[ii + step], arrays["joint"][ii, step - 1, axis], lw=1, label="force + motion forecast")
                ax.plot(real_time[ii + step], arrays["control"][ii, step - 1, axis], lw=.8, alpha=.7, label="motion control")
                ax.set_title(f"{name}: +{step} frames, plotted at target time")
            for ax in axes[axis]:
                ax.grid(alpha=.2)
                ax.set_xlabel("Recorded seconds")
                ax.legend(fontsize=7)
        fig.suptitle(f"{job} / {path.stem} / {domain} input — out-of-fold predictions")
        fig.tight_layout()
        filename = f"{job}_{path.stem}_{domain}.png"
        fig.savefig(output / filename, dpi=125)
        plt.close(fig)
        links.append((f"{job} / {path.stem} / {domain}", filename))
    body = "\n".join(f'<li><a href="{html.escape(file)}">{html.escape(label)}</a></li>' for label, file in links)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Force encoder v3 waveforms</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;line-height:1.7}</style>'
        '<h1>Force encoder v3: out-of-fold waveforms</h1>'
        '<p>Reconstruction targets the input domain. All future curves target Real forces. '
        'Forecasts are shifted to their target timestamps; none uses future observations.</p>'
        f'<p>Fixed display seed: {seed}. Full numeric results include all seeds.</p><ul>{body}</ul>')
    print(f"Waveforms: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    visualize(parser.parse_args().run_dir)

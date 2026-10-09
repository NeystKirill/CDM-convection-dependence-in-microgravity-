import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emulator.reference import Params, Solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--t-end", type=float, default=None, help="переопределить time.t_end из конфига")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--print-every", type=int, default=200)
    ap.add_argument("--out", default="out/first_flame.png")
    args = ap.parse_args()

    params = Params.from_yaml(args.config)
    t_end = args.t_end if args.t_end is not None else params.t_end
    solver = Solver(params)

    print(f"dx={params.dx:.4g} dy={params.dy:.4g} dt={params.dt:.4g} "
          f"nx={params.nx} ny={params.ny} t_end={t_end}")

    t0 = time.time()
    while solver.t < t_end:
        if args.max_steps is not None and solver.n_steps >= args.max_steps:
            break
        solver.step()
        if solver.n_steps % args.print_every == 0:
            d = solver.diagnostics()
            print(f"step={solver.n_steps:6d} t={solver.t:.4f} "
                  f"Tmax={d['Tmax']:.1f} Yf_min={d['Yf_min']:.3g} "
                  f"Yo_min={d['Yo_min']:.3g} w_max={d['w_max']:.3g} "
                  f"nan={d['has_nan']}")

    wall = time.time() - t0
    d = solver.diagnostics()
    print(f"\nГотово: {solver.n_steps} шагов за {wall:.1f} с, t={solver.t:.4f}")
    print(d)

    fields = solver.fields()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, key, title, cmap in zip(
        axes, ("T", "Yf", "Yo"),
        ("T, К", "Yf, доля топлива", "Yo, доля окислителя"),
        ("inferno", "viridis", "viridis"),
    ):
        im = ax.imshow(
            fields[key], origin="lower", aspect="auto", cmap=cmap,
            extent=[0, params.Lx, 0, params.Ly],
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel("x, м")
    fig.suptitle(f"t = {solver.t:.3f} с, шагов: {solver.n_steps}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()

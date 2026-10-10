import argparse
import time
from pathlib import Path

from emulator.common.history import Recorder
from .params import Params
from .solver import Solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="data/run_l2.h5")
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()

    params = Params.from_yaml(args.config)
    solver = Solver(params)
    recorder = Recorder(solver)

    t0 = time.time()
    while solver.t < params.t_end:
        solver.step()
        recorder.capture()
        if solver.n_steps % args.print_every == 0:
            q = solver.qoi()
            print(
                f"n={solver.n_steps:6d} t={solver.t:.4f} "
                f"Tmax={q['T_max']:.1f} z_f={q['z_f']*1000:.2f}mm "
                f"chi_rad={q['chi_rad']:.3f} Da={q['da']:.3f} regime={q['regime']}"
            )

    wall = time.time() - t0
    print(f"\nГотово: {solver.n_steps} шагов за {wall:.1f} с, t={solver.t:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recorder.to_hdf5(out_path)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()

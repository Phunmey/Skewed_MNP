import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import numpy as np

from replication_runner import run_replication_study

gdists = ["exponential", "halfnormal"]
sample_size = [300, 500, 1000, 1500, 2000]

beta_true = np.array([-0.72, 0.5, -0.2, 1.4,
                      0.05, -0.4, 1.0, 0.3,
                      0.81, 1.3, 0.3, -0.15])

delta_true = np.array([1.5, -1.1, 0.75])
gamma_true = np.array([0.3, -0.2])
psi_minus_true = np.array([[1.0 - 0.3 ** 2, 0.05],
                           [0.05, 1.0 - 0.2 ** 2]])
j, p = 4, 4


def build_cells():
    return list(itertools.product(gdists, sample_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-index", type=int, required=True, help="slurm_-array-task-id")
    ap.add_argument("--R", type=int, default=100, help="replicates per cell")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=10000)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--save-g", action="store_true", help="write per-observation g draws to disk.")
    ap.add_argument("--no-store-g", action="store_true", help="do not build the (n_samples, N).")
    ap.add_argument("--no-save-draws", action="store_true", help="skip writing <fit>_chain{c}.npz.")
    ap.add_argument("--out-root", type=str, default=os.environ.get("SCRATCH", "./multinomial_results"),
                    help="root output dir.")
    args = ap.parse_args()

    cells = build_cells()
    if not (0 <= args.cell_index < len(cells)):
        raise SystemExit(f"cell-index {args.cell_index} out of range 0..{len(cells) - 1}")

    g_dist, N = cells[args.cell_index]
    skew_id = "unconstrained"
    constrain_skewness = False

    out_dir = os.path.join(args.out_root, "smnp_multinomial", g_dist, f"result_N_{N}_{skew_id}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cell {args.cell_index}] arm={skew_id}  g_dist={g_dist}  N={N}")
    print(f"[cell {args.cell_index}] constrain_skewness={constrain_skewness}")
    print(f"[cell {args.cell_index}] R={args.R}  n_samples={args.n_samples} "
          f"burn_in={args.burn_in} thin={args.thin} n_chains={args.n_chains}")
    print(f"[cell {args.cell_index}] output - {out_dir}")

    run_replication_study(
        R=args.R, N=N, j=j, p=p,
        beta_true=beta_true,
        delta_true=delta_true,
        gamma_true=gamma_true,
        Psi_minus_true=psi_minus_true,
        g_dist=g_dist,
        constrain_skewness=constrain_skewness,
        run_kwargs=dict(n_samples=args.n_samples, burn_in=args.burn_in, n_chains=args.n_chains, thin=args.thin,
                        store_g=not args.no_store_g),
        output_dir=out_dir,
        base_seed=args.base_seed,
        save_g=args.save_g,
        save_draws=not args.no_save_draws,
        resume=True,
        skew_id=skew_id
    )

    print(f"[cell {args.cell_index}] DONE.")


if __name__ == "__main__":
    main()

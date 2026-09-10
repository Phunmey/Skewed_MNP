import argparse
import os
import shutil
import glob

from trend_plotting_file import analyze_sample_size_trends, analyze_misspecification

run_name = "smnp_multinomial"
arms = ["unconstrained"]
gdists = ["exponential", "halfnormal"]
samples_sizes = [300, 500, 1000, 1500]
j = 4


def cell_dir(scratch_root, g_dist, N, arm):
    return os.path.join(scratch_root, run_name, g_dist, f"result_N_{N}_{arm}")


def build_trends(scratch_root, keep_root):
    made_any = False
    for arm in arms:
        for g_dist in gdists:
            all_sample_results = {}
            for N in samples_sizes:
                cdir = cell_dir(scratch_root, g_dist, N, arm)
                rep_csv = os.path.join(cdir, "aggregate", "all_replicates.csv")
                if os.path.exists(rep_csv):
                    all_sample_results[f"N_{N}"] = {"sample_size": N, "output_dir": cdir,}
                else:
                    print(f"missing cell (skipped): {rep_csv}")

            if not all_sample_results:
                print(f"no finished cells for arm={arm} g_dist={g_dist}")
                continue

            agg_dir = os.path.join(scratch_root, run_name, g_dist, f"results_aggregate_{arm}")
            os.makedirs(agg_dir, exist_ok=True)
            print(f"\narm={arm} g_dist={g_dist}: {len(all_sample_results)} cells - {agg_dir}")
            analyze_sample_size_trends(all_sample_results, save_dir=agg_dir, J=j)
            made_any = True

            if keep_root:
                dest = os.path.join(keep_root, run_name, g_dist, f"results_aggregate_{arm}")
                os.makedirs(dest, exist_ok=True)
                for pattern in ("*.csv", "*.png", "*.pdf"):
                    for f in glob.glob(os.path.join(agg_dir, pattern)):
                        shutil.copy2(f, dest)

                for N in samples_sizes:
                    src_cell = os.path.join(cell_dir(scratch_root, g_dist, N, arm), "aggregate")
                    if os.path.isdir(src_cell):
                        cell_dest = os.path.join(keep_root, run_name, g_dist, f"result_N_{N}_{arm}", "aggregate")
                        os.makedirs(cell_dest, exist_ok=True)
                        for fn in ("all_replicates.csv", "summary.csv"):
                            sp = os.path.join(src_cell, fn)
                            if os.path.exists(sp):
                                shutil.copy2(sp, cell_dest)
                print(f"copied CSVs/PNGs - {dest}")

    if not made_any:
        print("nothing produced")


def build_misspecification(scratch_root, keep_root, misspec_root=None):
    misspec_sizes = [N for N in samples_sizes] # if N != 2000]
    print(f"\n scanning {scratch_root}/{run_name}/{{exponential,halfnormal}}/ "
          f"for matched + *_fitas_* cells across N={misspec_sizes}")
    agg_dir = os.path.join(scratch_root, run_name, "results_misspecification")
    df = analyze_misspecification(scratch_root, run_name=run_name, arm=arms[0],
                                  samples_sizes=misspec_sizes, save_dir=agg_dir, misspec_root=misspec_root)
    if df is None or df.empty:
        print("no cells found yet, nothing to copy")
        return

    if keep_root:
        dest = os.path.join(keep_root, run_name, "results_misspecification")
        os.makedirs(dest, exist_ok=True)
        for pattern in ("*.csv", "*.png", "*.pdf"):
            for f in glob.glob(os.path.join(agg_dir, pattern)):
                shutil.copy2(f, dest)
        print(f"copied CSV/PNG - {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-root", type=str, default=os.environ.get("SCRATCH", "./results_scratch_v2"))
    ap.add_argument("--keep-root", type=str, default=None, help="dir to copy ")
    ap.add_argument("--misspec-root", type=str, default=None)
    args = ap.parse_args()
    build_trends(args.scratch_root, args.keep_root)
    build_misspecification(args.scratch_root, args.keep_root, misspec_root=args.misspec_root)


if __name__ == "__main__":
    main()
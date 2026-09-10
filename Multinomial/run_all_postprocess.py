import argparse
import os
import subprocess
import sys
import pandas as pd

run_name = "smnp_multinomial"
arm = "unconstrained"
gdists = ["exponential", "halfnormal"]
sample_sizes = [300, 500, 1000, 1500]


def cell_dir(scratch_root, g_dist, N):
    return os.path.join(scratch_root, run_name, g_dist, f"result_N_{N}_{arm}")


def run_all(scratch_root, script_path, expected_chains):
    ran, skipped = [], []
    for g_dist in gdists:
        for N in sample_sizes:
            cdir = cell_dir(scratch_root, g_dist, N)
            if not os.path.isdir(cdir):
                print(f"missing cell (skipped): {cdir}")
                skipped.append((g_dist, N, "cell directory not found"))
                continue
            if not os.path.exists(os.path.join(cdir, "seed_manifest.csv")):
                print(f"missing seed_manifest.csv (skipped): {cdir}")
                skipped.append((g_dist, N, "seed_manifest.csv not found"))
                continue

            print(f"processing g_dist={g_dist} N={N} ...")
            result = subprocess.run(
                [sys.executable, script_path,
                 "--cell-dir", cdir,
                 "--N", str(N),
                 "--g-dist", g_dist,
                 "--expected-chains", str(expected_chains)],
                capture_output=True, text=True)
            if result.returncode != 0:
                print(f" FAILED g_dist={g_dist} N={N}:\n{result.stderr[-2000:]}")
                skipped.append((g_dist, N, f"subprocess failed: {result.stderr[-300:]}"))
                continue
            print(result.stdout[-500:])
            ran.append((g_dist, N))
    return ran, skipped


def consolidate(scratch_root, ran):
    comp_frames, block_frames, skip_frames = [], [], []
    for g_dist, N in ran:
        cdir = cell_dir(scratch_root, g_dist, N)
        outdir = os.path.join(cdir, "postprocess_recovery")
        comp_path = os.path.join(outdir, "recovery_component_summary.csv")
        block_path = os.path.join(outdir, "recovery_block_summary.csv")
        skip_path = os.path.join(outdir, "skipped_or_incomplete.csv")
        if os.path.exists(comp_path):
            comp_frames.append(pd.read_csv(comp_path))
        else:
            print(f"expected but missing: {comp_path}")
        if os.path.exists(block_path):
            block_frames.append(pd.read_csv(block_path))
        else:
            print(f"expected but missing: {block_path}")
        if os.path.exists(skip_path):
            df = pd.read_csv(skip_path)
            if len(df):
                df.insert(0, "N", N)
                df.insert(0, "g_dist", g_dist)
                skip_frames.append(df)

    out_dir = os.path.join(scratch_root, run_name, "recovery_all_cells")
    os.makedirs(out_dir, exist_ok=True)

    if comp_frames:
        pd.concat(comp_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "recovery_component_summary_all.csv"), index=False)
    if block_frames:
        pd.concat(block_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "recovery_block_summary_all.csv"), index=False)
    if skip_frames:
        pd.concat(skip_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "skipped_or_incomplete_all.csv"), index=False)

    return out_dir, len(comp_frames), len(block_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-root", type=str, default=os.environ.get("SCRATCH", "./results_scratch_v2"))
    ap.add_argument("--script-path", type=str, default="postprocess_smnp_recovery.py")
    ap.add_argument("--expected-chains", type=int, default=4)
    args = ap.parse_args()

    ran, skipped = run_all(args.scratch_root, args.script_path, args.expected_chains)
    print(f"\n{len(ran)} cells processed, {len(skipped)} skipped")

    out_dir, n_comp, n_block = consolidate(args.scratch_root, ran)
    print(f"consolidated {n_comp} cells' component summaries, "
          f"{n_block} cells' block summaries - {out_dir}")

    if skipped:
        print("\nSkipped cells:")
        for g_dist, N, reason in skipped:
            print(f"g_dist={g_dist} N={N}: {reason}")


if __name__ == "__main__":
    main()

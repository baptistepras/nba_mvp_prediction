import os
import subprocess
from train_models import PIPELINE_GROUPS, MAX_YEAR

# Configuration
PIPELINE_SCRIPTS = {
    "selected1956": "build_selectedStats_from1956.py",
    "selected1980": "build_selectedStats_from1980.py",
    "all1956": "build_allStats_from1956.py",
    "all1980": "build_allStats_from1980.py"
}


def main(pipelines=["all1980"], start=1980, end=2025):   
    # Resolve pipelines to run
    pipelines_to_run = []
    for p in pipelines:
        if p in PIPELINE_GROUPS:
            pipelines_to_run.extend(PIPELINE_GROUPS[p])
        elif p in PIPELINE_SCRIPTS:
            pipelines_to_run.append(p)
        else:
            print(f"[ERROR] Unknown pipeline or group: '{p}', skipping.")

    # Deduplicate
    pipelines_to_run = list(dict.fromkeys(pipelines_to_run))

    # Run
    for pipeline in pipelines_to_run:
        print()
        script_name = PIPELINE_SCRIPTS[pipeline]

        # Determine correct year range for this pipeline
        pipeline_min_year = 1956 if "1956" in pipeline else 1980

        year_start = start
        year_end = end

        if year_start < pipeline_min_year:
            print(f"[WARN] {pipeline}: start {year_start} < {pipeline_min_year}, forcing to {pipeline_min_year}.")
            year_start = pipeline_min_year
        if year_end > MAX_YEAR:
            print(f"[WARN] {pipeline}: end {year_end} > {MAX_YEAR}, forcing to {MAX_YEAR}.")
            year_end = MAX_YEAR
        if year_start > year_end:
            print(f"[ERROR] {pipeline}: start {year_start} > end {year_end}, skipping.")
            continue

        # Build command
        cmd = f"python scripts_data_process/{script_name} --start {year_start} --end {year_end}"

        print(f"\n[INFO] Running pipeline: {pipeline} -> {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    print("\n[INFO] All selected pipelines finished.")
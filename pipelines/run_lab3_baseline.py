import subprocess
import sys

def execute_pipeline():
    print("[INFO] Starting Lab 3: Baseline ML Pipeline")

    scripts = [
        "src.preprocess",
        "src.train",
        "src.evaluate"
    ]

    for script in scripts:
        print(f"\n[INFO] Executing {script}...")
        result = subprocess.run(
            [sys.executable, "-m", script]
        )

        if result.returncode != 0:
            print(f"[ERROR] {script} failed.")
            sys.exit(1)

    print("[SUCCESS] Lab 3 Pipeline completed!")

if __name__ == "__main__":
    execute_pipeline()

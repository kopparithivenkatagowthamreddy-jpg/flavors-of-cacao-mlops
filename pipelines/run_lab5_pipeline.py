import subprocess
import sys

def execute_pipeline():
    print("[INFO] Starting Lab 5: Production Data Pipeline")

    scripts = [
        "src.validate_data",
        "src.preprocess",
        "src.validate_outputs"
    ]

    for script in scripts:
        print(f"\n[INFO] Executing {script}...")
        result = subprocess.run(
            [sys.executable, "-m", script]
        )

        if result.returncode != 0:
            print(f"[ERROR] {script} failed.")
            sys.exit(1)

    print("[SUCCESS] Lab 5 Production Pipeline completed!")

if __name__ == "__main__":
    execute_pipeline()

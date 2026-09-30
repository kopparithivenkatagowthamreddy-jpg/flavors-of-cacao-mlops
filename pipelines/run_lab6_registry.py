import subprocess
import sys

def execute_pipeline():
    print("[INFO] Starting Lab 6: Model Registry Workflow")

    scripts = [
        "src.train_registry",
        "src.automate_lifecycle",
        "src.generate_registry_report"
    ]

    for script in scripts:
        print(f"\n[INFO] Executing {script}...")
        result = subprocess.run(
            [sys.executable, "-m", script]
        )

        if result.returncode != 0:
            print(f"[ERROR] {script} failed.")
            sys.exit(1)

    print("[SUCCESS] Lab 6 workflow completed.")
    print("[INFO] Check the registry report for publication status.")

if __name__ == "__main__":
    execute_pipeline()

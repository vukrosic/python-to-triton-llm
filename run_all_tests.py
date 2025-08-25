import glob
import subprocess
import sys

def main():
    """
    Runs all combined_implementation_*.py files in the data/ directory
    as separate processes and reports their success or failure based on
    exit codes.
    """
    # Use sorted glob to ensure a consistent order of execution
    test_files = sorted(glob.glob("data/combined_implementation*.py"))
    
    if not test_files:
        print("No test files found in 'data/' directory.")
        sys.exit(0)

    print(f"Found {len(test_files)} test files to verify.")
    
    results = {}
    all_passed = True

    for file_path in test_files:
        print(f"\n--- Running: {file_path} ---")
        try:
            # Use sys.executable to ensure we run with the same Python interpreter
            process = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                check=False, # Don't raise an exception for non-zero exit codes
                timeout=60   # Safety timeout of 60 seconds
            )
            
            # Print the output from the script
            if process.stdout:
                print(process.stdout.strip())
            if process.stderr:
                print("--- Stderr ---")
                print(process.stderr.strip())

            if process.returncode == 0:
                results[file_path] = "Pass"
                print(f"✅ PASSED: {file_path}")
            else:
                results[file_path] = "Fail"
                all_passed = False
                print(f"❌ FAILED: {file_path} (Exit Code: {process.returncode})")

        except subprocess.TimeoutExpired:
            results[file_path] = "Timeout"
            all_passed = False
            print(f"❌ TIMEOUT: {file_path} took longer than 60 seconds.")
        except Exception as e:
            results[file_path] = "Error"
            all_passed = False
            print(f"❌ ERROR: An exception occurred while running {file_path}: {e}")

    print("\n--- Overall Test Summary ---")
    for file_path, status in results.items():
        print(f"- {file_path}: {status}")
    
    if not all_passed:
        print("\nSome tests failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
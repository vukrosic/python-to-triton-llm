import glob
import subprocess
import sys

def run_test_on_file(file_path):
    """
    Runs a single test file and returns the result.
    
    Returns:
        tuple: (file_path, status, output)
        - file_path: the path to the test file
        - status: "Pass", "Fail", "Timeout", or "Error"
        - output: the stdout/stderr output from the test
    """
    print(f"--- Running: {file_path} ---")
    try:
        # Use sys.executable to ensure we run with the same Python interpreter
        process = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            check=False, # Don't raise an exception for non-zero exit codes
            timeout=60   # Safety timeout of 60 seconds
        )
        
        # Collect the output
        output = ""
        if process.stdout:
            output += process.stdout.strip()
        if process.stderr:
            output += "\n--- Stderr ---\n" + process.stderr.strip()

        if process.returncode == 0:
            status = "Pass"
            print(f"✅ PASSED: {file_path}")
        else:
            status = "Fail"
            print(f"❌ FAILED: {file_path} (Exit Code: {process.returncode})")

        return file_path, status, output

    except subprocess.TimeoutExpired:
        status = "Timeout"
        output = "Test took longer than 60 seconds."
        print(f"❌ TIMEOUT: {file_path} took longer than 60 seconds.")
        return file_path, status, output
    except Exception as e:
        status = "Error"
        output = f"An exception occurred: {e}"
        print(f"❌ ERROR: An exception occurred while running {file_path}: {e}")
        return file_path, status, output

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
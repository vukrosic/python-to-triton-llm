import os
import subprocess

def run_and_count_passed_tests():
    # Get the absolute path to the data directory
    # This assumes the script is run from the project root
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # List all Python files in the data directory
    python_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".py")]
    
    passed_count = 0
    total_tests = 0

    print("Starting test execution and counting...")
    print("-" * 40)

    for file_path in sorted(python_files): # Sort for consistent output
        total_tests += 1
        file_name = os.path.basename(file_path)
        print(f"Running test for {file_name}...")
        
        try:
            # Execute the Python file. This requires a CUDA-enabled environment
            # and PyTorch/Triton installed.
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                check=False # Do not raise an exception for non-zero exit codes
            )
            
            # Check for "PASSED" in the output
            if "Correctness test PASSED!" in result.stdout:
                passed_count += 1
                print(f"  {file_name}: PASSED")
            else:
                print(f"  {file_name}: FAILED")
                print("---- STDOUT ----")
                print(result.stdout)
                print("---- STDERR ----")
                print(result.stderr)
                print("----------------")
                
        except FileNotFoundError:
            print(f"  Error: 'python' command not found. Make sure Python is installed and in your PATH.")
            print(f"  Skipping {file_name}.")
        except Exception as e:
            print(f"  An unexpected error occurred while running {file_name}: {e}")
            print("---- STDOUT ----")
            print(result.stdout if 'result' in locals() else "No stdout captured")
            print("---- STDERR ----")
            print(result.stderr if 'result' in locals() else "No stderr captured")
            print("----------------")
        print("-" * 40)

    print("\nTest Summary:")
    print(f"Total tests attempted: {total_tests}")
    print(f"Tests PASSED: {passed_count}")
    print(f"Tests FAILED: {total_tests - passed_count}")

if __name__ == "__main__":
    run_and_count_passed_tests()

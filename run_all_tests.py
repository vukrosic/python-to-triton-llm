import glob
import importlib.util
import torch
import concurrent.futures
import traceback

def run_test_on_file(file_path):
    """
    Dynamically imports and tests the python and triton functions from a given file.
    Returns a tuple of (file_path, status, message).
    """
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        python_fn = None
        triton_fn = None
        for name in dir(test_module):
            if name.startswith("python_"):
                python_fn = getattr(test_module, name)
            elif name.startswith("triton_"):
                triton_fn = getattr(test_module, name)

        if python_fn is None or triton_fn is None:
            return file_path, "Fail", "Could not find python/triton functions."

        # Define test parameters based on the file being tested.
        # This is a simple way to manage parameters for different tests.
        if "combined_implementation_2.py" in file_path:
            params = {'start': 0, 'end': 64}
        elif "combined_implementation.py" in file_path:
            params = {'start': 0, 'end': 256, 'shape': (16, 16)}
        else:
            return file_path, "Skip", "No test parameters defined for this file."

        python_result = python_fn(**params)
        
        if not torch.cuda.is_available():
            return file_path, "Skip", "CUDA not available."
            
        triton_result = triton_fn(**params)

        are_close = torch.allclose(python_result.cuda(), triton_result)
        if are_close:
            return file_path, "Pass", ""
        else:
            return file_path, "Fail", "Results are not close."

    except Exception:
        return file_path, "Error", traceback.format_exc()

def main():
    test_files = glob.glob("data/combined_implementation*.py")
    
    if not test_files:
        print("No test files found in 'data/' directory.")
        return

    print(f"Found {len(test_files)} test files to verify.")

    results = []
    # Using sequential execution by default to avoid GPU memory issues.
    # Replace with the commented out ProcessPoolExecutor for parallel execution.
    print("Running tests sequentially...")
    for file_path in test_files:
        result = run_test_on_file(file_path)
        results.append(result)

    # --- Example of Parallel Execution ---
    # print("Running tests in parallel...")
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = list(executor.map(run_test_on_file, test_files))

    print("\n--- Test Summary ---")
    passed = 0
    for file_path, status, message in results:
        print(f"- {file_path}: {status}")
        if message and status != "Skip":
            print(f"  Info: {message}")
        if status == "Pass":
            passed += 1
    
    print(f"\n{passed}/{len(results)} tests passed.")

if __name__ == "__main__":
    main()

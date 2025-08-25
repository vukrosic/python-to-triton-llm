import csv
import glob
import traceback
from run_all_tests import run_test_on_file

def extract_training_pair(file_path):
    """Extracts function bodies using comment markers."""
    print(f"--- Processing: {file_path}")
    try:
        with open(file_path, 'r') as f:
            source_code = f.read()

        # Extract Python body
        py_start_marker = "# PYTHON_BODY_START"
        py_end_marker = "# PYTHON_BODY_END"
        py_start_index = source_code.find(py_start_marker)
        py_end_index = source_code.find(py_end_marker)

        if py_start_index == -1 or py_end_index == -1:
            print("  - FAILED: Could not find Python body markers.")
            return None, None
            
        python_body = source_code[py_start_index + len(py_start_marker) : py_end_index].strip()

        # Extract Triton kernel body
        triton_start_marker = "# TRITON_KERNEL_BODY_START"
        triton_end_marker = "# TRITON_KERNEL_BODY_END"
        triton_start_index = source_code.find(triton_start_marker)
        triton_end_index = source_code.find(triton_end_marker)

        if triton_start_index == -1 or triton_end_index == -1:
            print("  - FAILED: Could not find Triton kernel body markers.")
            return None, None

        triton_body = source_code[triton_start_index + len(triton_start_marker) : triton_end_index].strip()
        
        print("  - SUCCESS: Extracted both bodies using markers.")
        return python_body, triton_body

    except Exception as e:
        print(f"  - ERROR: An exception occurred: {e}")
        traceback.print_exc()
        return None, None

def main():
    passed_files = []
    test_files = glob.glob("data/combined_implementation*.py")
    
    print("Running tests to find passing files...")
    for file_path in test_files:
        _, status, _ = run_test_on_file(file_path)
        if status == "Pass":
            passed_files.append(file_path)
    
    print(f"\nFound {len(passed_files)} passed files.")

    if not passed_files:
        print("No passed files to extract data from.")
        return

    training_data = []
    for file_path in passed_files:
        python_body, triton_body = extract_training_pair(file_path)
        if python_body and triton_body:
            training_data.append({
                'python_function_body': python_body,
                'triton_kernel_body': triton_body
            })

    csv_file = "training_data.csv"
    print(f"\nWriting extracted data to {csv_file}...")
    if not training_data:
        print("WARNING: No data was extracted. The CSV file will be empty.")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['python_function_body', 'triton_kernel_body'])
        writer.writeheader()
        writer.writerows(training_data)
        
    print(f"Done. You can find the training data in {csv_file}")

if __name__ == "__main__":
    main()

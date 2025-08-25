import csv
import glob
import importlib.util
import inspect
import ast
import torch
import traceback
from run_all_tests import run_test_on_file

def get_function_body(func):
    """Extracts the indented body of a function from its source code."""
    try:
        source_lines = inspect.getsource(func).splitlines()
        def_line_index = -1
        for i, line in enumerate(source_lines):
            if line.strip().startswith('def '):
                def_line_index = i
                break
        
        if def_line_index == -1:
            return ""

        body_lines = source_lines[def_line_index + 1:]
        
        if not body_lines:
            return ""
        
        indentation = len(body_lines[0]) - len(body_lines[0].lstrip(' ')) 
        unindented_body = [line[indentation:] for line in body_lines]
        
        return "\n".join(unindented_body)

    except (TypeError, OSError):
        return ""

def extract_training_pair(file_path):
    """Extracts the python and triton function bodies from a file."""
    print(f"--- Processing: {file_path}")
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)

        python_fn_name = None
        triton_kernel_name = None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("python_"):
                    python_fn_name = node.name
                
                for decorator in node.decorator_list:
                    decorator_source = ast.get_source_segment(source, decorator)
                    if 'triton.jit' in decorator_source:
                        triton_kernel_name = node.name
                        break
        
        print(f"  - Found Python function name: {python_fn_name}")
        print(f"  - Found Triton kernel name: {triton_kernel_name}")

        if not python_fn_name or not triton_kernel_name:
            print("  - FAILED: Could not find both function names via AST parsing.")
            return None, None

        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        python_fn = getattr(module, python_fn_name)
        triton_kernel = getattr(module, triton_kernel_name)

        python_body = get_function_body(python_fn)
        triton_body = get_function_body(triton_kernel)

        if not python_body:
            print("  - FAILED: Could not extract Python function body.")
        if not triton_body:
            print("  - FAILED: Could not extract Triton kernel body.")

        if python_body and triton_body:
            lines = python_body.splitlines()
            if lines and lines[-1].strip().startswith("return "):
                python_body = "\n".join(lines[:-1]).strip()
            
            print("  - SUCCESS: Extracted both bodies.")
            return python_body, triton_body.strip()
        
        return None, None

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
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
        source = inspect.getsource(func)
        lines = source.splitlines()
        
        # Find the line with 'def ' - could be after decorators
        def_line_index = -1
        for i, line in enumerate(lines):
            if 'def ' in line and '(' in line:  # More robust check
                def_line_index = i
                break
        
        if def_line_index == -1:
            print(f"    - Could not find def line")
            return ""
        
        # Handle multi-line function signatures
        # Find where the function signature ends (look for ':')
        signature_end_index = def_line_index
        for i in range(def_line_index, len(lines)):
            if ':' in lines[i]:
                signature_end_index = i
                break
        
        # Get body lines (everything after the signature)
        body_lines = lines[signature_end_index + 1:]
        
        if not body_lines:
            print(f"    - No body lines found")
            return ""
        
        # Find the indentation of the first non-empty line in the body
        first_non_empty_idx = -1
        for i, line in enumerate(body_lines):
            if line.strip():  # Non-empty line
                first_non_empty_idx = i
                break
        
        if first_non_empty_idx == -1:
            print(f"    - No non-empty body lines found")
            return ""
        
        # Calculate indentation from the first non-empty body line
        first_non_empty = body_lines[first_non_empty_idx]
        indentation = len(first_non_empty) - len(first_non_empty.lstrip())
        
        # Remove the indentation from all body lines
        unindented_body = []
        for line in body_lines:
            if line.strip():  # Non-empty line
                if len(line) >= indentation and line[:indentation].isspace():
                    unindented_body.append(line[indentation:])
                else:
                    # Line has less indentation than expected, just strip what's there
                    unindented_body.append(line.lstrip())
            else:
                unindented_body.append('')  # Keep empty lines
        
        # Remove trailing empty lines
        while unindented_body and not unindented_body[-1].strip():
            unindented_body.pop()
        
        return "\n".join(unindented_body)
        
    except Exception as e:
        print(f"    - Error in get_function_body: {e}")
        import traceback
        traceback.print_exc()
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
                    if decorator_source and 'triton.jit' in decorator_source:
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
            # Remove the return statement from Python body if present
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
    else:
        print(f"Successfully extracted {len(training_data)} training pairs.")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['python_function_body', 'triton_kernel_body'])
        writer.writeheader()
        writer.writerows(training_data)
        
    print(f"Done. You can find the training data in {csv_file}")

if __name__ == "__main__":
    main()
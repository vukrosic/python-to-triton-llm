import csv
import glob
import importlib.util
import inspect
import ast
import torch
from run_all_tests import run_test_on_file

def get_function_body(func):
    """Extracts the body of a function from its source code."""
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
                is_triton_jit = False
                if isinstance(decorator, ast.Attribute) and decorator.attr == 'jit':
                    if isinstance(decorator.value, ast.Name) and decorator.value.id == 'triton':
                        is_triton_jit = True
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'jit':
                     if hasattr(decorator.func, 'value') and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == 'triton':
                          is_triton_jit = True
                if is_triton_jit:
                    triton_kernel_name = node.name
                    break
    
    if not python_fn_name or not triton_kernel_name:
        return None, None

    spec = importlib.util.spec_from_file_location("temp_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    python_fn = getattr(module, python_fn_name)
    triton_kernel = getattr(module, triton_kernel_name)

    python_body = get_function_body(python_fn)
    if python_body:
        lines = python_body.splitlines()
        if lines and lines[-1].strip().startswith("return "):
            python_body = "\n".join(lines[:-1]).strip()

    triton_body = get_function_body(triton_kernel).strip()

    return python_body, triton_body

def main():
    passed_files = []
    test_files = glob.glob("data/combined_implementation*.py")
    
    print("Running tests to find passing files...")
    for file_path in test_files:
        _, status, _ = run_test_on_file(file_path)
        if status == "Pass":
            passed_files.append(file_path)
    
    print(f"Found {len(passed_files)} passed files.")

    if not passed_files:
        print("No passed files to extract data from.")
        return

    training_data = []
    for file_path in passed_files:
        print(f"Extracting data from: {file_path}")
        python_body, triton_body = extract_training_pair(file_path)
        if python_body and triton_body:
            training_data.append({
                'python_function_body': python_body,
                'triton_kernel_body': triton_body
            })

    csv_file = "training_data.csv"
    print(f"Writing extracted data to {csv_file}...")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['python_function_body', 'triton_kernel_body'])
        writer.writeheader()
        writer.writerows(training_data)
        
    print(f"Done. You can find the training data in {csv_file}")

if __name__ == "__main__":
    main()

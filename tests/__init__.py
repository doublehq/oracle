import os
import glob
import importlib.util

# Get the directory containing this __init__.py file
test_dir = os.path.dirname(os.path.abspath(__file__))

# Find all test_*.py files
test_files = glob.glob(os.path.join(test_dir, "test_*.py"))

# Import each test module
for test_file in test_files:
    # Get the module name from the file path
    module_name = os.path.splitext(os.path.basename(test_file))[0]
    
    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Add it to this package's globals
    globals()[module_name] = module

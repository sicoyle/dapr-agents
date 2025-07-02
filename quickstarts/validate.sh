#!/bin/bash

# Quickstart Validation Script
# Validates a quickstart directory for common issues

set -e  # Exit on any error

QUICKSTART_DIR="$1"
if [ -z "$QUICKSTART_DIR" ]; then
    echo "❌ Usage: $0 <quickstart-directory>"
    exit 1
fi

if [ ! -d "$QUICKSTART_DIR" ]; then
    echo "❌ Directory $QUICKSTART_DIR does not exist"
    exit 1
fi

echo "🔍 Validating quickstart: $QUICKSTART_DIR"
cd "$QUICKSTART_DIR"

# 1. Check for README.md
if [ ! -f "README.md" ]; then
    echo "❌ Missing README.md"
    exit 1
fi
echo "✅ README.md exists"

# 2. Validate Python syntax for all .py files
python_files=$(find . -name "*.py" -type f)
if [ -n "$python_files" ]; then
    echo "🐍 Checking Python syntax..."
    for py_file in $python_files; do
        if ! python -m py_compile "$py_file" 2>/dev/null; then
            echo "❌ Python syntax error in: $py_file"
            exit 1
        fi
    done
    echo "✅ All Python files have valid syntax"
else
    echo "⚠️  No Python files found"
fi

# 3. Check requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "📦 Checking requirements.txt..."
    
    # Check for empty or invalid requirements
    if [ ! -s "requirements.txt" ]; then
        echo "⚠️  requirements.txt is empty"
    else
        # Basic validation of requirements format
        if grep -q "^[a-zA-Z]" requirements.txt; then
            echo "✅ requirements.txt format looks valid"
        else
            echo "❌ requirements.txt format appears invalid"
            exit 1
        fi
    fi
else
    echo "⚠️  No requirements.txt found"
fi

# 4. Check for components directory and YAML files
if [ -d "components" ]; then
    echo "🔧 Checking Dapr components..."
    yaml_files=$(find components -name "*.yaml" -o -name "*.yml" 2>/dev/null)
    if [ -n "$yaml_files" ]; then
        echo "✅ Found Dapr component files"
        # Basic YAML syntax check if python is available
        for yaml_file in $yaml_files; do
            if command -v python3 >/dev/null; then
                if ! python3 -c "import yaml; yaml.safe_load(open('$yaml_file'))" 2>/dev/null; then
                    echo "❌ YAML syntax error in: $yaml_file"
                    exit 1
                fi
            fi
        done
        echo "✅ All YAML files have valid syntax"
    else
        echo "⚠️  No component YAML files found in components directory"
    fi
else
    echo "⚠️  No components directory found"
fi

# 5. Run basic import test for main Python files
if [ -n "$python_files" ]; then
    echo "🧪 Testing basic imports..."
    for py_file in $python_files; do
        # Skip if file name suggests it's not meant to be imported
        basename_file=$(basename "$py_file" .py)
        if [[ "$basename_file" == *"test"* ]] || [[ "$basename_file" == *"example"* ]]; then
            continue
        fi
        
        # Try to do a basic import test (just compilation, not execution)
        if ! python -c "import ast; ast.parse(open('$py_file').read())" 2>/dev/null; then
            echo "❌ Failed to parse: $py_file"
            exit 1
        fi
    done
    echo "✅ All Python files can be parsed"
fi

# 6. Check README for basic content
if grep -q "# " README.md; then
    echo "✅ README.md has headers"
else
    echo "⚠️  README.md might be missing proper headers"
fi

echo "🎉 Validation completed successfully for: $QUICKSTART_DIR"

#!/usr/bin/env python3
"""
Simple test runner for dapr-agents tests.
"""
import sys
import subprocess
import os


def run_tests():
    """Run the test suite."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    
    # Change to project root
    os.chdir(project_root)
    
    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=dapr_agents",
        "--cov-report=term-missing",
        "--cov-report=html"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(run_tests()) 
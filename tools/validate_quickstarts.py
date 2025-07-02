#!/usr/bin/env python3

"""
Quickstart Validation Tool

Validates all quickstart directories, optionally with isolated virtual environments.
This replaces the complex bash logic that was in the Makefile.

Usage:
    python tools/validate_quickstarts.py [--isolated] [--quickstart <name>]
    
Options:
    --isolated    Create isolated virtual environments for each quickstart
    --quickstart  Validate only a specific quickstart
    --help        Show this help message
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class QuickstartValidator:
    """Handles validation of quickstart directories."""
    
    def __init__(self, use_isolated_envs: bool = False):
        self.use_isolated_envs = use_isolated_envs
        self.project_root = Path(__file__).parent.parent
        self.quickstarts_dir = self.project_root / "quickstarts"
        
        if not self.quickstarts_dir.exists():
            raise FileNotFoundError(f"Quickstarts directory not found: {self.quickstarts_dir}")
    
    def discover_quickstarts(self) -> List[str]:
        """Discover all quickstart directories."""
        quickstarts = []
        for item in self.quickstarts_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                quickstarts.append(item.name)
        return sorted(quickstarts)
    
    def validate_quickstart(self, quickstart_name: str) -> bool:
        """Validate a single quickstart directory."""
        quickstart_path = self.quickstarts_dir / quickstart_name
        
        if not quickstart_path.exists():
            print(f"❌ Quickstart directory not found: {quickstart_name}")
            return False
        
        print(f"\n=== Validating {quickstart_name} ===")
        
        if self.use_isolated_envs and (quickstart_path / "requirements.txt").exists():
            return self._validate_with_isolated_env(quickstart_path)
        else:
            return self._validate_with_current_env(quickstart_path)
    
    def _validate_with_isolated_env(self, quickstart_path: Path) -> bool:
        """Validate quickstart with an isolated virtual environment."""
        venv_path = quickstart_path / ".venv"
        
        try:
            print(f"Creating virtual environment for {quickstart_path.name}...")
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True)
            
            # Determine the activate script path
            if sys.platform == "win32":
                activate_script = venv_path / "Scripts" / "activate"
                python_executable = venv_path / "Scripts" / "python"
            else:
                activate_script = venv_path / "bin" / "activate"
                python_executable = venv_path / "bin" / "python"
            
            print(f"Installing requirements for {quickstart_path.name}...")
            subprocess.run([
                str(python_executable), "-m", "pip", "install", 
                "-r", "requirements.txt"
            ], cwd=quickstart_path, check=True, capture_output=True)
            
            print(f"Running validation script for {quickstart_path.name}...")
            result = subprocess.run([
                "bash", "../validate.sh", quickstart_path.name
            ], cwd=self.quickstarts_dir, capture_output=False)
            
            success = result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during isolated environment validation: {e}")
            success = False
        
        finally:
            # Clean up virtual environment
            if venv_path.exists():
                print(f"Cleaning up virtual environment for {quickstart_path.name}...")
                if sys.platform == "win32":
                    subprocess.run(["rmdir", "/s", "/q", str(venv_path)], shell=True, capture_output=True)
                else:
                    subprocess.run(["rm", "-rf", str(venv_path)], capture_output=True)
        
        return success
    
    def _validate_with_current_env(self, quickstart_path: Path) -> bool:
        """Validate quickstart with the current environment."""
        print(f"Running validation script for {quickstart_path.name}...")
        result = subprocess.run([
            "bash", "./validate.sh", quickstart_path.name
        ], cwd=self.quickstarts_dir, capture_output=False)
        
        return result.returncode == 0
    
    def validate_all(self, specific_quickstart: Optional[str] = None) -> bool:
        """Validate all quickstarts or a specific one."""
        if specific_quickstart:
            quickstarts = [specific_quickstart]
        else:
            quickstarts = self.discover_quickstarts()
        
        print(f"🧪 Validating {len(quickstarts)} quickstart(s)...")
        if self.use_isolated_envs:
            print("📦 Using isolated virtual environments")
        else:
            print("🔄 Using current environment")
        
        failed_quickstarts = []
        
        for quickstart in quickstarts:
            try:
                success = self.validate_quickstart(quickstart)
                if success:
                    print(f"✅ {quickstart} validated successfully")
                else:
                    print(f"❌ {quickstart} validation failed")
                    failed_quickstarts.append(quickstart)
                
                # Brief pause between validations
                if len(quickstarts) > 1:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Validation interrupted by user")
                return False
            except Exception as e:
                print(f"❌ Unexpected error validating {quickstart}: {e}")
                failed_quickstarts.append(quickstart)
        
        # Summary
        if failed_quickstarts:
            print(f"\n❌ Validation failed for: {', '.join(failed_quickstarts)}")
            return False
        else:
            print(f"\n🎉 All quickstart validations completed successfully!")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Dapr Agents quickstart directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/validate_quickstarts.py
    python tools/validate_quickstarts.py --isolated
    python tools/validate_quickstarts.py --quickstart 01-hello-world
    python tools/validate_quickstarts.py --isolated --quickstart 01-hello-world
        """
    )
    
    parser.add_argument(
        "--isolated", 
        action="store_true",
        help="Create isolated virtual environments for quickstarts with requirements.txt"
    )
    
    parser.add_argument(
        "--quickstart",
        type=str,
        help="Validate only the specified quickstart directory"
    )
    
    args = parser.parse_args()
    
    try:
        validator = QuickstartValidator(use_isolated_envs=args.isolated)
        success = validator.validate_all(specific_quickstart=args.quickstart)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
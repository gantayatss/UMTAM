#!/usr/bin/env python3
"""
Auto-fix UMTAM imports
======================

Run this script to automatically fix import statements in your existing installation.

Usage:
    python fix_imports.py
"""

import os
from pathlib import Path


def fix_file_imports(filepath, is_src_file=False):
    """Fix imports in a single file."""
    
    # Read current content
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already fixed
    if 'from src.umtam_optimizer import' in content or 'sys.path.insert' in content:
        print(f"  ✅ {filepath.name} - already fixed")
        return False
    
    # Skip if it's in src/ directory (doesn't need fixing)
    if is_src_file:
        print(f"  ➡️  {filepath.name} - skipping (src file)")
        return False
    
    # Prepare new content
    new_lines = []
    added_path_setup = False
    in_imports = False
    
    for line in content.split('\n'):
        # Add path setup after docstring
        if not added_path_setup and (
            line.startswith('import ') or 
            line.startswith('from ')
        ) and not in_imports:
            new_lines.append('import sys')
            new_lines.append('from pathlib import Path')
            new_lines.append('# Add parent directory to path to import from src/')
            new_lines.append('sys.path.insert(0, str(Path(__file__).parent.parent))')
            new_lines.append('')
            added_path_setup = True
            in_imports = True
        
        # Skip existing sys/Path imports if we added them
        if added_path_setup and (
            line.strip() == 'import sys' or
            line.strip() == 'from pathlib import Path'
        ):
            continue
        
        # Fix umtam imports
        line = line.replace('from umtam_optimizer import', 'from src.umtam_optimizer import')
        line = line.replace('from umtam_diagnostics import', 'from src.umtam_diagnostics import')
        line = line.replace('from fineweb_loader import', 'from src.fineweb_loader import')
        
        new_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"  ✅ {filepath.name} - fixed!")
    return True


def main():
    """Fix all UMTAM files."""
    
    print("="*60)
    print("UMTAM Import Auto-Fix")
    print("="*60)
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"\nWorking directory: {current_dir}")
    
    # Check if we're in the right place
    if not (current_dir / 'src').exists():
        print("\n❌ Error: 'src/' directory not found!")
        print("   Please run this script from your project root directory.")
        print("   Expected structure:")
        print("     umtam-code/")
        print("     ├── src/")
        print("     ├── tests/")
        print("     ├── real/")
        print("     └── ...")
        return 1
    
    files_fixed = 0
    
    # Fix files in tests/
    print("\n📁 Fixing tests/...")
    tests_dir = current_dir / 'tests'
    if tests_dir.exists():
        for py_file in tests_dir.glob('*.py'):
            if fix_file_imports(py_file):
                files_fixed += 1
    else:
        print("  ⚠️  tests/ directory not found")
    
    # Fix files in real/
    print("\n📁 Fixing real/...")
    real_dir = current_dir / 'real'
    if real_dir.exists():
        for py_file in real_dir.glob('*.py'):
            if fix_file_imports(py_file):
                files_fixed += 1
    else:
        print("  ⚠️  real/ directory not found")
    
    # Fix files in examples/
    print("\n📁 Fixing examples/...")
    examples_dir = current_dir / 'examples'
    if examples_dir.exists():
        for py_file in examples_dir.glob('*.py'):
            if fix_file_imports(py_file):
                files_fixed += 1
    else:
        print("  ⚠️  examples/ directory not found")
    
    # Summary
    print("\n" + "="*60)
    if files_fixed > 0:
        print(f"✅ Fixed {files_fixed} file(s)!")
        print("\nNow test it:")
        print("  python tests/test_umtam.py")
    else:
        print("✅ All files already fixed or no files to fix!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

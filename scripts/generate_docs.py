#!/usr/bin/env python3
"""
Generate HTML documentation from docstrings using pdoc3.

Usage:
    python scripts/generate_docs.py
"""
import subprocess
import sys
from pathlib import Path


def generate_documentation():
    """
    Generate HTML documentation for the project using pdoc3.
    """
    # Output directory for documentation
    output_dir = Path("docs/html")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Module to document
    module = "app"

    print(f"Generating documentation for {module}...")

    try:
        # Run pdoc3 to generate HTML documentation
        cmd = [
            "pdoc3",
            "--html",
            "--output-dir", str(output_dir),
            "--force",
            module
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("Documentation generated successfully!")
        print(f"Output directory: {output_dir.absolute()}")
        print(f"\nTo view the documentation, open: {output_dir.absolute()}/app/index.html")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1
    except FileNotFoundError:
        print("Error: pdoc3 not found. Install it with: pip install pdoc3")
        return 1


if __name__ == "__main__":
    sys.exit(generate_documentation())

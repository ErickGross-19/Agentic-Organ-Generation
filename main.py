#!/usr/bin/env python3
"""
Organ Generator - Main Entry Point

This is the main entry point for the Organ Generator executable.
Run this file directly or use: python -m gui

Usage:
    python main.py          # Launch GUI
    python main.py --help   # Show help
    python main.py --cli    # Run CLI mode instead
"""

import sys
import argparse


def main():
    """Main entry point for Organ Generator."""
    parser = argparse.ArgumentParser(
        prog="OrganGenerator",
        description="Organ Generator - LLM-driven automated design of 3D vascular organ structures",
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode instead of GUI",
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    
    args = parser.parse_args()
    
    if args.version:
        try:
            from gui import __version__
            print(f"Organ Generator v{__version__}")
        except ImportError:
            print("Organ Generator v1.0.0")
        print("https://github.com/ErickGross-19/Agentic-Organ-Generation")
        return 0
    
    if args.cli:
        from automation.cli import main as cli_main
        return cli_main()
    
    try:
        from gui import launch_gui
        launch_gui()
        return 0
    except ImportError as e:
        print(f"Error: Could not import GUI module: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

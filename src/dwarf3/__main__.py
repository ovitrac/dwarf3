"""
Allow running dwarf3 as a module: python -m dwarf3

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

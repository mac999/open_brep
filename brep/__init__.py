"""
brep - A from-scratch B-Rep (Boundary Representation) solid modeling kernel.

Layered architecture (see PRD.md):
    Layer 1  topology.py / euler_ops.py  - Half-Edge data structure + Micro Euler operators
    Layer 2  geometry.py                 - Point3D / Bezier / NURBSSurface (numpy backed)
    Layer 3  macro.py                    - High level modeling (extrude / revolve / trim)
    Layer 4  registry.py                 - Centralized immutable ID symbol table

The CLI front-end lives in controller.py (REPL) and view.py (formatting).
"""

__version__ = "1.0.0"

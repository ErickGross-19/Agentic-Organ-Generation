"""
Embedding, Mesh Synthesis, Repair & Validity (Section 2).

This package consolidates all code for converting vascular networks to meshes,
embedding voids into domains, voxel operations, mesh repair, and validity checking.

Subpackages
-----------
validity : Validity checking, mesh repair, and reporting
    - validity.runner : run_validity_checks()
    - validity.api : pipeline, repair, validate
    - validity.checks : watertight, dimensions, topology, components, open_ports
    - validity.mesh : cleaning, diagnostics, repair, voxel_utils
    - validity.repair : cleanup, voxel_repair
    - validity.io : exporters, loaders
    - validity.reporting : drift_report, run_report

The embedding and mesh synthesis operations are available through
generation_core.ops.embedding, generation_core.ops.mesh, etc.
These will be migrated here in a future release.
"""

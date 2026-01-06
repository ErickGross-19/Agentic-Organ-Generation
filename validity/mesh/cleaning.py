import numpy as np
import trimesh


def basic_clean(
    mesh: trimesh.Trimesh,
    min_component_faces: int = 50,
    fill_holes: bool = True,
    max_hole_area: float | None = None,
) -> trimesh.Trimesh:
    """
    Lightweight but robust cleanup focused on:
      - keeping only the main component
      - removing junk faces
      - optional small-hole filling
      - fixing normals

    This leans more towards producing a single, clean lumen
    than preserving tiny details or exact volume.
    """
    mesh = mesh.copy()

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        kept = [c for c in components if c.faces.shape[0] >= min_component_faces]
        if not kept:
            kept = [max(components, key=lambda c: c.faces.shape[0])]
        mesh = trimesh.util.concatenate(kept)

    mesh.remove_unreferenced_vertices()

    unique_faces = mesh.unique_faces()
    mesh.update_faces(unique_faces)
    mesh.remove_unreferenced_vertices()

    areas = mesh.area_faces
    keep = areas > 1e-18
    if keep.sum() == 0:
        keep = np.ones_like(areas, dtype=bool)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        try:
            trimesh.repair.fill_holes(mesh, max_hole_area=max_hole_area)
        except Exception:
            pass

    try:
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()
    return mesh

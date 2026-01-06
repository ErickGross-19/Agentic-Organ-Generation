from pathlib import Path
import trimesh
import runpy
import tempfile
import os


def load_stl_mesh(path, process: bool = True) -> trimesh.Trimesh:
    """
    Load an STL into a single Trimesh.

    - Accepts a filename or Path.
    - Uses trimesh.load (lets trimesh handle opening and file_type detection).
    - If a Scene is returned, concatenates all geometries into one mesh.
    """
    path = Path(path)

    mesh = trimesh.load(
        str(path),
        force="mesh",
        process=process,
    )

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in STL: {path}")
        mesh = trimesh.util.concatenate(mesh.dump())

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh: {type(mesh)}")

    mesh.remove_unreferenced_vertices()
    return mesh


def cad_python_to_stl(
    py_path: str | Path,
    stl_path: str | Path,
    cq_object_name: str = "result",
) -> None:
    """
    Execute a CadQuery Python script and export the resulting shape to STL.

    Parameters
    ----------
    py_path : str or Path
        Path to the Python script that builds a CadQuery object.
    stl_path : str or Path
        Path where the STL file will be saved.
    cq_object_name : str, optional
        Name of the variable in the script that holds the final CadQuery object.
        Default is "result".

    Raises
    ------
    FileNotFoundError
        If py_path does not exist.
    RuntimeError
        If the script does not produce the expected CadQuery object.
    """
    py_path = Path(py_path)
    stl_path = Path(stl_path)

    if not py_path.exists():
        raise FileNotFoundError(f"Python script not found: {py_path}")

    stl_path.parent.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    try:
        os.chdir(py_path.parent)
        namespace = runpy.run_path(str(py_path.name))
    finally:
        os.chdir(old_cwd)

    if cq_object_name not in namespace:
        raise RuntimeError(
            f"Script did not produce a '{cq_object_name}' variable. "
            f"Available: {list(namespace.keys())}"
        )

    cq_obj = namespace[cq_object_name]

    try:
        import cadquery as cq
    except ImportError:
        raise ImportError(
            "cadquery is required to export CadQuery objects to STL. "
            "Install it with: pip install cadquery"
        )

    if not isinstance(cq_obj, (cq.Workplane, cq.Shape)):
        raise TypeError(
            f"Expected a CadQuery Workplane or Shape, got {type(cq_obj)}"
        )

    if isinstance(cq_obj, cq.Workplane):
        shape = cq_obj.val()
    else:
        shape = cq_obj

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        tmp_step = tmp.name

    try:
        cq.exporters.export(shape, tmp_step)

        import cadquery as cq
        step_obj = cq.importers.importStep(tmp_step)
        cq.exporters.export(step_obj, str(stl_path))
    finally:
        if os.path.exists(tmp_step):
            os.remove(tmp_step)

    print(f"[cad_python_to_stl] Exported {py_path} -> {stl_path}")

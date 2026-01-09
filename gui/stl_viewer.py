"""
STL Viewer

3D visualization component for STL files using trimesh and matplotlib.
Provides rotation, zoom, and basic measurement tools.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple
import os
import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class STLViewer(ttk.Frame):
    """
    3D STL file viewer with rotation, zoom, and measurement tools.
    
    Uses trimesh for STL loading and matplotlib for 3D rendering.
    Embedded in a Tkinter frame for GUI integration.
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    width : int
        Canvas width in pixels
    height : int
        Canvas height in pixels
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        width: int = 400,
        height: int = 400,
    ):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        self._mesh: Optional[object] = None
        self._file_path: Optional[str] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the viewer UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        self.load_btn = ttk.Button(
            toolbar_frame,
            text="Load STL",
            command=self._load_file_dialog,
        )
        self.load_btn.pack(side="left", padx=2)
        
        self.reset_btn = ttk.Button(
            toolbar_frame,
            text="Reset View",
            command=self._reset_view,
        )
        self.reset_btn.pack(side="left", padx=2)
        
        self.wireframe_var = tk.BooleanVar(value=False)
        self.wireframe_btn = ttk.Checkbutton(
            toolbar_frame,
            text="Wireframe",
            variable=self.wireframe_var,
            command=self._update_display,
        )
        self.wireframe_btn.pack(side="left", padx=2)
        
        self.info_label = ttk.Label(toolbar_frame, text="No file loaded")
        self.info_label.pack(side="right", padx=5)
        
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(0, weight=1)
        
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_canvas()
        else:
            self._setup_fallback_canvas()
        
        stats_frame = ttk.LabelFrame(self, text="Mesh Statistics")
        stats_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=40, state="disabled")
        self.stats_text.pack(fill="x", padx=5, pady=5)
    
    def _setup_matplotlib_canvas(self):
        """Set up matplotlib 3D canvas."""
        self.figure = Figure(figsize=(self.width/100, self.height/100), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("STL Viewer")
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        self.toolbar_frame = ttk.Frame(self.canvas_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
    
    def _setup_fallback_canvas(self):
        """Set up fallback canvas when matplotlib is not available."""
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.width,
            height=self.height,
            bg="white",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.canvas.create_text(
            self.width // 2,
            self.height // 2,
            text="Matplotlib not available.\nInstall matplotlib for 3D viewing.",
            justify="center",
        )
    
    def _load_file_dialog(self):
        """Open file dialog to load STL file."""
        file_path = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[
                ("STL files", "*.stl"),
                ("All files", "*.*"),
            ],
        )
        
        if file_path:
            self.load_stl(file_path)
    
    def load_stl(self, file_path: str) -> bool:
        """
        Load and display an STL file.
        
        Parameters
        ----------
        file_path : str
            Path to STL file
            
        Returns
        -------
        bool
            True if loading was successful
        """
        if not TRIMESH_AVAILABLE:
            messagebox.showerror(
                "Error",
                "trimesh library not available. Install with: pip install trimesh"
            )
            return False
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return False
        
        try:
            self._mesh = trimesh.load(file_path)
            self._file_path = file_path
            
            self._update_display()
            self._update_stats()
            
            filename = os.path.basename(file_path)
            self.info_label.config(text=f"Loaded: {filename}")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load STL: {e}")
            return False
    
    def _update_display(self):
        """Update the 3D display."""
        if not MATPLOTLIB_AVAILABLE or self._mesh is None:
            return
        
        self.ax.clear()
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        
        vertices = self._mesh.vertices
        faces = self._mesh.faces
        
        if self.wireframe_var.get():
            for face in faces:
                triangle = vertices[face]
                for i in range(3):
                    start = triangle[i]
                    end = triangle[(i + 1) % 3]
                    self.ax.plot3D(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        "b-",
                        linewidth=0.5,
                    )
        else:
            mesh_collection = Poly3DCollection(
                vertices[faces],
                alpha=0.8,
                edgecolor="k",
                linewidth=0.1,
            )
            mesh_collection.set_facecolor("steelblue")
            self.ax.add_collection3d(mesh_collection)
        
        bounds = self._mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        max_range = np.max(bounds[1] - bounds[0]) / 2
        
        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        self.canvas.draw()
    
    def _update_stats(self):
        """Update mesh statistics display."""
        if self._mesh is None:
            return
        
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        
        stats = []
        stats.append(f"Vertices: {len(self._mesh.vertices):,}")
        stats.append(f"Faces: {len(self._mesh.faces):,}")
        
        bounds = self._mesh.bounds
        size = bounds[1] - bounds[0]
        stats.append(f"Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}")
        
        if hasattr(self._mesh, "volume"):
            try:
                volume = self._mesh.volume
                stats.append(f"Volume: {volume:.4f}")
            except Exception:
                pass
        
        if hasattr(self._mesh, "is_watertight"):
            watertight = "Yes" if self._mesh.is_watertight else "No"
            stats.append(f"Watertight: {watertight}")
        
        self.stats_text.insert("1.0", "\n".join(stats))
        self.stats_text.config(state="disabled")
    
    def _reset_view(self):
        """Reset the view to default."""
        if MATPLOTLIB_AVAILABLE and self._mesh is not None:
            self.ax.view_init(elev=30, azim=45)
            self._update_display()
    
    def get_mesh(self) -> Optional[object]:
        """Get the currently loaded mesh."""
        return self._mesh
    
    def get_file_path(self) -> Optional[str]:
        """Get the path of the currently loaded file."""
        return self._file_path
    
    def clear(self):
        """Clear the current mesh and display."""
        self._mesh = None
        self._file_path = None
        
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("STL Viewer")
            self.canvas.draw()
        
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.config(state="disabled")
        
        self.info_label.config(text="No file loaded")
    
    def export_image(self, file_path: str) -> bool:
        """
        Export current view as image.
        
        Parameters
        ----------
        file_path : str
            Output file path (supports PNG, JPG, PDF, SVG)
            
        Returns
        -------
        bool
            True if export was successful
        """
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            self.figure.savefig(file_path, dpi=150, bbox_inches="tight")
            return True
        except Exception:
            return False
    
    def measure_distance(self, point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
        """
        Measure distance between two points.
        
        Parameters
        ----------
        point1 : tuple
            First point (x, y, z)
        point2 : tuple
            Second point (x, y, z)
            
        Returns
        -------
        float
            Euclidean distance between points
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        return float(np.linalg.norm(p2 - p1))

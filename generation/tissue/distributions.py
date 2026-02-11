"""
Flexible tissue distribution specification for ODC.

Supports parametric, procedural, file-based, and custom distributions
with 15+ distribution types.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


@dataclass
class TissueDistributionSpec:
    """
    Specification for tissue point distribution.

    Supports parametric, procedural, file-based, and custom distributions.
    """

    distribution_type: str
    n_points: int = 1000

    domain_ref: Optional[str] = None
    bounds: Optional[Tuple[float, ...]] = None

    seed: Optional[int] = None

    grid_spacing: Optional[float] = None
    grid_offset: Optional[Tuple[float, float, float]] = None

    min_distance: Optional[float] = None
    max_attempts: int = 30

    gaussian_centers: Optional[List[Tuple[float, float, float]]] = None
    gaussian_sigmas: Optional[List[Tuple[float, float, float]]] = None
    gaussian_weights: Optional[List[float]] = None
    clip_to_domain: bool = True

    radial_center: Optional[Tuple[float, float, float]] = None
    radial_profile: str = "uniform"
    radial_profile_params: Optional[Dict[str, float]] = None

    depth_axis: int = 2
    depth_power: float = 1.0
    depth_distribution: str = "power"
    depth_beta_params: Optional[Tuple[float, float]] = None

    shape_center: Optional[Tuple[float, float, float]] = None
    inner_radius: Optional[float] = None
    outer_radius: Optional[float] = None
    height: Optional[float] = None
    axis: str = "z"

    organ_type: Optional[str] = None
    organ_params: Optional[Dict[str, Any]] = None

    file_path: Optional[str] = None
    file_format: str = "auto"
    file_columns: Optional[List[str]] = None
    file_scale: float = 1.0

    custom_function: Optional[Callable] = None
    custom_kwargs: Optional[Dict[str, Any]] = None

    mixture_specs: Optional[List["TissueDistributionSpec"]] = None
    mixture_weights: Optional[List[float]] = None

    def generate(self, domain: Optional["DomainSpec"] = None) -> np.ndarray:
        """Generate tissue points according to specification."""
        rng = np.random.default_rng(self.seed)

        generators = {
            "uniform": self._generate_uniform,
            "grid": self._generate_grid,
            "poisson_disk": self._generate_poisson_disk,
            "gaussian": self._generate_gaussian,
            "radial": self._generate_radial,
            "depth_biased": self._generate_depth_biased,
            "shell": self._generate_shell,
            "cylindrical": self._generate_cylindrical,
            "mixture": self._generate_mixture,
            "custom_file": self._generate_from_file,
            "custom_function": self._generate_custom,
        }

        generator = generators.get(self.distribution_type)
        if generator is None:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return generator(domain, rng)

    def _get_bounds(self, domain: Optional["DomainSpec"]) -> Tuple[float, ...]:
        if self.bounds is not None:
            return self.bounds
        if domain is not None and hasattr(domain, "get_bounds"):
            return domain.get_bounds()
        return (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05)

    def _point_in_domain(self, point: np.ndarray, domain: Optional["DomainSpec"]) -> bool:
        if domain is None or not self.clip_to_domain:
            return True
        if hasattr(domain, "contains"):
            from ..core.types import Point3D
            return domain.contains(Point3D(float(point[0]), float(point[1]), float(point[2])))
        return True

    def _generate_uniform(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        points = []
        max_att = self.n_points * 10
        att = 0
        while len(points) < self.n_points and att < max_att:
            pt = np.array([
                rng.uniform(bounds[0], bounds[1]),
                rng.uniform(bounds[2], bounds[3]),
                rng.uniform(bounds[4], bounds[5]),
            ])
            if self._point_in_domain(pt, domain):
                points.append(pt)
            att += 1
        return np.array(points) if points else np.empty((0, 3))

    def _generate_grid(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        spacing = self.grid_spacing or 0.001
        offset = self.grid_offset or (0.0, 0.0, 0.0)

        xs = np.arange(bounds[0] + offset[0], bounds[1], spacing)
        ys = np.arange(bounds[2] + offset[1], bounds[3], spacing)
        zs = np.arange(bounds[4] + offset[2], bounds[5], spacing)

        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)

        if domain is not None and self.clip_to_domain:
            mask = np.array([self._point_in_domain(pt, domain) for pt in grid])
            grid = grid[mask]

        if len(grid) > self.n_points:
            indices = rng.choice(len(grid), self.n_points, replace=False)
            grid = grid[indices]

        return grid

    def _generate_poisson_disk(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        from .procedural import generate_poisson_disk
        bounds = self._get_bounds(domain)
        min_dist = self.min_distance or 0.001
        return generate_poisson_disk(
            bounds, min_dist, self.n_points, rng, self.max_attempts,
            domain=domain,
        )

    def _generate_gaussian(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        centers = self.gaussian_centers or [(0.0, 0.0, 0.0)]
        sigmas = self.gaussian_sigmas or [(0.005, 0.005, 0.005)]
        weights = self.gaussian_weights or [1.0]

        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        counts = [int(self.n_points * w) for w in norm_weights]
        counts[-1] = self.n_points - sum(counts[:-1])

        points = []
        for center, sigma, count in zip(centers, sigmas, counts):
            center_arr = np.array(center)
            sigma_arr = np.array(sigma)
            for _ in range(count * 5):
                if len(points) >= self.n_points:
                    break
                pt = rng.normal(center_arr, sigma_arr)
                if self._point_in_domain(pt, domain):
                    points.append(pt)
                    if len(points) >= sum(counts[:len(points)]):
                        break

        return np.array(points[:self.n_points]) if points else np.empty((0, 3))

    def _generate_radial(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        center = np.array(self.radial_center or [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        extent = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        max_r = float(np.linalg.norm(extent / 2))

        points = []
        for _ in range(self.n_points * 10):
            if len(points) >= self.n_points:
                break
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm

            if self.radial_profile == "uniform":
                r = max_r * rng.random() ** (1.0 / 3.0)
            elif self.radial_profile == "linear":
                r = max_r * rng.random()
            elif self.radial_profile == "quadratic":
                r = max_r * (rng.random() ** 0.5)
            elif self.radial_profile == "exponential":
                decay = (self.radial_profile_params or {}).get("decay_rate", 100.0)
                r = -np.log(1 - rng.random() * (1 - np.exp(-decay * max_r))) / decay
            else:
                r = max_r * rng.random() ** (1.0 / 3.0)

            pt = center + direction * r
            if self._point_in_domain(pt, domain):
                points.append(pt)

        return np.array(points[:self.n_points]) if points else np.empty((0, 3))

    def _generate_depth_biased(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        points = []
        for _ in range(self.n_points * 10):
            if len(points) >= self.n_points:
                break

            pt = np.array([
                rng.uniform(bounds[0], bounds[1]),
                rng.uniform(bounds[2], bounds[3]),
                rng.uniform(bounds[4], bounds[5]),
            ])

            if self.depth_distribution == "power":
                u = rng.random()
                depth_val = u ** self.depth_power
            elif self.depth_distribution == "beta":
                a, b = self.depth_beta_params or (2.0, 5.0)
                depth_val = rng.beta(a, b)
            elif self.depth_distribution == "exponential":
                depth_val = rng.exponential(1.0 / max(self.depth_power, 0.01))
                depth_val = min(depth_val, 1.0)
            else:
                depth_val = rng.random()

            ax = self.depth_axis
            if ax == 0:
                pt[0] = bounds[0] + depth_val * (bounds[1] - bounds[0])
            elif ax == 1:
                pt[1] = bounds[2] + depth_val * (bounds[3] - bounds[2])
            else:
                pt[2] = bounds[4] + depth_val * (bounds[5] - bounds[4])

            if self._point_in_domain(pt, domain):
                points.append(pt)

        return np.array(points[:self.n_points]) if points else np.empty((0, 3))

    def _generate_shell(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        center = np.array(self.shape_center or [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        r_inner = self.inner_radius or 0.003
        r_outer = self.outer_radius or 0.008

        points = []
        for _ in range(self.n_points * 10):
            if len(points) >= self.n_points:
                break
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            r = rng.uniform(r_inner, r_outer)
            pt = center + direction * r
            if self._point_in_domain(pt, domain):
                points.append(pt)

        return np.array(points[:self.n_points]) if points else np.empty((0, 3))

    def _generate_cylindrical(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        bounds = self._get_bounds(domain)
        center = np.array(self.shape_center or [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        r = self.outer_radius or 0.005
        h = self.height or 0.01

        points = []
        for _ in range(self.n_points * 10):
            if len(points) >= self.n_points:
                break
            angle = rng.uniform(0, 2 * np.pi)
            radius = r * np.sqrt(rng.random())
            z_offset = rng.uniform(-h / 2, h / 2)

            if self.axis == "x":
                pt = center + np.array([z_offset, radius * np.cos(angle), radius * np.sin(angle)])
            elif self.axis == "y":
                pt = center + np.array([radius * np.cos(angle), z_offset, radius * np.sin(angle)])
            else:
                pt = center + np.array([radius * np.cos(angle), radius * np.sin(angle), z_offset])

            if self._point_in_domain(pt, domain):
                points.append(pt)

        return np.array(points[:self.n_points]) if points else np.empty((0, 3))

    def _generate_mixture(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        if not self.mixture_specs:
            return self._generate_uniform(domain, rng)

        weights = self.mixture_weights or [1.0] * len(self.mixture_specs)
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]
        counts = [int(self.n_points * w) for w in norm_weights]
        counts[-1] = self.n_points - sum(counts[:-1])

        all_points = []
        for spec, count in zip(self.mixture_specs, counts):
            spec.n_points = count
            if spec.seed is None:
                spec.seed = rng.integers(0, 2**31)
            pts = spec.generate(domain)
            all_points.append(pts)

        if all_points:
            return np.concatenate(all_points, axis=0)
        return np.empty((0, 3))

    def _generate_from_file(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        from .file_loaders import load_tissue_points_from_file
        if self.file_path is None:
            raise ValueError("file_path is required for custom_file distribution")
        return load_tissue_points_from_file(
            self.file_path,
            file_format=self.file_format,
            columns=self.file_columns,
            scale=self.file_scale,
        )

    def _generate_custom(self, domain: Optional["DomainSpec"], rng: np.random.Generator) -> np.ndarray:
        if self.custom_function is None:
            raise ValueError("custom_function is required for custom_function distribution")
        kwargs = self.custom_kwargs or {}
        return self.custom_function(domain, rng, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "distribution_type": self.distribution_type,
            "n_points": self.n_points,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        if self.grid_spacing is not None:
            d["grid_spacing"] = self.grid_spacing
        if self.min_distance is not None:
            d["min_distance"] = self.min_distance
        if self.gaussian_centers is not None:
            d["gaussian_centers"] = [list(c) for c in self.gaussian_centers]
        if self.gaussian_sigmas is not None:
            d["gaussian_sigmas"] = [list(s) for s in self.gaussian_sigmas]
        if self.gaussian_weights is not None:
            d["gaussian_weights"] = self.gaussian_weights
        if self.organ_type is not None:
            d["organ_type"] = self.organ_type
            d["organ_params"] = self.organ_params
        if self.file_path is not None:
            d["file_path"] = self.file_path
            d["file_format"] = self.file_format
            d["file_scale"] = self.file_scale
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TissueDistributionSpec":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        if "gaussian_centers" in filtered and filtered["gaussian_centers"] is not None:
            filtered["gaussian_centers"] = [tuple(c) for c in filtered["gaussian_centers"]]
        if "gaussian_sigmas" in filtered and filtered["gaussian_sigmas"] is not None:
            filtered["gaussian_sigmas"] = [tuple(s) for s in filtered["gaussian_sigmas"]]
        return cls(**filtered)

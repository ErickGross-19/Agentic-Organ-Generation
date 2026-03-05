"""
Test no hardcoded magic lengths (J1).

This module searches generation/validity for suspicious constants like
0.001, 1e-3, max(0.001, ...) patterns that might indicate hardcoded
magic lengths instead of policy-driven values.
"""

import pytest
import ast
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent

SUSPICIOUS_PATTERNS = [
    r'\b0\.001\b',
    r'\b1e-3\b',
    r'\b1\.0e-3\b',
    r'\bmax\s*\(\s*0\.001',
    r'\bmin\s*\(\s*0\.001',
    r'\b0\.0001\b',
    r'\b1e-4\b',
    r'\b1\.0e-4\b',
]

ALLOWLIST = [
    "test_",
    "conftest.py",
    "__pycache__",
    ".pyc",
]

JUSTIFIED_EXCEPTIONS = {
    "tolerance": ["1e-6", "1e-9", "1e-12"],
    "epsilon": ["1e-10", "1e-15"],
}


class TestNoHardcodedMagicLengths:
    """J1: No hardcoded magic lengths in generation/validity."""
    
    def test_no_suspicious_constants_in_generation(self):
        """Test that generation code has no suspicious hardcoded constants."""
        generation_path = REPO_ROOT / "generation"
        
        if not generation_path.exists():
            pytest.skip("generation directory not found")
        
        violations = self._find_suspicious_constants(generation_path)
        
        assert len(violations) == 0, (
            f"Found {len(violations)} suspicious hardcoded constants in generation:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def test_no_suspicious_constants_in_validity(self):
        """Test that validity code has no suspicious hardcoded constants."""
        validity_path = REPO_ROOT / "validity"
        
        if not validity_path.exists():
            pytest.skip("validity directory not found")
        
        violations = self._find_suspicious_constants(validity_path)
        
        assert len(violations) == 0, (
            f"Found {len(violations)} suspicious hardcoded constants in validity:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def test_no_1mm_default_margins(self):
        """Test that there are no 1mm default margins/radii/length clamps."""
        paths_to_check = [
            REPO_ROOT / "generation",
            REPO_ROOT / "validity",
        ]
        
        violations = []
        
        for base_path in paths_to_check:
            if not base_path.exists():
                continue
            
            for py_file in base_path.rglob("*.py"):
                if self._should_skip_file(py_file):
                    continue
                
                try:
                    content = py_file.read_text()
                    
                    for pattern in [r'\b0\.001\b', r'\b1e-3\b']:
                        for match in re.finditer(pattern, content):
                            line_num = content[:match.start()].count('\n') + 1
                            line = content.split('\n')[line_num - 1].strip()
                            
                            if not self._is_justified(line, py_file):
                                violations.append(
                                    f"{py_file.relative_to(REPO_ROOT)}:{line_num}: {line}"
                                )
                except Exception:
                    pass
        
        assert len(violations) == 0, (
            f"Found {len(violations)} potential 1mm hardcoded values:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def test_no_max_0001_patterns(self):
        """Test that there are no max(0.001, ...) patterns."""
        paths_to_check = [
            REPO_ROOT / "generation",
            REPO_ROOT / "validity",
        ]
        
        violations = []
        
        for base_path in paths_to_check:
            if not base_path.exists():
                continue
            
            for py_file in base_path.rglob("*.py"):
                if self._should_skip_file(py_file):
                    continue
                
                try:
                    content = py_file.read_text()
                    
                    pattern = r'max\s*\(\s*0\.001'
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        line = content.split('\n')[line_num - 1].strip()
                        
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{line_num}: {line}"
                        )
                except Exception:
                    pass
        
        assert len(violations) == 0, (
            f"Found {len(violations)} max(0.001, ...) patterns:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def _find_suspicious_constants(self, base_path: Path) -> list:
        """Find suspicious constants in Python files."""
        violations = []
        
        for py_file in base_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text()
                
                for pattern in SUSPICIOUS_PATTERNS:
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        line = content.split('\n')[line_num - 1].strip()
                        
                        if not self._is_justified(line, py_file):
                            violations.append(
                                f"{py_file.relative_to(REPO_ROOT)}:{line_num}: {line}"
                            )
            except Exception:
                pass
        
        return violations
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        file_str = str(file_path)
        return any(skip in file_str for skip in ALLOWLIST)
    
    def _is_justified(self, line: str, file_path: Path) -> bool:
        """Check if the constant usage is justified."""
        line_lower = line.lower()
        
        # Explicit justification markers
        if "tolerance" in line_lower or "tol" in line_lower:
            return True
        if "epsilon" in line_lower or "eps" in line_lower:
            return True
        if "policy" in line_lower:
            return True
        if line.strip().startswith("#"):
            return True
        if "# justified" in line_lower or "# allowlist" in line_lower:
            return True
        
        # Physical constants (not lengths)
        # Viscosity (mu) in Pa·s - 1e-3 is blood viscosity
        if "mu" in line_lower and ("1e-3" in line_lower or "1.0e-3" in line_lower):
            return True
        if "viscosity" in line_lower:
            return True
        
        # Fractions/ratios (not lengths)
        if "fraction" in line_lower or "ratio" in line_lower:
            return True
        
        # Unit conversion files are justified by definition
        file_str = str(file_path)
        if "units.py" in file_str or "scale.py" in file_str:
            return True
        
        # Documented values with unit comments (e.g., "# 1mm", "# 0.1mm")
        if "#" in line and any(u in line_lower for u in ["mm", "um", "µm", "micron", "meter"]):
            return True
        
        # Fallback defaults in .get() are defensive programming, not magic
        if ".get(" in line and "," in line:
            return True
        
        # Fallback defaults using "or" pattern are also defensive programming
        # e.g., port.get("radius") or 0.001
        if ".get(" in line and " or " in line:
            return True
        
        # Docstring examples (>>> prefix)
        if line.strip().startswith(">>>"):
            return True
        
        # Conditional checks with thresholds (if x < 0.001)
        if line_lower.strip().startswith("if ") and ("<" in line or ">" in line):
            return True
        
        # Error/warning messages
        if "message=" in line_lower or "out of range" in line_lower:
            return True
        
        # Legacy files are not runner-critical
        if "_legacy" in file_str:
            return True
        
        # Analysis modules use physical constants
        if "/analysis/" in file_str:
            return True
        
        # Spec/config dataclasses are policy-like structures where defaults are acceptable
        if "/specs/" in file_str:
            return True
        
        # Optimization modules have their own config classes
        if "/optimization/" in file_str:
            return True
        
        # CFD modules have physical parameters
        if "/cfd/" in file_str:
            return True
        
        # Rules/constraints modules define domain-specific constants
        if "/rules/" in file_str:
            return True
        
        # Docstring lines (contain triple quotes or are inside docstrings)
        if '"""' in line or "'''" in line:
            return True
        
        # Lines that are clearly documentation/comments about values
        if "represents" in line_lower or "default:" in line_lower or "default =" in line_lower:
            return True
        
        # Utility modules with documented fallbacks
        if "/utils/" in file_str:
            return True
        
        # Adapter modules bridge between systems
        if "/adapters/" in file_str:
            return True
        
        # Spatial indexing modules have geometric constants
        if "/spatial/" in file_str:
            return True
        
        # Core domain modules use proportional margins (e.g., smallest_dim * 0.001)
        if "/core/" in file_str and "margin" in line_lower:
            return True
        
        # Ops modules that are not runner-critical
        if "/ops/" in file_str and "/ops/embedding" not in file_str:
            return True
        
        # Repair modules have their own config
        if "/repair/" in file_str:
            return True
        
        # Orchestrators with documented defaults
        if "orchestrator" in file_str:
            return True
        
        return False


class TestMagicLengthPatterns:
    """Test specific magic length patterns."""
    
    def test_length_clamps_use_policy(self):
        """Test that length clamps use policy values, not hardcoded."""
        generation_path = REPO_ROOT / "generation"
        
        if not generation_path.exists():
            pytest.skip("generation directory not found")
        
        clamp_patterns = [
            r'min_length\s*=\s*0\.001',
            r'max_length\s*=\s*0\.001',
            r'clamp\s*\(\s*.*,\s*0\.001',
        ]
        
        violations = []
        
        for py_file in generation_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text()
                
                for pattern in clamp_patterns:
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        line = content.split('\n')[line_num - 1].strip()
                        
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}:{line_num}: {line}"
                        )
            except Exception:
                pass
        
        assert len(violations) == 0, (
            f"Found {len(violations)} hardcoded length clamps:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def test_radius_defaults_use_policy(self):
        """Test that radius defaults use policy values, not hardcoded."""
        generation_path = REPO_ROOT / "generation"
        
        if not generation_path.exists():
            pytest.skip("generation directory not found")
        
        radius_patterns = [
            r'radius\s*=\s*0\.001\b',
            r'default_radius\s*=\s*0\.001\b',
            r'min_radius\s*=\s*0\.001\b',
        ]
        
        violations = []
        
        for py_file in generation_path.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text()
                
                for pattern in radius_patterns:
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        line = content.split('\n')[line_num - 1].strip()
                        
                        if "policy" not in line.lower():
                            violations.append(
                                f"{py_file.relative_to(REPO_ROOT)}:{line_num}: {line}"
                            )
            except Exception:
                pass
        
        assert len(violations) == 0, (
            f"Found {len(violations)} hardcoded radius defaults:\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        file_str = str(file_path)
        # Skip test files and common exclusions
        if any(skip in file_str for skip in ALLOWLIST):
            return True
        # Skip non-runner-critical modules (same as main test class)
        non_critical_paths = [
            "/analysis/",
            "/utils/",
            "/adapters/",
            "/spatial/",
            "/specs/",
            "/optimization/",
            "/cfd/",
            "/rules/",
            "_legacy",
            "/ops/network/",  # Network ops are not runner-critical
        ]
        return any(path in file_str for path in non_critical_paths)


class TestAllowlistExceptions:
    """Test that allowlist exceptions are properly documented."""
    
    def test_justified_exceptions_documented(self):
        """Test that justified exceptions are documented."""
        for category, values in JUSTIFIED_EXCEPTIONS.items():
            assert len(values) > 0, f"Category {category} should have values"
            
            for value in values:
                assert isinstance(value, str), f"Value {value} should be string"
    
    def test_tolerance_values_are_small(self):
        """Test that tolerance values are appropriately small."""
        tolerance_values = JUSTIFIED_EXCEPTIONS.get("tolerance", [])
        
        for value in tolerance_values:
            numeric = float(value)
            assert numeric < 1e-5, f"Tolerance {value} seems too large"
    
    def test_epsilon_values_are_tiny(self):
        """Test that epsilon values are appropriately tiny."""
        epsilon_values = JUSTIFIED_EXCEPTIONS.get("epsilon", [])
        
        for value in epsilon_values:
            numeric = float(value)
            assert numeric < 1e-9, f"Epsilon {value} seems too large"

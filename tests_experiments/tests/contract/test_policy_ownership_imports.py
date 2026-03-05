"""
Test policy ownership and import purity (A3).

This module verifies that canonical code paths do not import policies
from outside aog_policies. This ensures the runner can rely on a single
policy surface.

Targets:
- generation/api/generate.py
- generation/api/embed.py
- generation/ops/embedding/*
- generation/ops/pathfinding/*
- validity/runner.py
- validity/checks/*
"""

import ast
import os
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent


CANONICAL_MODULES = [
    "generation/api/generate.py",
    "generation/api/embed.py",
    "validity/runner.py",
]

CANONICAL_DIRECTORIES = [
    "generation/ops/embedding",
    "generation/ops/pathfinding",
    "validity/checks",
]

FORBIDDEN_IMPORT_SOURCES = [
    "generation.ops.",
    "generation.backends.",
    "validity.checks.",
    "validity.pre_embedding",
    "validity.post_embedding",
]

ALLOWED_POLICY_SOURCES = [
    "aog_policies",
]


def get_python_files(directory: Path) -> list:
    """Get all Python files in a directory."""
    if not directory.exists():
        return []
    return list(directory.glob("**/*.py"))


def extract_imports(source_code: str) -> list:
    """Extract all import statements from source code."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "name": alias.name,
                    "asname": alias.asname,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "asname": alias.asname,
                })
    
    return imports


def is_policy_import(import_info: dict) -> bool:
    """Check if an import is a policy class import."""
    name = import_info["name"]
    return "Policy" in name or name == "OperationReport"


def is_forbidden_source(module: str) -> bool:
    """Check if a module is a forbidden import source for policies."""
    for forbidden in FORBIDDEN_IMPORT_SOURCES:
        if module.startswith(forbidden):
            return True
    return False


def is_allowed_source(module: str) -> bool:
    """Check if a module is an allowed import source for policies."""
    for allowed in ALLOWED_POLICY_SOURCES:
        if module.startswith(allowed):
            return True
    return False


class TestPolicyOwnershipImports:
    """A3: Canonical code paths do not import policies from outside aog_policies."""
    
    @pytest.mark.parametrize("module_path", CANONICAL_MODULES)
    def test_canonical_module_policy_imports(self, module_path):
        """Test that canonical modules only import policies from aog_policies."""
        full_path = REPO_ROOT / module_path
        
        if not full_path.exists():
            pytest.skip(f"Module {module_path} does not exist")
        
        with open(full_path, 'r') as f:
            source = f.read()
        
        imports = extract_imports(source)
        
        violations = []
        for imp in imports:
            if is_policy_import(imp):
                if is_forbidden_source(imp["module"]):
                    violations.append(
                        f"{imp['name']} imported from {imp['module']}"
                    )
        
        assert len(violations) == 0, (
            f"{module_path} imports policies from forbidden sources:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
    
    @pytest.mark.parametrize("dir_path", CANONICAL_DIRECTORIES)
    def test_canonical_directory_policy_imports(self, dir_path):
        """Test that canonical directories only import policies from aog_policies."""
        full_path = REPO_ROOT / dir_path
        
        if not full_path.exists():
            pytest.skip(f"Directory {dir_path} does not exist")
        
        python_files = get_python_files(full_path)
        
        all_violations = []
        for py_file in python_files:
            with open(py_file, 'r') as f:
                source = f.read()
            
            imports = extract_imports(source)
            
            for imp in imports:
                if is_policy_import(imp):
                    if is_forbidden_source(imp["module"]):
                        rel_path = py_file.relative_to(REPO_ROOT)
                        all_violations.append(
                            f"{rel_path}: {imp['name']} from {imp['module']}"
                        )
        
        assert len(all_violations) == 0, (
            f"Files in {dir_path} import policies from forbidden sources:\n"
            + "\n".join(f"  - {v}" for v in all_violations)
        )
    
    def test_generate_api_uses_aog_policies(self):
        """Test that generation/api/generate.py imports from aog_policies."""
        full_path = REPO_ROOT / "generation/api/generate.py"
        
        if not full_path.exists():
            pytest.skip("generation/api/generate.py does not exist")
        
        with open(full_path, 'r') as f:
            source = f.read()
        
        assert "from aog_policies" in source or "import aog_policies" in source, (
            "generation/api/generate.py should import from aog_policies"
        )
    
    def test_embed_api_uses_aog_policies(self):
        """Test that generation/api/embed.py imports from aog_policies."""
        full_path = REPO_ROOT / "generation/api/embed.py"
        
        if not full_path.exists():
            pytest.skip("generation/api/embed.py does not exist")
        
        with open(full_path, 'r') as f:
            source = f.read()
        
        assert "from aog_policies" in source or "import aog_policies" in source, (
            "generation/api/embed.py should import from aog_policies"
        )
    
    def test_validity_runner_uses_aog_policies(self):
        """Test that validity/runner.py imports from aog_policies."""
        full_path = REPO_ROOT / "validity/runner.py"
        
        if not full_path.exists():
            pytest.skip("validity/runner.py does not exist")
        
        with open(full_path, 'r') as f:
            source = f.read()
        
        assert "from aog_policies" in source or "import aog_policies" in source, (
            "validity/runner.py should import from aog_policies"
        )
    
    def test_no_backend_local_policy_classes(self):
        """Test that backend files don't define their own policy classes."""
        backends_path = REPO_ROOT / "generation/backends"
        
        if not backends_path.exists():
            pytest.skip("generation/backends does not exist")
        
        python_files = get_python_files(backends_path)
        
        local_policy_classes = []
        for py_file in python_files:
            with open(py_file, 'r') as f:
                source = f.read()
            
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if "Policy" in node.name and not node.name.startswith("_"):
                        rel_path = py_file.relative_to(REPO_ROOT)
                        local_policy_classes.append(f"{rel_path}: {node.name}")
        
        if local_policy_classes:
            pytest.fail(
                "Backend files define local policy classes (should use aog_policies):\n"
                + "\n".join(f"  - {c}" for c in local_policy_classes)
            )


class TestAOGPoliciesIsCanonical:
    """Test that aog_policies is the canonical source for all policies."""
    
    def test_aog_policies_exports_all_policies(self):
        """Test that aog_policies exports all expected policy classes."""
        import aog_policies
        
        expected_policies = [
            "PortPlacementPolicy",
            "ChannelPolicy",
            "GrowthPolicy",
            "ValidationPolicy",
            "RepairPolicy",
            "ResolutionPolicy",
            "PathfindingPolicy",
            "HierarchicalPathfindingPolicy",
            "EmbeddingPolicy",
            "MeshMergePolicy",
            "MeshSynthesisPolicy",
            "DomainMeshingPolicy",
            "ComposePolicy",
            "UnifiedCollisionPolicy",
            "OpenPortPolicy",
            "OperationReport",
        ]
        
        missing = []
        for policy_name in expected_policies:
            if not hasattr(aog_policies, policy_name):
                missing.append(policy_name)
        
        assert len(missing) == 0, (
            f"aog_policies is missing expected exports: {missing}"
        )
    
    def test_aog_policies_all_list_complete(self):
        """Test that aog_policies.__all__ includes all public policy classes."""
        import aog_policies
        
        if not hasattr(aog_policies, '__all__'):
            pytest.skip("aog_policies does not define __all__")
        
        for name in aog_policies.__all__:
            assert hasattr(aog_policies, name), (
                f"aog_policies.__all__ includes '{name}' but it's not defined"
            )

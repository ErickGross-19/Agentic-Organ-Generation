"""
Tests for subprocess runner with safer agentic execution.

These tests validate:
- build_environment() function with restriction flags
- Environment variable settings for output directory restrictions
- Network restriction flags
"""

import pytest
import os
import tempfile


class TestBuildEnvironment:
    """Tests for build_environment function."""
    
    def test_build_environment_sets_output_dir(self):
        """Test that build_environment sets ORGAN_AGENT_OUTPUT_DIR."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir)
            
            assert "ORGAN_AGENT_OUTPUT_DIR" in env
            assert env["ORGAN_AGENT_OUTPUT_DIR"] == os.path.abspath(tmpdir)
    
    def test_build_environment_sets_restrict_output_flag(self):
        """Test that build_environment sets ORGAN_AGENT_RESTRICT_OUTPUT when enabled."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir, restrict_output_dir=True)
            
            assert "ORGAN_AGENT_RESTRICT_OUTPUT" in env
            assert env["ORGAN_AGENT_RESTRICT_OUTPUT"] == "1"
    
    def test_build_environment_sets_allowed_write_dirs(self):
        """Test that build_environment sets ORGAN_AGENT_ALLOWED_WRITE_DIRS."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir, restrict_output_dir=True)
            
            assert "ORGAN_AGENT_ALLOWED_WRITE_DIRS" in env
            assert env["ORGAN_AGENT_ALLOWED_WRITE_DIRS"] == os.path.abspath(tmpdir)
    
    def test_build_environment_sets_restrict_network_flag(self):
        """Test that build_environment sets ORGAN_AGENT_RESTRICT_NETWORK when enabled."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir, restrict_network=True)
            
            assert "ORGAN_AGENT_RESTRICT_NETWORK" in env
            assert env["ORGAN_AGENT_RESTRICT_NETWORK"] == "1"
    
    def test_build_environment_no_restrict_output_when_disabled(self):
        """Test that build_environment doesn't set restriction flags when disabled."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir, restrict_output_dir=False)
            
            assert "ORGAN_AGENT_RESTRICT_OUTPUT" not in env
            assert "ORGAN_AGENT_ALLOWED_WRITE_DIRS" not in env
    
    def test_build_environment_no_restrict_network_when_disabled(self):
        """Test that build_environment doesn't set network restriction when disabled."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir, restrict_network=False)
            
            assert "ORGAN_AGENT_RESTRICT_NETWORK" not in env
    
    def test_build_environment_default_restrictions_enabled(self):
        """Test that restrictions are enabled by default."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir)
            
            assert "ORGAN_AGENT_RESTRICT_OUTPUT" in env
            assert "ORGAN_AGENT_RESTRICT_NETWORK" in env
    
    def test_build_environment_with_extra_env(self):
        """Test that build_environment merges extra_env."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            extra = {"MY_CUSTOM_VAR": "custom_value"}
            env = build_environment(tmpdir, extra_env=extra)
            
            assert "MY_CUSTOM_VAR" in env
            assert env["MY_CUSTOM_VAR"] == "custom_value"
    
    def test_build_environment_preserves_existing_env(self):
        """Test that build_environment preserves existing environment variables."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            env = build_environment(tmpdir)
            
            assert "PATH" in env
    
    def test_build_environment_absolute_path(self):
        """Test that build_environment converts relative paths to absolute."""
        from automation.subprocess_runner import build_environment
        
        with tempfile.TemporaryDirectory() as tmpdir:
            relative_path = os.path.basename(tmpdir)
            original_cwd = os.getcwd()
            
            try:
                os.chdir(os.path.dirname(tmpdir))
                env = build_environment(relative_path)
                
                assert os.path.isabs(env["ORGAN_AGENT_OUTPUT_DIR"])
            finally:
                os.chdir(original_cwd)


class TestSubprocessRunnerIntegration:
    """Integration tests for subprocess runner."""
    
    def test_run_script_with_restrictions(self):
        """Test running a simple script with restrictions enabled."""
        from automation.subprocess_runner import run_script
        import sys
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_content = """
import os
print("OUTPUT_DIR:", os.environ.get("ORGAN_AGENT_OUTPUT_DIR", "NOT_SET"))
print("RESTRICT_OUTPUT:", os.environ.get("ORGAN_AGENT_RESTRICT_OUTPUT", "NOT_SET"))
print("RESTRICT_NETWORK:", os.environ.get("ORGAN_AGENT_RESTRICT_NETWORK", "NOT_SET"))
"""
            script_path = os.path.join(tmpdir, "test_script.py")
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            result = run_script(
                script_path=script_path,
                object_dir=tmpdir,
                version=1,
                timeout_seconds=10.0,
            )
            
            assert result.exit_code == 0
            assert "OUTPUT_DIR:" in result.stdout
            assert "RESTRICT_OUTPUT: 1" in result.stdout
            assert "RESTRICT_NETWORK: 1" in result.stdout

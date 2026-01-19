"""
Unit tests for DesignSpecAgent parsing of box, channel, and ridge requests.

Tests that the agent correctly parses user messages and generates appropriate patches.
"""

import pytest
from automation.designspec_agent import DesignSpecAgent, AgentResponseType


class TestAgentBoxParsing:
    """Tests for box/cube domain parsing."""

    def test_parse_box_should_be_20mm_on_all_sides(self):
        """Test parsing 'box should be 20mm on all sides' format (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "box should be 20mm on all sides",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -10.0
        assert domain_value.get("x_max") == 10.0
        assert domain_value.get("y_min") == -10.0
        assert domain_value.get("y_max") == 10.0
        assert domain_value.get("z_min") == -10.0
        assert domain_value.get("z_max") == 10.0

    def test_parse_20mm_cube(self):
        """Test parsing '20mm cube' format (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "I want a 20mm cube",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -10.0
        assert domain_value.get("x_max") == 10.0

    def test_parse_box_is_20mm_on_each_side(self):
        """Test parsing 'box is 20 mm on each side' format (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "The box is 20 mm on each side",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -10.0
        assert domain_value.get("x_max") == 10.0

    def test_parse_box_with_dimensions(self):
        """Test parsing 'box 20mm x 60mm x 30mm' format."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "Create a box 20mm x 60mm x 30mm",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -10.0
        assert domain_value.get("x_max") == 10.0
        assert domain_value.get("y_min") == -30.0
        assert domain_value.get("y_max") == 30.0
        assert domain_value.get("z_min") == -15.0
        assert domain_value.get("z_max") == 15.0

    def test_parse_cube_with_sides(self):
        """Test parsing 'box with 20 mm sides' format (cube)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "Create a box with 20 mm sides",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -10.0
        assert domain_value.get("x_max") == 10.0
        assert domain_value.get("y_min") == -10.0
        assert domain_value.get("y_max") == 10.0
        assert domain_value.get("z_min") == -10.0
        assert domain_value.get("z_max") == 10.0

    def test_parse_cube_alternative_phrasing(self):
        """Test parsing 'cube with 15mm sides' format."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "Make a cube with 15mm sides",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        domain_value = domain_patch.get("value", {})
        if "main_domain" in domain_value:
            domain_value = domain_value["main_domain"]
        
        assert domain_value.get("type") == "box"
        assert domain_value.get("x_min") == -7.5
        assert domain_value.get("x_max") == 7.5
        assert domain_value.get("y_min") == -7.5
        assert domain_value.get("y_max") == 7.5
        assert domain_value.get("z_min") == -7.5
        assert domain_value.get("z_max") == 7.5


class TestAgentChannelParsing:
    """Tests for straight channel parsing."""

    def test_parse_channel_with_inlet_and_outlet(self):
        """Test that straight channel creates both inlet and outlet ports (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "x_min": -10, "x_max": 10,
                    "y_min": -10, "y_max": 10,
                    "z_min": -10, "z_max": 10,
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a straight channel through it",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        component_patch = next(
            (p for p in patches if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None
        
        component_value = component_patch.get("value", {})
        ports = component_value.get("ports", {})
        inlets = ports.get("inlets", [])
        outlets = ports.get("outlets", [])
        
        assert len(inlets) > 0, "Channel should have inlet"
        assert len(outlets) > 0, "Channel should have outlet"

    def test_parse_straight_channel_through(self):
        """Test parsing 'straight channel through it' creates primitive_channels component."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a straight channel through it",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        component_patch = next(
            (p for p in patches if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None
        
        component_value = component_patch.get("value", {})
        assert component_value.get("build", {}).get("type") == "primitive_channels"

    def test_parse_channel_with_radius(self):
        """Test parsing channel with explicit radius."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a straight channel through it with radius 0.3mm",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        component_patch = next(
            (p for p in patches if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None
        
        component_value = component_patch.get("value", {})
        inlets = component_value.get("ports", {}).get("inlets", [])
        assert len(inlets) > 0
        assert inlets[0].get("radius") == 0.3


class TestAgentRidgeParsing:
    """Tests for ridge feature parsing."""

    def test_parse_ridge_on_left_side(self):
        """Test parsing 'ridge on left side' adds ridge feature (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "x_min": -10, "x_max": 10,
                    "y_min": -10, "y_max": 10,
                    "z_min": -10, "z_max": 10,
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a ridge on left side",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            assert any(r.get("face") == "-x" for r in ridges)

    def test_parse_ridge_on_the_left(self):
        """Test parsing 'ridge on the left' adds ridge feature (Phase 3)."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "x_min": -10, "x_max": 10,
                    "y_min": -10, "y_max": 10,
                    "z_min": -10, "z_max": 10,
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a ridge on the left",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            assert any(r.get("face") == "-x" for r in ridges)

    def test_parse_ridge_on_left_face(self):
        """Test parsing 'ridge on left face' adds ridge feature."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a ridge on left face",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            assert any(r.get("face") == "-x" for r in ridges)

    def test_parse_ridge_on_right_face(self):
        """Test parsing 'ridge on right face' adds ridge feature."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add a ridge on right face",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            assert any(r.get("face") == "+x" for r in ridges)

    def test_parse_multiple_ridges(self):
        """Test parsing 'ridge on left and right faces' adds both ridges."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response = agent.process_message(
            "Add ridges on left face and right face",
            spec,
        )
        
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        assert response.patch_proposal is not None
        
        patches = response.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            faces = [r.get("face") for r in ridges]
            assert "-x" in faces
            assert "+x" in faces


class TestAgentCombinedParsing:
    """Tests for combined domain + component + ridge parsing."""

    def test_parse_cube_channel_ridge_combined(self):
        """Test that a message with cube, channel, and ridge generates all patches."""
        agent = DesignSpecAgent()
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        
        response = agent.process_message(
            "Create a box with 20 mm sides",
            spec,
        )
        assert response.response_type == AgentResponseType.PATCH_PROPOSAL
        
        patches = response.patch_proposal.patches
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None
        
        spec_with_domain = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test"},
            "policies": {},
            "domains": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [20, 20, 20],
                }
            },
            "components": [],
        }
        
        response2 = agent.process_message(
            "Add a straight channel through it",
            spec_with_domain,
        )
        assert response2.response_type == AgentResponseType.PATCH_PROPOSAL
        
        patches2 = response2.patch_proposal.patches
        component_patch = next(
            (p for p in patches2 if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None
        
        spec_with_component = dict(spec_with_domain)
        spec_with_component["components"] = [{
            "id": "channel_1",
            "domain_ref": "main_domain",
            "build": {"type": "primitive_channels"},
            "ports": {"inlets": [], "outlets": []},
        }]
        
        response3 = agent.process_message(
            "Add ridges on left face and right face",
            spec_with_component,
        )
        assert response3.response_type == AgentResponseType.PATCH_PROPOSAL
        
        patches3 = response3.patch_proposal.patches
        ridge_patch = next(
            (p for p in patches3 if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None

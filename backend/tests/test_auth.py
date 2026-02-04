"""Tests for authentication endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, SessionLocal
from app.api.auth import rate_limiter

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    """Create fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    # Clear rate limiter state between tests
    rate_limiter._memory_store.clear()
    yield
    Base.metadata.drop_all(bind=engine)

class TestRegistration:
    def test_register_success(self):
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["username"] == "testuser"

    def test_register_duplicate_username(self):
        client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "DifferentPass123"
        })
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_register_short_password(self):
        response = client.post("/api/auth/register", json={
            "username": "testuser",
            "password": "short"
        })
        assert response.status_code == 422  # Validation error

    def test_register_short_username(self):
        response = client.post("/api/auth/register", json={
            "username": "ab",
            "password": "TestPass123"
        })
        assert response.status_code == 422

class TestLogin:
    def test_login_success(self):
        # Register first
        client.post("/api/auth/register", json={
            "username": "loginuser",
            "password": "TestPass123"
        })
        # Then login
        response = client.post("/api/auth/login", json={
            "username": "loginuser",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_wrong_password(self):
        client.post("/api/auth/register", json={
            "username": "wrongpassuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/login", json={
            "username": "wrongpassuser",
            "password": "WrongPassword123"
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self):
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "TestPass123"
        })
        assert response.status_code == 401

class TestAuthenticatedEndpoints:
    def get_token(self):
        client.post("/api/auth/register", json={
            "username": "authuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/login", json={
            "username": "authuser",
            "password": "TestPass123"
        })
        return response.json()["access_token"]

    def test_get_me_authenticated(self):
        token = self.get_token()
        response = client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert response.status_code == 200
        assert response.json()["username"] == "authuser"

    def test_get_me_no_token(self):
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_get_me_invalid_token(self):
        response = client.get("/api/auth/me", headers={
            "Authorization": "Bearer invalid_token"
        })
        assert response.status_code == 401

class TestApiKeys:
    def get_token(self):
        client.post("/api/auth/register", json={
            "username": "apikeyuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/login", json={
            "username": "apikeyuser",
            "password": "TestPass123"
        })
        return response.json()["access_token"]

    def test_save_api_key(self):
        token = self.get_token()
        response = client.post("/api/auth/api-keys",
            json={"provider": "openai", "api_key": "sk-test123"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["provider"] == "openai"
        assert response.json()["has_key"] == True

    def test_get_api_keys(self):
        token = self.get_token()
        client.post("/api/auth/api-keys",
            json={"provider": "openai", "api_key": "sk-test123"},
            headers={"Authorization": f"Bearer {token}"}
        )
        response = client.get("/api/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_delete_api_key(self):
        token = self.get_token()
        client.post("/api/auth/api-keys",
            json={"provider": "openai", "api_key": "sk-test123"},
            headers={"Authorization": f"Bearer {token}"}
        )
        response = client.delete("/api/auth/api-keys/openai",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200

class TestPreferences:
    def get_token(self):
        client.post("/api/auth/register", json={
            "username": "prefsuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/login", json={
            "username": "prefsuser",
            "password": "TestPass123"
        })
        return response.json()["access_token"]

    def test_get_preferences(self):
        token = self.get_token()
        response = client.get("/api/auth/preferences",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert "theme" in response.json()

    def test_update_preferences(self):
        token = self.get_token()
        response = client.put("/api/auth/preferences",
            json={"theme": "light"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["theme"] == "light"

class TestPasswordChange:
    def get_token(self):
        client.post("/api/auth/register", json={
            "username": "changepassuser",
            "password": "TestPass123"
        })
        response = client.post("/api/auth/login", json={
            "username": "changepassuser",
            "password": "TestPass123"
        })
        return response.json()["access_token"]

    def test_change_password_success(self):
        token = self.get_token()
        response = client.post("/api/auth/change-password",
            json={
                "current_password": "TestPass123",
                "new_password": "NewTestPass456"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        # Verify new password works
        login_response = client.post("/api/auth/login", json={
            "username": "changepassuser",
            "password": "NewTestPass456"
        })
        assert login_response.status_code == 200

    def test_change_password_wrong_current(self):
        token = self.get_token()
        response = client.post("/api/auth/change-password",
            json={
                "current_password": "WrongPassword123",
                "new_password": "NewTestPass456"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 400

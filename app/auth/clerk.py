"""
app/auth/clerk.py
Clerk JWT authentication middleware for FastAPI.
Verifies Clerk-issued JWTs and extracts user identity.
Falls back to bypass mode if Clerk keys are not configured.
"""

from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser:
    """Represents a verified Clerk user."""

    def __init__(self, user_id: str, email: str = "", name: str = "") -> None:
        """
        Initialise an authenticated user.

        Args:
            user_id: Clerk user identifier.
            email: User email address.
            name: User display name.
        """
        self.user_id = user_id
        self.email   = email
        self.name    = name

    def __repr__(self) -> str:
        return f"AuthenticatedUser(user_id={self.user_id!r})"


async def _verify_clerk_token(token: str) -> dict:
    """
    Verify a Clerk JWT by calling the Clerk backend API.

    Args:
        token: JWT Bearer token from Authorization header.

    Returns:
        Decoded token payload dict.

    Raises:
        HTTPException 401: If token is invalid or expired.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            "https://api.clerk.com/v1/tokens/verify",
            headers={
                "Authorization": f"Bearer {settings.clerk_secret_key}",
                "Content-Type":  "application/json",
            },
            params={"token": token},
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )

    return response.json()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> AuthenticatedUser:
    """
    FastAPI dependency that extracts and verifies the current user.

    If AUTH is disabled (no Clerk keys or auth_bypass=True), returns
    a synthetic anonymous user for development/demo mode.

    Args:
        credentials: HTTP Bearer credentials from Authorization header.

    Returns:
        AuthenticatedUser instance.

    Raises:
        HTTPException 401: If authentication fails in production mode.
    """
    # Bypass mode for local development / hackathon demo
    if not settings.auth_enabled:
        logger.debug("Auth bypass active — returning anonymous user")
        return AuthenticatedUser(user_id="anonymous", email="demo@intelliresearch.ai")

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please include a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = await _verify_clerk_token(credentials.credentials)
        user_id = payload.get("sub", "unknown")
        logger.info("User authenticated", user_id=user_id)
        return AuthenticatedUser(
            user_id=user_id,
            email=payload.get("email", ""),
            name=payload.get("name", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth verification failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication verification failed.",
        ) from exc


# Convenience type alias for dependency injection in route handlers
CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]

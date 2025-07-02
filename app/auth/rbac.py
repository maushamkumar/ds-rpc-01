from typing import List, Dict
from app.logger.log import logging


logger = logging.getLogger(__name__)


class RolePermissions:
    """
    Handles role-based access control by mapping roles to accessible sources.
    """

    # Mapping of roles to their allowed sources
    PERMISSIONS: Dict[str, List[str]] = {
        "finance": ["finance", "general"],
        "marketing": ["marketing", "general"],
        "hr": ["hr", "general"],
        "engineering": ["engineering", "general"],
        "c_level": ["finance", "marketing", "hr", "engineering", "general"],
        "employee": ["general"],
        "general": ["general"]
    }

    @classmethod
    def get_allowed_sources(cls, role: str) -> List[str]:
        """
        Get all accessible sources for a given role.

        Args:
            role (str): Role of the user.

        Returns:
            List[str]: List of accessible sources. Defaults to ["general"] if role not found.
        """
        return cls.PERMISSIONS.get(role, ["general"])

    @classmethod
    def can_access_source(cls, role: str, source: str) -> bool:
        """
        Check if the given role has access to a particular source.

        Args:
            role (str): Role of the user.
            source (str): Source user is trying to access.

        Returns:
            bool: True if access is allowed, else False.
        """
        return source in cls.get_allowed_sources(role)

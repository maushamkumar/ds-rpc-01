from dataclasses import dataclass
from typing import Dict
import hashlib
import sys
from app.logger.log import logging
from app.exception.exception_handler import (
    AppException, 
    AuthenticationError, 
    UserNotFoundError, 
    PasswordMismatchError
)

logger = logging.getLogger(__name__)

@dataclass
class User:
    username: str
    password_hash: str
    role: str

class UserManager:
    def __init__(self):
        """
        Initialize the UserManager with a predefined set of users.
        """
        try:
            logger.info(f"{'='*20} UserManager Initialization Started {'='*20}")
            self.users: Dict[str, User] = self._initialize_users()
            logger.info("UserManager users initialized.")
        except Exception as e:
            logger.exception("Exception occurred during UserManager initialization.")
            raise AppException(e) from e

    def _initialize_users(self) -> Dict[str, User]:
        """
        Create a dictionary of predefined users with hashed passwords.

        Returns:
            Dict[str, User]: Dictionary containing username as key and User dataclass as value.
        """
        return {
            uname: self._create_user(uname, pwd, role)
            for uname, pwd, role in [
                ("finance_user", "finance123", "finance"),
                ("marketing_user", "marketing123", "marketing"),
                ("hr_user", "hr123", "hr"),
                ("engineering_user", "eng123", "engineering"),
                ("ceo", "ceo123", "c_level"),
                ("employee", "emp123", "employee"),
                ("general_user", "general123", "general")
            ]
        }

    def _create_user(self, username: str, password: str, role: str) -> User:
        """
        Create a User object with hashed password.

        Args:
            username (str): Username of the user.
            password (str): Plain text password.
            role (str): Role assigned to the user.

        Returns:
            User: A User dataclass instance.
        """
        return User(
            username=username,
            password_hash=self._hash_password(password),
            role=role
        )

    def _hash_password(self, password: str) -> str:
        """
        Hash the password using SHA-256.

        Args:
            password (str): Plain text password.

        Returns:
            str: SHA-256 hashed password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, username: str, password: str) -> Dict[str, str]:
        """
        Authenticate the user by verifying the username and password.

        Args:
            username (str): Input username.
            password (str): Input password.

        Raises:
            UserNotFoundError: If the username is not found.
            PasswordMismatchError: If the password does not match.

        Returns:
            Dict[str, str]: Dictionary containing the authenticated username and role.
        """
        logger.info(f"Authenticating user: {username}")
        user = self.users.get(username)

        if not user:
            raise UserNotFoundError(username)

        if user.password_hash != self._hash_password(password):
            raise PasswordMismatchError()

        logger.info(f"User '{username}' authenticated successfully.")
        return {"username": user.username, "role": user.role}

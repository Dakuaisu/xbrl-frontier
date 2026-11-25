"""Redact secret-service exception text before bounded diagnostic output."""
import re


def secret_error(exc: Exception) -> str:
    redacted = re.sub(r"\S+", "[redacted]", str(exc))
    return redacted[:160]

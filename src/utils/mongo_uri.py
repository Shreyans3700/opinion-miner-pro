"""MongoDB URI helpers."""

from __future__ import annotations

from urllib.parse import quote_plus, unquote_plus, urlsplit, urlunsplit


def normalize_mongo_uri(uri: str) -> str:
    """Escape username/password in MongoDB URI if required.

    This keeps already-escaped credentials stable by unquoting then requoting.
    """
    if not uri or not uri.startswith(("mongodb://", "mongodb+srv://")):
        return uri

    parts = urlsplit(uri)
    if "@" not in parts.netloc:
        return uri

    userinfo, hostinfo = parts.netloc.rsplit("@", 1)
    if ":" in userinfo:
        username, password = userinfo.split(":", 1)
        safe_user = quote_plus(unquote_plus(username))
        safe_password = quote_plus(unquote_plus(password))
        safe_userinfo = f"{safe_user}:{safe_password}"
    else:
        safe_userinfo = quote_plus(unquote_plus(userinfo))

    safe_netloc = f"{safe_userinfo}@{hostinfo}"
    return urlunsplit(
        (parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment)
    )

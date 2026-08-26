"""The API envelope every JSON endpoint answers with.

One shape for every response, success or failure, so the browser branches on
``status`` alone and never has to know which endpoint it called:

    {"status": true, "status_code": 200, "message": "", "data": {}}

``message`` is empty on success and carries a user-facing sentence on failure —
the browser prints it verbatim, so it is written for a reader, not a log.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import request

# Conventional codes: 200 success, 400 malformed body, 422 valid request that
# failed validation or processing, 404 unknown resource, 409 already running.
HTTP_OK = 200
HTTP_BAD_REQUEST = 400


def make_api_response(
    status: bool = True,
    message: str = "",
    data: dict[str, Any] | None = None,
    status_code: int = HTTP_OK,
    reflect_status: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Build the standard API envelope.

    Args:
        status: True on success.
        message: User-facing error message, empty on success.
        data: Result payload.
        status_code: HTTP status code, mirrored into the body.
        reflect_status: When True, return (body, status_code) so Flask sets the
            HTTP status as well as the body.

    Returns:
        The response dict, or a (dict, int) tuple when reflect_status is True.
    """
    response = {
        "status": status,
        "status_code": status_code,
        "message": message,
        "data": data if data is not None else {},
    }

    return (response, status_code) if reflect_status else response


def check_json_data(view: Callable) -> Callable:
    """Parse the JSON request body and pass it as the view's first argument.

    Returns a 400 envelope when the body is not valid JSON. Endpoints reading
    ``request.form`` or ``request.files`` must not use this: there is no JSON
    body for it to parse.

    Args:
        view: The view function to wrap.

    Returns:
        The wrapped view.
    """

    @wraps(view)
    def decorator(*args: Any, **kwargs: Any) -> Any:
        try:
            json_data = request.get_json()
        except Exception:  # noqa: BLE001 - any parse failure is the same answer
            return make_api_response(
                status=False,
                message="Wrong request format",
                data={},
                status_code=HTTP_BAD_REQUEST,
                reflect_status=True,
            )

        return view(json_data or {}, *args, **kwargs)

    return decorator

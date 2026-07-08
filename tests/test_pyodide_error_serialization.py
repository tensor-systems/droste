"""The Pyodide host-response must be JSON-serializable on the error path.

``run_rlm`` returns a ``RLMError`` dataclass; the Deno relay ``json.dumps`` the
host response. Without serializing the dataclass, that dump raises and the relay
emits no output — an opaque failure, and one that also drops the HTTP status the
host injects (402 = out of balance). These guard that regression.
"""

import json

from droste.loop.rlm import RLMError
from droste.substrates.pyodide import serialize_error


def test_serialize_error_dataclass_is_json_serializable() -> None:
    err = RLMError(type="JsException", message="ModelRelay HTTP 402: insufficient balance")
    serialized = serialize_error(err)
    assert serialized == {
        "type": "JsException",
        "message": "ModelRelay HTTP 402: insufficient balance",
        "code": None,
        "details": None,
    }
    # The crux: the host response must json.dumps without raising.
    json.dumps({"answer": None, "error": serialized})


def test_serialize_error_none_passthrough() -> None:
    assert serialize_error(None) is None


def test_serialize_error_dict_passthrough() -> None:
    assert serialize_error({"type": "X", "message": "y"}) == {"type": "X", "message": "y"}

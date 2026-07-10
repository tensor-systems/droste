import json
import sys
import traceback

from .runner import RUNNER_PROTOCOL_VERSION, main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {
            # Exception envelopes are responses too — a host parsing for the
            # documented protocol_version must find it on every output path.
            "protocol_version": RUNNER_PROTOCOL_VERSION,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.exit(1)

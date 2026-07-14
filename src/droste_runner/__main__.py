import json
import sys
import traceback

from .protocol import build_exception_response
from .run import main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = build_exception_response(exc, traceback.format_exc())
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.exit(1)

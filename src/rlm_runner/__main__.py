import json
import sys
import traceback

from .runner import main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.exit(1)

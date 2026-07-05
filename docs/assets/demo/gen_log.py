"""Deterministic demo log for the README GIF (seeded).

Writes ``server.log`` (~435 kB) into the target dir: routine payments-service
traffic with exactly one failed charge (cus_9982, $14.99, card_declined /
insufficient_funds) and 66 upstream timeouts blaming payments-v2 — so the
recorded answer is checkable against ground truth.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timezone

random.seed(42)

SERVICES = ["api-gw", "checkout", "payments", "auth", "catalog", "email"]
PATHS = ["/v1/charge", "/v1/checkout", "/v1/session", "/v1/products", "/v1/refund", "/healthz"]
TS_BASE = 1751265600
N = 3200
FAIL_SLOT = 1847


def ts(i: int) -> str:
    t = TS_BASE + i * 7 + random.randint(0, 5)
    stamp = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.")
    return stamp + f"{random.randint(0, 999):03d}Z"


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    timeout_slots = set(random.sample(range(200, N - 100), 66))
    lines: list[str] = []
    for i in range(N):
        t = ts(i)
        if i == FAIL_SLOT:
            rid = f"req_{random.randrange(16**10):010x}"
            lines.append(
                f"{t} ERROR payments charge_failed customer=cus_9982 amount=1499 "
                f"currency=usd reason=card_declined decline_code=insufficient_funds request_id={rid}"
            )
            lines.append(f"{t} WARN  checkout order_abandoned customer=cus_9982 cart_total=1499 step=payment")
            continue
        if i in timeout_slots:
            ms = random.randint(5001, 5450)
            rid = f"req_{random.randrange(16**10):010x}"
            lines.append(
                f"{t} ERROR api-gw upstream_timeout upstream=payments-v2 path=/v1/charge "
                f"latency_ms={ms} attempt={random.randint(1, 3)} request_id={rid}"
            )
            continue
        svc = random.choice(SERVICES)
        path = random.choice(PATHS)
        ms = random.randint(8, 900)
        cus = f"cus_{random.randint(1000, 9999)}"
        status = random.choice([200] * 18 + [201] * 3 + [404, 429])
        rid = f"req_{random.randrange(16**10):010x}"
        lines.append(
            f"{t} INFO  {svc} request path={path} status={status} customer={cus} "
            f"latency_ms={ms} request_id={rid}"
        )
        if random.random() < 0.06:
            amount = random.choice([499, 999, 1499, 2999, 4999])
            lines.append(f"{t} INFO  payments charge_succeeded customer={cus} amount={amount} currency=usd")

    target = os.path.join(outdir, "server.log")
    with open(target, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")


if __name__ == "__main__":
    main()

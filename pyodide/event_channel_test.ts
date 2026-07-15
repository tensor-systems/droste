import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import { closeSync, openSync } from "node:fs";
import {
  EventChannel,
  eventChannelFromEnvironment,
  RelayEventChannelError,
} from "../src/droste/substrates/_relay/event_channel.ts";

function assertChannelError(
  action: () => unknown,
  code: RelayEventChannelError["code"],
): void {
  const error = assertThrows(action, RelayEventChannelError);
  assertEquals(error.code, code);
  assertEquals(error.message, "dedicated relay event channel is unavailable");
}

Deno.test("event channel requires one explicit descriptor above fd2", () => {
  assertChannelError(
    () => eventChannelFromEnvironment(() => undefined, () => 0),
    "missing_descriptor",
  );
  for (const raw of ["", "-1", "1", "2", "3.0", " 3", "3 ", "fd3"]) {
    assertChannelError(
      () => eventChannelFromEnvironment(() => raw, () => 0),
      "invalid_descriptor",
    );
  }
});

Deno.test("event channel probes the descriptor and completes partial writes", () => {
  const received: number[] = [];
  const channel = eventChannelFromEnvironment(
    () => "3",
    (_descriptor, bytes) => {
      if (bytes.length === 0) return 0;
      const length = Math.min(2, bytes.length);
      received.push(...bytes.subarray(0, length));
      return length;
    },
    () => {},
  );

  channel.writeFrame('{"type":"progress"}');

  assertEquals(
    new TextDecoder().decode(new Uint8Array(received)),
    '{"type":"progress"}\n',
  );
  assertEquals(channel.failure, null);
});

Deno.test("event channel latches descriptor and frame write failures", () => {
  assertChannelError(
    () =>
      eventChannelFromEnvironment(
        () => "3",
        () => {
          throw new Error("private descriptor detail");
        },
        () => {},
      ),
    "descriptor_unavailable",
  );

  const channel = new EventChannel(3, () => {
    throw new Error("private write detail");
  });
  assertChannelError(() => channel.writeFrame("{}"), "write_failed");
  const firstFailure = channel.failure;
  assertChannelError(() => channel.writeFrame("{}"), "write_failed");
  assert(channel.failure === firstFailure);
  assertChannelError(
    () => new EventChannel(3, () => 1).writeFrame("{}\n{}"),
    "write_failed",
  );
  assertChannelError(
    () => new EventChannel(3, () => 1).writeFrame(""),
    "write_failed",
  );

  const partial: number[] = [];
  let writes = 0;
  const partialChannel = new EventChannel(3, (_descriptor, bytes) => {
    if (writes++ === 0) {
      partial.push(...bytes.subarray(0, 2));
      return 2;
    }
    throw new Error("peer closed after a partial frame");
  });
  assertChannelError(
    () => partialChannel.writeFrame('{"type":"progress"}'),
    "write_failed",
  );
  assertEquals(new TextDecoder().decode(new Uint8Array(partial)), '{"');
});

Deno.test("read-only descriptor fails on the first event frame", () => {
  const descriptor = openSync(new URL(import.meta.url), "r");
  try {
    const channel = eventChannelFromEnvironment(() => String(descriptor));
    assertChannelError(
      () => channel.writeFrame('{"type":"progress"}'),
      "write_failed",
    );
  } finally {
    closeSync(descriptor);
  }
});

import { assertEquals, assertRejects } from "jsr:@std/assert@1";
import { startProviderDuplex } from "../src/droste/substrates/_relay/provider_duplex.ts";

Deno.test("provider duplex pump serializes one frame and acknowledgement", async () => {
  const session = startProviderDuplex(async (emit) => {
    assertEquals(
      await emit('{"call_id":"call-1","kind":"checkpoint"}'),
      "ack-1",
    );
    assertEquals(
      await emit('{"call_id":"call-1","kind":"terminal"}'),
      "ack-2",
    );
  });

  assertEquals(
    await session.receive(),
    '{"call_id":"call-1","kind":"checkpoint"}',
  );
  session.requestCancellation("call-1");
  assertEquals(session.cancellation_requested("call-1"), true);
  assertEquals(session.cancellation_requested("call-2"), false);
  assertEquals(session.requestActiveCancellation(), true);
  await session.send("ack-1");
  assertEquals(
    await session.receive(),
    '{"call_id":"call-1","kind":"terminal"}',
  );
  await session.send("ack-2");
  await session.close();
});

Deno.test("provider duplex pump fails when remote ends without terminal", async () => {
  const session = startProviderDuplex(() => {});
  await assertRejects(() => session.receive(), Error, "without a terminal");
});

Deno.test("provider duplex pump retains cancellation before the first frame", async () => {
  let emitFirstFrame!: () => void;
  const firstFrame = new Promise<void>((resolve) => {
    emitFirstFrame = resolve;
  });
  const session = startProviderDuplex(async (emit) => {
    await firstFrame;
    await emit('{"call_id":"call-1","kind":"terminal"}');
  }, "call-1");

  assertEquals(session.requestActiveCancellation(), true);
  assertEquals(session.cancellation_requested("call-1"), true);
  emitFirstFrame();
  await session.receive();
  await session.send("ack");
  await session.close();
});

Deno.test("provider duplex pump rejects provider frame fan-out", async () => {
  const session = startProviderDuplex(async (emit) => {
    const first = emit('{"kind":"checkpoint"}');
    await assertRejects(
      () => emit('{"kind":"checkpoint"}'),
      Error,
      "before acknowledgement",
    );
    await first;
    await emit('{"kind":"terminal"}');
  });

  await session.receive();
  await session.send("ack-1");
  await session.receive();
  await session.send("ack-2");
  await session.close();
});

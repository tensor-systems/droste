import { eventChannelFromEnvironment } from "../src/droste/substrates/_relay/event_channel.ts";

const mode = Deno.args[0];
const channel = eventChannelFromEnvironment();

function frame(type: "startup" | "code", seq: number, body: object): string {
  return JSON.stringify({
    run_id: "event-channel-probe",
    seq,
    timestamp: "2026-07-15T00:00:00Z",
    type,
    version: 2,
    persistence_class: type === "startup" ? "transient" : "configurable",
    depth: 0,
    ...body,
  });
}

channel.writeFrame(frame("startup", 1, { engine_version: "test" }));

if (mode === "large") {
  for (let index = 0; index < 64; index += 1) {
    channel.writeFrame(
      frame("code", index + 2, {
        iteration: index + 1,
        code: `event-${index}:` + "e".repeat(65_536),
      }),
    );
    console.error(`diagnostic-${index}:` + "d".repeat(65_536));
  }
  await Deno.stdout.write(
    new TextEncoder().encode('{"answer":"ok","error":null}\n'),
  );
} else if (mode === "cancel") {
  console.error("event-channel-probe-ready");
  await new Promise(() => {});
} else if (mode === "fail") {
  console.error("event-channel-probe-process-failure");
  Deno.exit(17);
} else {
  throw new Error("unknown event channel probe mode");
}

import { eventChannelFromEnvironment } from "../src/droste/substrates/_relay/event_channel.ts";

const mode = Deno.args[0];
const channel = eventChannelFromEnvironment();
const fixture = await Deno.readTextFile(
  new URL(
    "../src/droste/testing/fixtures/trace-v3-lifecycle.ndjson",
    import.meta.url,
  ),
);
if (!fixture.endsWith("\n")) throw new Error("invalid Trace fixture");
const frames = fixture.slice(0, -1).split("\n");

if (mode === "large") {
  for (let repetition = 0; repetition < 128; repetition += 1) {
    for (const frame of frames) channel.writeFrame(frame);
    console.error(`diagnostic-${repetition}:` + "d".repeat(32_768));
  }
  await Deno.stdout.write(
    new TextEncoder().encode('{"answer":"ok","error":null}\n'),
  );
} else if (mode === "cancel") {
  channel.writeFrame(frames[0]);
  console.error("event-channel-probe-ready");
  await new Promise(() => {});
} else if (mode === "fail") {
  channel.writeFrame(frames[0]);
  console.error("event-channel-probe-process-failure");
  Deno.exit(17);
} else {
  throw new Error("unknown event channel probe mode");
}

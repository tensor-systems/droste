// The relay's one canonical Trace ABI transport. The host opens a descriptor
// (fd3 by convention) and names it through DROSTE_RELAY_EVENT_FD. stdout and
// stderr remain owned by the unary response and diagnostics respectively.
import { writeSync } from "node:fs";

export const RELAY_EVENT_FD_ENV = "DROSTE_RELAY_EVENT_FD";

export type RelayEventChannelErrorCode =
  | "missing_descriptor"
  | "invalid_descriptor"
  | "descriptor_unavailable"
  | "write_failed";

export class RelayEventChannelError extends Error {
  readonly code: RelayEventChannelErrorCode;

  constructor(code: RelayEventChannelErrorCode) {
    super("dedicated relay event channel is unavailable");
    this.name = "RelayEventChannelError";
    this.code = code;
  }
}

type DescriptorWriter = (descriptor: number, bytes: Uint8Array) => number;

function parseDescriptor(raw: string | undefined): number {
  if (raw === undefined) {
    throw new RelayEventChannelError("missing_descriptor");
  }
  if (!/^[0-9]+$/.test(raw)) {
    throw new RelayEventChannelError("invalid_descriptor");
  }
  const descriptor = Number(raw);
  if (!Number.isSafeInteger(descriptor) || descriptor < 3) {
    throw new RelayEventChannelError("invalid_descriptor");
  }
  return descriptor;
}

export class EventChannel {
  readonly #descriptor: number;
  readonly #write: DescriptorWriter;
  #failure: RelayEventChannelError | null = null;

  constructor(descriptor: number, writer: DescriptorWriter = writeSync) {
    if (!Number.isSafeInteger(descriptor) || descriptor < 3) {
      throw new RelayEventChannelError("invalid_descriptor");
    }
    this.#descriptor = descriptor;
    this.#write = writer;
  }

  get failure(): RelayEventChannelError | null {
    return this.#failure;
  }

  probe(): void {
    if (this.#failure !== null) {
      throw this.#failure;
    }
    try {
      if (this.#write(this.#descriptor, new Uint8Array()) !== 0) {
        this.#fail("descriptor_unavailable");
      }
    } catch {
      this.#fail("descriptor_unavailable");
    }
  }

  writeFrame(frame: string): void {
    if (this.#failure !== null) {
      throw this.#failure;
    }
    if (frame.length === 0 || frame.includes("\n") || frame.includes("\r")) {
      this.#fail("write_failed");
    }

    const bytes = new TextEncoder().encode(frame + "\n");
    let offset = 0;
    try {
      while (offset < bytes.length) {
        const written = this.#write(this.#descriptor, bytes.subarray(offset));
        if (
          !Number.isInteger(written) || written <= 0 ||
          written > bytes.length - offset
        ) {
          this.#fail("write_failed");
        }
        offset += written;
      }
    } catch (error) {
      if (error instanceof RelayEventChannelError) {
        throw error;
      }
      this.#fail("write_failed");
    }
  }

  #fail(code: RelayEventChannelErrorCode): never {
    this.#failure ??= new RelayEventChannelError(code);
    throw this.#failure;
  }
}

export function eventChannelFromEnvironment(
  readEnvironment: (name: string) => string | undefined = Deno.env.get,
  writer: DescriptorWriter = writeSync,
): EventChannel {
  const channel = new EventChannel(
    parseDescriptor(readEnvironment(RELAY_EVENT_FD_ENV)),
    writer,
  );
  channel.probe();
  return channel;
}

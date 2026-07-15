// The relay's one canonical Trace ABI transport. The host opens a descriptor
// (fd3 by convention) and names it through DROSTE_RELAY_EVENT_FD. stdout and
// stderr remain owned by the unary response and diagnostics respectively.
import { fstatSync, writeSync } from "node:fs";

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
type DescriptorInspector = (descriptor: number) => void;

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

  writeFrame(frame: string): void {
    if (this.#failure !== null) {
      throw this.#failure;
    }
    if (frame.length === 0 || frame.includes("\n") || frame.includes("\r")) {
      throw new TypeError("event frame must be one non-empty line");
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
    } catch {
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
  inspect: DescriptorInspector = (descriptor) => {
    fstatSync(descriptor);
  },
): EventChannel {
  const descriptor = parseDescriptor(readEnvironment(RELAY_EVENT_FD_ENV));
  try {
    inspect(descriptor);
  } catch {
    throw new RelayEventChannelError("descriptor_unavailable");
  }
  return new EventChannel(descriptor, writer);
}

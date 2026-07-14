// Bounded per-call message pump for provider bridge v2.
//
// The receiver pulls one frame, applies it in its own interpreter, and pushes
// one acknowledgement before the provider may emit again. This avoids calling
// back into a suspended Pyodide interpreter and bounds queued data to one frame.

export interface ProviderDuplexSession {
  receive(): Promise<string>;
  send(ackJson: string): Promise<void>;
  cancellation_requested(callId: string): boolean;
  requestCancellation(callId: string): void;
  requestActiveCancellation(): boolean;
  close(): Promise<void>;
}

type Emit = (frameJson: string) => Promise<string>;
const MAX_FRAME_BYTES = 8 * 1024 * 1024;

class MessagePump implements ProviderDuplexSession {
  #frame: string | null = null;
  #receiveResolve: ((frame: string) => void) | null = null;
  #receiveReject: ((reason: unknown) => void) | null = null;
  #ackResolve: ((ack: string) => void) | null = null;
  #ackReject: ((reason: unknown) => void) | null = null;
  #failure: Error | null = null;
  #closed = false;
  #terminalEmitted = false;
  #activeCallId: string | null = null;
  #cancelledCallId: string | null = null;
  #doneResolve!: () => void;
  #done = new Promise<void>((resolve) => {
    this.#doneResolve = resolve;
  });

  constructor(activeCallId: string | null) {
    if (activeCallId !== null && !activeCallId) {
      throw new Error("duplex active call_id must be non-empty");
    }
    this.#activeCallId = activeCallId;
  }

  readonly emit: Emit = (frameJson) => {
    if (this.#closed) return Promise.reject(new Error("duplex session closed"));
    if (this.#failure) return Promise.reject(this.#failure);
    if (this.#frame !== null || this.#ackResolve !== null) {
      return Promise.reject(
        new Error("duplex provider emitted before acknowledgement"),
      );
    }
    if (new TextEncoder().encode(frameJson).byteLength > MAX_FRAME_BYTES) {
      return Promise.reject(new Error("duplex frame exceeds 8 MiB"));
    }
    const parsed = JSON.parse(frameJson);
    const callId = typeof parsed?.call_id === "string" ? parsed.call_id : null;
    if (callId !== null) {
      if (this.#activeCallId !== null && this.#activeCallId !== callId) {
        return Promise.reject(new Error("duplex frame call_id mismatch"));
      }
      if (this.#activeCallId === null) this.#activeCallId = callId;
    }
    if (parsed?.kind === "terminal") {
      if (this.#terminalEmitted) {
        return Promise.reject(
          new Error("duplex provider emitted more than one terminal"),
        );
      }
      this.#terminalEmitted = true;
    }
    const ack = new Promise<string>((resolve, reject) => {
      this.#ackResolve = resolve;
      this.#ackReject = reject;
    });
    if (this.#receiveResolve) {
      const resolve = this.#receiveResolve;
      this.#clearReceiver();
      resolve(frameJson);
    } else {
      this.#frame = frameJson;
    }
    return ack;
  };

  receive(): Promise<string> {
    if (this.#frame !== null) {
      const frame = this.#frame;
      this.#frame = null;
      return Promise.resolve(frame);
    }
    if (this.#failure) return Promise.reject(this.#failure);
    if (this.#closed) return Promise.reject(new Error("duplex session closed"));
    if (this.#receiveResolve) {
      return Promise.reject(
        new Error("duplex session already has a pending receive"),
      );
    }
    return new Promise<string>((resolve, reject) => {
      this.#receiveResolve = resolve;
      this.#receiveReject = reject;
    });
  }

  send(ackJson: string): Promise<void> {
    if (!this.#ackResolve) {
      return Promise.reject(
        new Error("duplex acknowledgement has no pending frame"),
      );
    }
    const resolve = this.#ackResolve;
    this.#clearAck();
    resolve(ackJson);
    return Promise.resolve();
  }

  cancellation_requested(callId: string): boolean {
    return this.#cancelledCallId === callId;
  }

  requestCancellation(callId: string): void {
    if (!callId) throw new Error("duplex cancellation requires call_id");
    if (this.#cancelledCallId !== null && this.#cancelledCallId !== callId) {
      throw new Error("duplex cancellation call_id mismatch");
    }
    this.#cancelledCallId = callId;
  }

  requestActiveCancellation(): boolean {
    if (this.#activeCallId === null) return false;
    this.requestCancellation(this.#activeCallId);
    return true;
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    const error = new Error("duplex session closed");
    this.#receiveReject?.(error);
    this.#ackReject?.(error);
    this.#clearReceiver();
    this.#clearAck();
    // Terminal acknowledgement normally lets the provider task finish in the
    // same event-loop turn. Never let a wedged/killed producer make receiver
    // cleanup unbounded.
    await Promise.race([
      this.#done,
      new Promise<void>((resolve) => setTimeout(resolve, 0)),
    ]);
  }

  fail(reason: unknown): void {
    if (this.#failure) return;
    this.#failure = reason instanceof Error
      ? reason
      : new Error(String(reason));
    this.#receiveReject?.(this.#failure);
    this.#ackReject?.(this.#failure);
    this.#clearReceiver();
    this.#clearAck();
    this.#doneResolve();
  }

  finish(): void {
    if (!this.#terminalEmitted) {
      this.fail(new Error("duplex provider ended without a terminal frame"));
    } else {
      this.#doneResolve();
    }
  }

  #clearReceiver(): void {
    this.#receiveResolve = null;
    this.#receiveReject = null;
  }

  #clearAck(): void {
    this.#ackResolve = null;
    this.#ackReject = null;
  }
}

export function startProviderDuplex(
  run: (emit: Emit) => Promise<void> | void,
  activeCallId: string | null = null,
): ProviderDuplexSession {
  const pump = new MessagePump(activeCallId);
  queueMicrotask(async () => {
    try {
      await run(pump.emit);
      pump.finish();
    } catch (error) {
      pump.fail(error);
    }
  });
  return {
    receive: () => pump.receive(),
    send: (ackJson: string) => pump.send(ackJson),
    cancellation_requested: (callId: string) =>
      pump.cancellation_requested(callId),
    requestCancellation: (callId: string) => pump.requestCancellation(callId),
    requestActiveCancellation: () => pump.requestActiveCancellation(),
    close: () => pump.close(),
  };
}

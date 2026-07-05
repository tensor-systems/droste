# Regenerating the demo GIF

Real run, real timing — no faked output. Requirements: `vhs` (brew), an
OpenAI-compatible key.

1. `python3 gen_log.py rec/` — writes the deterministic `server.log`
   (seeded: one failed charge for cus_9982, 66 payments-v2 timeouts).
2. Create `env.sh` beside the tape (never commit it):
   `export OPENAI_API_KEY=… OPENAI_BASE_URL=… DROSTE_MODEL=… PATH=<venv-bin>:$PATH`
3. `vhs demo.tape` → `demo.gif`. The tape Waits on the answer text, so
   run-length variance doesn't cut the recording.

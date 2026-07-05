# Regenerating the demo GIF

Real run, real timing — no faked output. Requirements: `vhs` (brew), an
OpenAI-compatible key.

1. `python3 gen_log.py rec/` — writes the deterministic `server.log`
   (seeded: one failed charge for cus_9982, 66 payments-v2 timeouts).
2. Create `env.sh` beside the tape (never commit it):
   `export OPENAI_API_KEY=… OPENAI_BASE_URL=… DROSTE_MODEL=… PATH=<venv-bin>:$PATH`
3. `vhs demo.tape` → raw gif; shrink for the README:
   `ffmpeg -i demo.gif -vf "setpts=0.75*PTS,fps=8,scale=840:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=8[p];[s1][p]paletteuse=dither=none" d8.gif && gifsicle -O3 --lossy=120 d8.gif -o demo.gif`
   The tape Waits for the prompt to return, so run-length variance can't
   cut the recording.

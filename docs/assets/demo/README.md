# Regenerating the demo GIF

Real run, real timing — no faked output. Requirements: `vhs` (brew), an
OpenAI-compatible key.

1. `python3 gen_log.py rec/` — writes the deterministic `server.log`
   (seeded: one failed charge for cus_9982, 66 payments-v2 timeouts).
2. Create `env.sh` beside the tape (never commit it):
   `export OPENAI_API_KEY=… OPENAI_BASE_URL=… DROSTE_MODEL=… PATH=<venv-bin>:$PATH`
3. `vhs demo.tape` → raw gif; shrink for the README:
   `ffmpeg -i demo.gif -vf "setpts=0.75*PTS,fps=10,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=none" d8.gif && gifsicle -O3 --lossy=40 d8.gif -o demo.gif`
   The tape Waits for the prompt to return, so run-length variance can't
   cut the recording.

Encode at the tape's native resolution — never rescale. The tape renders
1600px wide at a 24pt font precisely so text survives GIF quantization and
reads crisply on retina; the fuzz trifecta to avoid is downscaling (resamples
every glyph), a tiny palette (max_colors=8 destroys font antialiasing —
especially on the dark theme), and aggressive gifsicle lossy. 64 colors /
lossy=40 lands ~4 MB, well under GitHub's render limit.

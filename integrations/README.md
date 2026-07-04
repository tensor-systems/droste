# Integrations: Droste as a harness skill

Droste is an engine, not an agent. A general-purpose agent (OpenClaw, Hermes,
Claude Code, ...) can **delegate to `droste`** when a question outgrows tool
calls — i.e. needs recursive computation over a dataset too large for the
context window.

Each subdirectory is a ready-to-drop skill/plugin for one harness.

- `openclaw/SKILL.md` — an [OpenClaw](https://github.com/openclaw/openclaw)
  AgentSkill (also compatible with the Anthropic/Codex `SKILL.md` convention used
  by several harnesses). Teaches the agent when to call `droste ask` /
  `droste ask --db` and how to bound cost. Requires `droste` on PATH
  (`uv tool install droste`).

To use with OpenClaw: copy `openclaw/SKILL.md` to
`~/.config/openclaw/skills/droste/SKILL.md` (or the harness's skills dir), or
fork OpenClaw and upstream it under `skills/droste/`.

#!/usr/bin/env python3
"""Export a Claude Code session transcript to readable Markdown, with secrets redacted.

The brief asks for the AI transcripts alongside the code. Claude Code stores each
session as JSONL under ~/.claude/projects/<sanitised-cwd>/, which is complete but
unreadable, and -- importantly -- may contain secrets: anything printed to a
terminal or shown in a file-change notification is captured verbatim, so a .env
edit puts the API key straight into the log.

    python export_transcript.py                        # newest session -> transcript.md
    python export_transcript.py -o ~/convo.md          # choose the output path
    python export_transcript.py --jsonl                # also write redacted JSONL
    python export_transcript.py --full                 # do not truncate tool output

Redaction is applied to every output. Always skim the result before sharing it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECTS_DIR = Path.home() / ".claude" / "projects"

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{6,}"), "sk-ant-REDACTED"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "ghp_REDACTED"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA_REDACTED"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "sk-REDACTED"),
    # Private keys, in any armour. A session log captures whatever was printed to
    # the terminal, so one `cat ~/.ssh/id_ed25519` would otherwise publish a key
    # verbatim -- and these transcripts are committed to a public repo.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[PRIVATE KEY REDACTED]",
    ),
]
# Deliberately no email redaction. These transcripts are emailed to the reviewer
# rather than published (convo/ is gitignored), so scrubbing the recruiter's own
# address out of a document addressed to them costs readability and buys nothing.
# Re-add a pattern here if the transcripts ever become public.

TOOL_OUTPUT_LIMIT = 1_500


def redact(text: str) -> tuple[str, int]:
    """Strip credentials. Returns ``(clean_text, redaction_count)``."""
    total = 0
    for pattern, replacement in SECRET_PATTERNS:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


def project_dir(cwd: Path) -> Path:
    """Claude Code sanitises the cwd into the directory name by replacing / with -."""
    return PROJECTS_DIR / str(cwd.resolve()).replace("/", "-")


def newest_session(directory: Path) -> Path:
    sessions = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        raise SystemExit(f"No .jsonl transcripts in {directory}")
    return sessions[0]


def blocks_of(message: dict) -> list:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def render(path: Path, *, full: bool) -> tuple[str, int]:
    lines: list[str] = [
        f"# Claude Code transcript — `{path.stem}`",
        "",
        "Exported with `export_transcript.py`. Credentials are redacted; tool output is "
        + ("shown in full." if full else f"truncated to {TOOL_OUTPUT_LIMIT:,} characters."),
        "",
        "---",
        "",
    ]
    redactions = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        rendered: list[str] = []
        for block in blocks_of(message):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")

            if kind == "text" and block.get("text", "").strip():
                text, n = redact(block["text"])
                redactions += n
                rendered.append(text.rstrip())

            elif kind == "thinking" and block.get("thinking", "").strip():
                text, n = redact(block["thinking"])
                redactions += n
                rendered.append(
                    "<details><summary>thinking</summary>\n\n"
                    f"```\n{text.rstrip()}\n```\n\n</details>"
                )

            elif kind == "tool_use":
                payload, n = redact(json.dumps(block.get("input", {}), indent=2)[:TOOL_OUTPUT_LIMIT])
                redactions += n
                rendered.append(f"**→ {block.get('name', '?')}**\n\n```json\n{payload}\n```")

            elif kind == "tool_result":
                body = block.get("content")
                if isinstance(body, list):
                    body = "\n".join(
                        b.get("text", "") for b in body if isinstance(b, dict)
                    )
                body = str(body or "")
                if not full and len(body) > TOOL_OUTPUT_LIMIT:
                    body = body[:TOOL_OUTPUT_LIMIT] + f"\n… [{len(body) - TOOL_OUTPUT_LIMIT:,} more characters]"
                body, n = redact(body)
                redactions += n
                rendered.append(
                    f"<details><summary>tool result</summary>\n\n```\n{body}\n```\n\n</details>"
                )

        if rendered:
            lines.append(f"### {'User' if role == 'user' else 'Assistant'}")
            lines.append("")
            lines.extend(rendered)
            lines.append("")

    return "\n".join(lines), redactions


def render_txt(path: Path) -> tuple[str, int]:
    """Plain dialogue: 'User:' / 'AI:' turns, one per block, blank-line separated.

    Tool *calls* appear as one-line markers so the flow still makes sense; tool
    *results* are omitted, since they dominate the byte count and the point of
    this format is readability. Use the Markdown or JSONL export when the full
    record is wanted.
    """
    out: list[str] = []
    redactions = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        speaker = "User" if role == "user" else "AI"
        for block in blocks_of(message):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                text, n = redact(block["text"].strip())
                redactions += n
                out.append(f"{speaker}: {text}")
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                args = block.get("input") or {}
                hint = args.get("command") or args.get("file_path") or args.get("query") or ""
                hint, n = redact(str(hint).split("\n")[0][:100])
                redactions += n
                out.append(f"AI: [tool: {name}{(' ' + hint) if hint else ''}]")

    return "\n\n".join(out) + "\n", redactions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="path to a .jsonl (default: newest for this project)")
    parser.add_argument("-o", "--output", default="transcript.md")
    parser.add_argument("--jsonl", action="store_true", help="also write a redacted copy of the raw JSONL")
    parser.add_argument("--full", action="store_true", help="do not truncate tool output")
    parser.add_argument("--format", choices=("md", "txt"), default="md")
    parser.add_argument("--all", action="store_true",
                        help="concatenate every session for this project, oldest first")
    args = parser.parse_args()

    if args.session:
        paths = [Path(args.session).expanduser()]
    elif args.all:
        paths = sorted(project_dir(Path.cwd()).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    else:
        paths = [newest_session(project_dir(Path.cwd()))]

    chunks: list[str] = []
    redactions = 0
    for path in paths:
        if args.format == "txt":
            body, n = render_txt(path)
            header = f"===== session {path.stem} =====\n\n" if len(paths) > 1 else ""
        else:
            body, n = render(path, full=args.full)
            header = ""
        chunks.append(header + body)
        redactions += n

    document = ("\n\n" if args.format == "txt" else "\n\n---\n\n").join(chunks)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    for path in paths:
        print(f"Source    : {path}")
    print(f"Output    : {out}  ({len(document):,} chars, format={args.format})")

    if args.jsonl:
        clean, n = redact("\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in paths))
        redactions += n
        jsonl_out = out.with_suffix(".jsonl")
        jsonl_out.write_text(clean, encoding="utf-8")
        print(f"JSONL     : {jsonl_out}")

    print(f"Redactions: {redactions} credential(s) removed")
    if redactions:
        print("\n  A live credential was present in the transcript. It is removed from the")
        print("  exports above, but it still exists in the original log on disk -- and if")
        print("  it was ever a real key, treat it as exposed and rotate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

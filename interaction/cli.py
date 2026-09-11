"""Terminal client for a pending interactive provider request."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .contracts import VERSION
from .store import InteractionConflict, InteractionError, InteractionStore


def describe(request, draft: list[dict]) -> None:
    print(f"\n{request.label} · {request.agent} · episode {request.episode} · turn {request.turn}")
    supplied = request.input
    if supplied["kind"] == "initial_observation":
        print("\nInitial observation:\n" + supplied.get("text", ""))
    else:
        print("\nTool results:")
        for result in supplied.get("results", []):
            state = "error" if result.get("is_error") else "result"
            print(f"  {result.get('tool_call_id')} ({state}): {result.get('content', '')}")
    print("\nAvailable tools:")
    for tool in request.available_tools:
        schema = tool.input_schema
        required = set(schema.get("required", []))
        fields = []
        for name, spec in schema.get("properties", {}).items():
            allowed = spec.get("enum")
            shape = " | ".join(map(str, allowed)) if allowed else spec.get("type", "value")
            fields.append(f"{name}{'*' if name in required else ''}: {shape}")
        print(f"  {tool.name} — {tool.description}")
        print(f"    {', '.join(fields) if fields else 'JSON object'}")
    print(f"\nDraft: {json.dumps(draft, ensure_ascii=False, indent=2) if draft else 'empty'}")


def run(agent: str, root: Path) -> int:
    store = InteractionStore(root)
    request = store.current(agent)
    draft = store.load_draft(agent, request.request_id) if request else []
    if request:
        describe(request, draft)
    else:
        print(f"No pending interaction for {agent}.")
    print("\nCommands: tools, call <tool-name> <JSON-object>, draft, remove <number>, "
          "submit, done, refresh, quit")
    while True:
        try:
            line = input("human> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        command, _, rest = line.partition(" ")
        try:
            if command in ("quit", "q"):
                return 0
            if command in ("tools", "refresh"):
                request = store.current(agent)
                draft = store.load_draft(agent, request.request_id) if request else []
                if request:
                    describe(request, draft)
                else:
                    print("No pending interaction.")
                continue
            if request is None:
                print("No pending interaction; use refresh after the episode starts.")
                continue
            if command == "call":
                name, separator, body = rest.partition(" ")
                if not separator:
                    print("usage: call <tool-name> <JSON-object>")
                    continue
                value = json.loads(body)
                if not isinstance(value, dict):
                    print("tool input must be a JSON object")
                    continue
                if name not in {tool.name for tool in request.available_tools}:
                    print(f"unknown tool {name!r}")
                    continue
                draft.append({"id": f"human-{uuid.uuid4().hex}", "name": name, "input": value})
                store.save_draft(agent, request.request_id, draft)
                print(f"drafted call {len(draft)}")
            elif command == "draft":
                print(json.dumps(draft, ensure_ascii=False, indent=2) if draft else "empty")
            elif command == "remove":
                number = int(rest)
                if number < 1 or number > len(draft):
                    print("no such draft call")
                    continue
                draft.pop(number - 1)
                store.save_draft(agent, request.request_id, draft)
                print("removed")
            elif command in ("submit", "done"):
                if command == "submit" and not draft:
                    print("draft is empty; use done to finish without calls")
                    continue
                payload = {"version": VERSION, "request_id": request.request_id,
                           "submission_id": f"human-{uuid.uuid4().hex}",
                           "action": "tool_calls" if command == "submit" else "end_turn",
                           "tool_calls": draft if command == "submit" else []}
                won = store.submit(agent, request.request_id, payload)
                print(f"submitted {won.action}; use refresh for the next turn")
                request = None
                draft = []
            elif command:
                print("unknown command")
        except (ValueError, json.JSONDecodeError) as error:
            print(f"invalid value: {error}")
        except InteractionConflict as error:
            print(f"not submitted: {error}")
        except InteractionError as error:
            print(f"request changed: {error}")
            request = store.current(agent)
            draft = store.load_draft(agent, request.request_id) if request else []
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="agent identifier to control")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "interactions",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run(args.agent, args.root)


if __name__ == "__main__":
    sys.exit(main())

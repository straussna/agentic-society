# Declared invariants

The nine invariants are the README table, and docs/design.md states each in full; harness.py enforces them, and findings cite the invariant number. The anthropic and openai SDKs are imported inside providers/anthropic.py and providers/openai.py; each adapter refuses a `*_BASE_URL` and checks its API key in `preflight`. Sandbox containers run with `--network none`, and view.py's http.server serves a read-only dashboard on 127.0.0.1. check.py removes only containers that carry its own pid.

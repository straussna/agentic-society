# Module structure

harness.py is a single module that hashes itself at import and records the digest in every trace; its docstring is a table of contents of its sections. Judge its responsibilities by section, not by file size. Its tunables are module globals set from config.toml by `apply_config`.

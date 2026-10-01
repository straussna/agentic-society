# Module structure

harness.py is a single module that hashes itself at import and records the digest in every trace; its docstring is a table of contents of its sections. Judge its responsibilities by section, not by file size. Its tunables are the fields of one frozen `Settings` held in `harness.SETTINGS`, built by `read_config`, `overlay`, `with_channels` and `with_tools` and replaced whole by `load_config`, `apply_config`, `apply_channels`, `apply_tools` and `start()`.

# Deferred imports

harness.py imports experiment inside functions on the `--manifest` and CLI paths, and providers/human.py imports interaction.store inside methods. Both edges run only after both modules are fully imported.

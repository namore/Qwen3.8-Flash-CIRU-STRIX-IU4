# CIRU Flash IU4 v5.0.1 runtime

Source runtime for AMD Strix Halo (gfx1151). Start with [INSTALL.md](INSTALL.md).

This package retains the qualified inference source and adds public build and
launch instructions. Compact expert weights stay resident in packed form; long
prefill uses temporary BF16 tiles, while short expert operations use native
four-bit integer dot products. The SSD sidecar supplies the model's n-gram/hash
embedding table. MTP depth defaults to 3, with one request slot.

See [source identity](SOURCE_IDENTITY.json), [LICENSE](LICENSE), and
[third-party notices](THIRD_PARTY_NOTICES.md). All model benchmarks and weights
are in the [model repository](https://huggingface.co/jcbtc/Qwen3.8-Flash-CIRU-STRIX-IU4).

The supplied source includes prior upstream backend and kernel identifiers;
users do not need to set those switches manually. Use the release launcher.

v5.0.1 restores the image/MTP QSA position-gap correction. Image input
is enabled by default for MODEL_RELEASE=5.0 with the matching projector;
set ENABLE_VISION=0 for text-only operation. Model weights are unchanged.

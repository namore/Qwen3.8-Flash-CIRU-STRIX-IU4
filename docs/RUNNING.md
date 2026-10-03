# Running v5.0.1

See [the complete installation and API guide](../INSTALL.md).

Set `SSD_CACHE_DIR` to enable the persistent text and image prompt cache. `SSD_CACHE_MIB` limits its total size and defaults to 32768 MiB in the CIRU launcher. Set it explicitly when a strict storage budget is required.
`SSD_CACHE_PREFIX_ADVANTAGE` controls the minimum prefix improvement for admission and restore; 8192 tokens is the default. Entries include target, draft and MTP state and are isolated by model, PLE sidecars, draft model and runtime state format. Complete images use their source-byte hash, token count and position count. Audio and video bypass this cache. Changing the projector or image preprocessing settings invalidates snapshots.

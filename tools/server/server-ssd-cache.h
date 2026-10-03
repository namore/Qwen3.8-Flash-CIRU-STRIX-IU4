// Persistent, single-owner prompt-state cache used by llama-server.
//
// This interface intentionally exposes handles and complete state entry only.
// File layout, hashing, locking, recovery and retention policy are private to
// the implementation.

#pragma once

#include "common.h"
#include "llama.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct common_speculative;
struct server_tokens;

struct server_ssd_cache_digest {
    std::array<uint8_t, 16> bytes = {};

    bool operator == (const server_ssd_cache_digest & other) const {
        return bytes == other.bytes;
    }
};

struct server_ssd_cache_config {
    std::string directory;
    uint64_t max_bytes = 0;
    int32_t prefix_advantage = 0;
};

struct server_ssd_cache_compatibility {
    // These are SHA-256 prefixes, not build identifiers.  A llama.cpp build
    // change is deliberately not a compatibility gate; failed restores are.
    server_ssd_cache_digest model;
    server_ssd_cache_digest adapters;
    // Set only for a speculative configuration with a complete persistent
    // state contract.  This keeps ordinary, separate-KV, and shared-KV
    // snapshots from ever sharing a cache entry.
    server_ssd_cache_digest draft_model;
    server_ssd_cache_digest projector;
    int32_t image_min_tokens = -1;
    int32_t image_max_tokens = -1;
    uint32_t projector_settings = 0;
    uint32_t state_layout_version = 3;
    uint32_t speculative_state_version = 0;
    uint32_t speculative_kv_topology = 0;
    uint32_t n_ctx = 0;
    uint32_t n_batch = 0;
    uint32_t kv_type = 0;
};

struct server_ssd_cache_handle {
    uint64_t id = 0;
    uint32_t n_tokens = 0;
    server_ssd_cache_digest digest;

    explicit operator bool() const { return id != 0; }
};

struct server_ssd_cache_stats {
    uint64_t bytes = 0;
    uint64_t entries = 0;
    uint64_t lookup_hits = 0;
    uint64_t lookup_misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t write_failures = 0;
    uint64_t busy_drops = 0;
    uint64_t capture_us = 0;
    uint64_t restore_us = 0;
};

// Returns a digest over a versioned, domain-separated sequence of complete
// token prefixes.  The same input produces the same result regardless of how
// it is split across update() calls.
class server_ssd_prefix_hasher {
public:
    server_ssd_prefix_hasher();
    server_ssd_prefix_hasher(const server_ssd_prefix_hasher &) = delete;
    server_ssd_prefix_hasher & operator = (const server_ssd_prefix_hasher &) = delete;
    ~server_ssd_prefix_hasher();

    void update(llama_token token);
    void update_image(const std::string & id, uint32_t n_tokens, uint32_t n_pos);
    server_ssd_cache_digest finish() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

server_ssd_cache_digest server_ssd_prefix_digest(const llama_tokens & tokens);
server_ssd_cache_digest server_ssd_cache_file_identity(const std::string & path);
server_ssd_cache_digest server_ssd_cache_files_identity(const std::vector<std::string> & paths);
server_ssd_cache_digest server_ssd_cache_adapter_identity(const std::vector<common_adapter_lora_info> & adapters);

// SSD only displaces a volatile candidate when it clears the configured token
// advantage.  A tie remains in RAM (and the caller retains active-over-RAM
// tie-breaking).
bool server_ssd_cache_should_restore(uint32_t active_tokens, uint32_t ram_tokens, uint32_t ssd_tokens, uint32_t prefix_advantage);
bool server_ssd_cache_should_enter(uint32_t candidate_tokens, uint32_t ancestor_tokens, uint32_t prefix_advantage);

class server_ssd_prompt_cache {
public:
    server_ssd_prompt_cache(const server_ssd_cache_config & config, const server_ssd_cache_compatibility & compatibility);
    ~server_ssd_prompt_cache();

    server_ssd_prompt_cache(const server_ssd_prompt_cache &) = delete;
    server_ssd_prompt_cache & operator = (const server_ssd_prompt_cache &) = delete;

    bool enabled() const;
    server_ssd_cache_handle lookup(const llama_tokens & tokens, bool require_strict_prefix = true);
    server_ssd_cache_handle lookup(const server_tokens & tokens, bool require_strict_prefix = true);

    // Captures the current complete target, optional independent draft, and
    // optional speculative implementation state.
    // A busy writer, duplicate, insufficient prefix advance, or failed
    // admission is a normal no-op and never delays prompt processing.
    void enter(const llama_tokens & tokens, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id);
    void enter(const server_tokens & tokens, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id);

    // Restores only the selected lookup result.  On failure the caller can
    // continue normal prompt processing; the corrupt entry is retired.
    bool restore(const server_ssd_cache_handle & handle, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id);

    server_ssd_cache_stats stats() const;
    void shutdown();

#if defined(LLAMA_SERVER_SSD_CACHE_TESTING)
    // Synthetic metadata-only admission hooks used by the dedicated cache
    // module test.  They are not part of production builds or server APIs.
    bool test_admit(const llama_tokens & tokens, uint64_t target_size);
    uint64_t test_parent_id(const llama_tokens & tokens) const;
    bool test_record_restore(const llama_tokens & tokens);
#endif

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

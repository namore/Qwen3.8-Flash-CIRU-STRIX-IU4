#include "server-ssd-cache.h"
#include "server-common.h"

#include <filesystem>
#include <fstream>
#include <cstring>

namespace {

#pragma pack(push, 1)
struct test_snapshot_header {
    char magic[8];
    uint32_t format_version;
    uint32_t header_size;
    uint64_t id;
    uint64_t parent_id;
    uint8_t model[16];
    uint8_t adapters[16];
    uint8_t draft_model[16];
    uint8_t projector[16];
    int32_t image_min_tokens;
    int32_t image_max_tokens;
    uint32_t projector_settings;
    uint32_t state_layout_version;
    uint32_t speculative_state_version;
    uint32_t speculative_kv_topology;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t kv_type;
    uint32_t n_tokens;
    uint8_t prefix_digest[16];
    uint64_t target_size;
    uint64_t draft_size;
    uint64_t speculative_size;
    uint8_t payload_sha256[32];
};

struct test_lineage_header {
    char magic[8];
    uint32_t format_version;
    uint32_t record_size;
    uint64_t n_records;
};

struct test_lineage_record {
    uint64_t id;
    uint64_t parent_id;
    uint32_t n_tokens;
    uint8_t prefix_digest[16];
};
#pragma pack(pop)

static_assert(sizeof(test_snapshot_header) == 208, "test snapshot format must match the fixed cache header");
static_assert(sizeof(test_lineage_header) == 24, "test lineage header must match the cache header");
static_assert(sizeof(test_lineage_record) == 36, "test lineage record must match the cache record");

void write_snapshot(
        const std::filesystem::path & directory,
        uint64_t id,
        uint64_t parent_id,
        const llama_tokens & tokens,
        const server_ssd_cache_compatibility & compatibility,
        uint64_t target_bytes = 16,
        uint64_t draft_bytes = 0,
        uint64_t speculative_bytes = 0) {
    test_snapshot_header header = {};
    memcpy(header.magic, "LSSDC03", 8);
    header.format_version = 3;
    header.header_size = sizeof(header);
    header.id = id;
    header.parent_id = parent_id;
    memcpy(header.model, compatibility.model.bytes.data(), sizeof(header.model));
    memcpy(header.adapters, compatibility.adapters.bytes.data(), sizeof(header.adapters));
    memcpy(header.draft_model, compatibility.draft_model.bytes.data(), sizeof(header.draft_model));
    memcpy(header.projector, compatibility.projector.bytes.data(), sizeof(header.projector));
    header.image_min_tokens = compatibility.image_min_tokens;
    header.image_max_tokens = compatibility.image_max_tokens;
    header.projector_settings = compatibility.projector_settings;
    header.state_layout_version = compatibility.state_layout_version;
    header.speculative_state_version = compatibility.speculative_state_version;
    header.speculative_kv_topology = compatibility.speculative_kv_topology;
    header.n_ctx = compatibility.n_ctx;
    header.n_batch = compatibility.n_batch;
    header.kv_type = compatibility.kv_type;
    header.n_tokens = uint32_t(tokens.size());
    const auto digest = server_ssd_prefix_digest(tokens);
    memcpy(header.prefix_digest, digest.bytes.data(), sizeof(header.prefix_digest));
    header.target_size = target_bytes;
    header.draft_size = draft_bytes;
    header.speculative_size = speculative_bytes;

    std::ofstream output(directory / (std::to_string(id) + ".snapshot"), std::ios::binary);
    output.write(reinterpret_cast<const char *>(&header), sizeof(header));
    for (uint64_t i = 0; i < target_bytes + draft_bytes + speculative_bytes; ++i) output.put(0);
}

server_ssd_cache_config cache_config(const std::filesystem::path & directory, uint64_t max_bytes = 1024 * 1024) {
    server_ssd_cache_config config;
    config.directory = directory.string();
    config.max_bytes = max_bytes;
    config.prefix_advantage = 1;
    return config;
}

} // namespace

int main() {
    const llama_tokens tokens = { 1, 42, -7, 99 };

    server_ssd_prefix_hasher incremental;
    for (const auto token : tokens) {
        incremental.update(token);
    }
    const auto digest = server_ssd_prefix_digest(tokens);
    if (!(incremental.finish() == digest)) return 1;

    auto changed = tokens;
    changed.back()++;
    if (server_ssd_prefix_digest(tokens) == server_ssd_prefix_digest(changed)) return 1;

    // A prefix digest is unambiguous even when two prefixes have the same
    // token bytes before their final element.
    changed.pop_back();
    if (server_ssd_prefix_digest(tokens) == server_ssd_prefix_digest(changed)) return 1;

    server_ssd_prefix_hasher image_a;
    image_a.update(1);
    image_a.update_image(std::string(64, 'a'), 4, 2);
    server_ssd_prefix_hasher image_b;
    image_b.update(1);
    image_b.update_image(std::string(64, 'b'), 4, 2);
    if (image_a.finish() == image_b.finish()) return 1;
    server_ssd_prefix_hasher image_other_pos;
    image_other_pos.update(1);
    image_other_pos.update_image(std::string(64, 'a'), 4, 3);
    if (image_a.finish() == image_other_pos.finish()) return 1;

    if (server_ssd_cache_should_restore(100, 110, 119, 10)) return 1;
    if (!server_ssd_cache_should_restore(100, 110, 120, 10)) return 1;
    if (server_ssd_cache_should_restore(100, 110, 110, 1)) return 1;
    if (server_ssd_cache_should_enter(119, 110, 10)) return 1;
    if (!server_ssd_cache_should_enter(120, 110, 10)) return 1;

    server_ssd_cache_compatibility compatibility;
    server_ssd_cache_config disabled;
    server_ssd_prompt_cache no_cache(disabled, compatibility);
    if (no_cache.enabled()) return 1;
    if (no_cache.lookup(tokens)) return 1;

    const auto directory = std::filesystem::temp_directory_path() / "llama-server-ssd-cache-test";
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::create_directories(directory);

    // Model and adapter identities are stable for the same inputs and change
    // when a compatibility-relevant input changes.
    const auto identity_file = directory / "identity.gguf";
    std::ofstream(identity_file) << "one";
    const auto model_identity = server_ssd_cache_file_identity(identity_file.string());
    if (!(model_identity == server_ssd_cache_file_identity(identity_file.string()))) return 1;
    const auto files_identity = server_ssd_cache_files_identity({ identity_file.string() });
    if (!(files_identity == server_ssd_cache_files_identity({ identity_file.string() }))) return 1;
    std::ofstream(identity_file, std::ios::trunc) << "a different model";
    if (model_identity == server_ssd_cache_file_identity(identity_file.string())) return 1;
    if (files_identity == server_ssd_cache_files_identity({ identity_file.string() })) return 1;
    common_adapter_lora_info adapter { identity_file.string(), 1.0f, "", "prefix", nullptr };
    const auto adapter_identity = server_ssd_cache_adapter_identity({ adapter });
    if (!(adapter_identity == server_ssd_cache_adapter_identity({ adapter }))) return 1;
    adapter.scale = 0.5f;
    if (adapter_identity == server_ssd_cache_adapter_identity({ adapter })) return 1;

    {
        // Corrupt entries are ignored and cleaned during startup recovery.
        std::ofstream(directory / "broken.snapshot", std::ios::binary) << "not a snapshot";
        const auto config = cache_config(directory);
        server_ssd_prompt_cache first(config, compatibility);
        if (!first.enabled() || std::filesystem::exists(directory / "broken.snapshot")) return 1;

        // The cache directory has a single live owner.
        server_ssd_prompt_cache second(config, compatibility);
        if (second.enabled()) return 1;
    }
    {
        const auto config = cache_config(directory);
        server_ssd_prompt_cache after_release(config, compatibility);
        if (!after_release.enabled()) return 1;
    }

    // Recovery accepts a complete published entry, cleans a temporary one,
    // deduplicates equal prefixes, and retains a child after its payload-only
    // parent is removed.  The only cache interaction is through lookup.
    const auto recovery = directory / "recovery";
    std::filesystem::create_directories(recovery);
    const llama_tokens parent_tokens = { 1, 2 };
    const llama_tokens child_tokens = { 1, 2, 3, 4 };
    write_snapshot(recovery, 1, 0, parent_tokens, compatibility);
    write_snapshot(recovery, 2, 0, parent_tokens, compatibility); // duplicate payload
    write_snapshot(recovery, 3, 1, child_tokens, compatibility);
    std::ofstream(recovery / "interrupted.snapshot.tmp", std::ios::binary) << "partial";
    {
        server_ssd_prompt_cache cache(cache_config(recovery), compatibility);
        if (!cache.enabled() || cache.stats().entries != 2) return 1;
        if (std::filesystem::exists(recovery / "1.snapshot") == std::filesystem::exists(recovery / "2.snapshot") ||
            std::filesystem::exists(recovery / "interrupted.snapshot.tmp")) return 1;
        const auto child = cache.lookup(child_tokens, false);
        if (!child || child.id != 3 || child.n_tokens != child_tokens.size()) return 1;
        server_tokens missing_chunk(llama_tokens { LLAMA_TOKEN_NULL }, false);
        if (cache.lookup(missing_chunk)) return 1;
        mtmd::input_chunks chunks(mtmd_test_create_input_chunks());
        server_tokens missing_identity(chunks, true);
        if (cache.lookup(missing_identity)) return 1;
    }
    std::filesystem::remove(std::filesystem::exists(recovery / "1.snapshot") ? recovery / "1.snapshot" : recovery / "2.snapshot");
    {
        server_ssd_prompt_cache cache(cache_config(recovery), compatibility);
        const auto child = cache.lookup(child_tokens, false);
        if (!child || child.id != 3) return 1;
    }

    // Stored files with a mismatched model, configuration, or adapter stack
    // are cache misses and are removed during startup scanning.
    const auto incompatible = directory / "incompatible";
    std::filesystem::create_directories(incompatible);
    auto wrong_model = compatibility;
    wrong_model.model.bytes[0] = 1;
    auto wrong_config = compatibility;
    wrong_config.n_ctx = 1;
    auto wrong_adapter = compatibility;
    wrong_adapter.adapters.bytes[0] = 1;
    auto wrong_draft = compatibility;
    wrong_draft.draft_model.bytes[0] = 1;
    auto wrong_spec_state = compatibility;
    wrong_spec_state.speculative_state_version = 1;
    auto wrong_topology = compatibility;
    wrong_topology.speculative_kv_topology = 2;
    auto wrong_kv_type = compatibility;
    wrong_kv_type.kv_type = 1;
    write_snapshot(incompatible, 1, 0, parent_tokens, wrong_model);
    write_snapshot(incompatible, 2, 0, child_tokens, wrong_config);
    write_snapshot(incompatible, 3, 0, llama_tokens { 5 }, wrong_adapter);
    write_snapshot(incompatible, 4, 0, llama_tokens { 6 }, wrong_draft);
    write_snapshot(incompatible, 5, 0, llama_tokens { 7 }, wrong_spec_state);
    write_snapshot(incompatible, 6, 0, llama_tokens { 8 }, wrong_topology);
    write_snapshot(incompatible, 7, 0, llama_tokens { 9 }, wrong_kv_type);
    auto wrong_projector = compatibility;
    wrong_projector.projector.bytes[0] = 1;
    auto wrong_image_limits = compatibility;
    wrong_image_limits.image_max_tokens = 256;
    write_snapshot(incompatible, 8, 0, llama_tokens { 10 }, wrong_projector);
    write_snapshot(incompatible, 9, 0, llama_tokens { 11 }, wrong_image_limits);
    {
        server_ssd_prompt_cache cache(cache_config(incompatible), compatibility);
        if (cache.stats().entries != 0) return 1;
        for (uint64_t id = 1; id <= 9; ++id) {
            if (std::filesystem::exists(incompatible / (std::to_string(id) + ".snapshot"))) return 1;
        }
    }

    // MTP snapshots account for all three payloads and maintain separate-KV
    // and target-shared profiles as incompatible cache populations.  Shared
    // KV stores target plus implementation state only.
    auto separate_mtp = compatibility;
    separate_mtp.speculative_state_version = 1;
    separate_mtp.speculative_kv_topology = 1;
    separate_mtp.draft_model.bytes[0] = 1;
    const auto separate = directory / "separate-mtp";
    std::filesystem::create_directories(separate);
    write_snapshot(separate, 1, 0, parent_tokens, separate_mtp, 16, 8, 12);
    {
        const uint64_t expected_bytes = sizeof(test_snapshot_header) + 16 + 8 + 12 +
            sizeof(test_lineage_header) + sizeof(test_lineage_record);
        server_ssd_prompt_cache cache(cache_config(separate, expected_bytes), separate_mtp);
        if (cache.stats().entries != 1 || cache.stats().bytes != expected_bytes || !cache.lookup(parent_tokens, false)) return 1;
    }

    auto shared_mtp = separate_mtp;
    shared_mtp.speculative_kv_topology = 2;
    const auto shared = directory / "shared-mtp";
    std::filesystem::create_directories(shared);
    write_snapshot(shared, 1, 0, parent_tokens, shared_mtp, 16, 0, 12);
    {
        const uint64_t expected_bytes = sizeof(test_snapshot_header) + 16 + 12 +
            sizeof(test_lineage_header) + sizeof(test_lineage_record);
        server_ssd_prompt_cache cache(cache_config(shared, expected_bytes), shared_mtp);
        if (cache.stats().entries != 1 || cache.stats().bytes != expected_bytes || !cache.lookup(parent_tokens, false)) return 1;
    }
    {
        server_ssd_prompt_cache cache(cache_config(shared, sizeof(test_snapshot_header) + 16 + 12 + 1024), separate_mtp);
        if (cache.stats().entries != 0 || std::filesystem::exists(shared / "1.snapshot")) return 1;
    }

    // Metadata and payloads count together.  Startup recovery removes a
    // committed file that would exceed the configured total capacity.
    const auto capacity = directory / "capacity";
    std::filesystem::create_directories(capacity);
    write_snapshot(capacity, 1, 0, parent_tokens, compatibility);
    write_snapshot(capacity, 2, 0, child_tokens, compatibility);
    const uint64_t one_entry_limit = sizeof(test_snapshot_header) + 16 + 24 + 36;
    {
        server_ssd_prompt_cache cache(cache_config(capacity, one_entry_limit), compatibility);
        if (cache.stats().entries != 1 || cache.stats().bytes > one_entry_limit) return 1;
    }
    if (std::filesystem::exists(capacity / "1.snapshot") == std::filesystem::exists(capacity / "2.snapshot")) return 1;

    // Exercise admission policy without a llama_context: the dedicated
    // module owns deduplication, parent links, byte accounting, and the
    // observed-use priority that chooses an eviction victim.
    const auto policy = directory / "policy";
    std::filesystem::create_directories(policy);
    const llama_tokens policy_parent = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    auto policy_child = policy_parent;
    policy_child.insert(policy_child.end(), { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 });
    llama_tokens policy_candidate(100, 7);
    {
        server_ssd_prompt_cache cache(cache_config(policy, 800), compatibility);
        if (!cache.test_admit(policy_parent, 100) || cache.test_admit(policy_parent, 100)) return 1;
        if (!cache.test_admit(policy_child, 100) || cache.test_parent_id(policy_child) == 0) return 1;
        for (int hit = 0; hit < 100; ++hit) {
            if (!cache.test_record_restore(policy_child)) return 1;
        }
        if (!cache.test_admit(policy_candidate, 100)) return 1;
        if (cache.lookup(policy_parent, false) || !cache.lookup(policy_child, false) || !cache.lookup(policy_candidate, false)) return 1;
        if (cache.stats().entries != 2 || cache.stats().evictions != 1 || cache.stats().bytes > 800) return 1;
    }

    std::filesystem::remove_all(directory, ignored);
    return 0;
}

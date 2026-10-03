#include "server-ssd-cache.h"

#include "log.h"
#include "server-common.h"
#include "speculative.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr uint32_t SSD_CACHE_FORMAT_VERSION = 3;
constexpr char SSD_CACHE_MAGIC[8] = { 'L', 'S', 'S', 'D', 'C', '0', '3', '\0' };
constexpr char SSD_LINEAGE_MAGIC[8] = { 'L', 'S', 'S', 'D', 'L', '0', '3', '\0' };

// Small private SHA-256 implementation.  Keeping it here avoids making a
// server-only cache format depend on an optional crypto library.
struct sha256 {
    uint32_t state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    uint64_t bits = 0;
    uint8_t buffer[64] = {};
    size_t used = 0;

    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t block[64]) {
        static const uint32_t k[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,
            0xc67178f2U,
        };
        uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(block[4*i]) << 24) | (uint32_t(block[4*i + 1]) << 16) | (uint32_t(block[4*i + 2]) << 8) | block[4*i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const void * data, size_t size) {
        const auto * p = static_cast<const uint8_t *>(data);
        bits += uint64_t(size) * 8;
        while (size) {
            const size_t n = std::min(size, sizeof(buffer) - used);
            memcpy(buffer + used, p, n);
            used += n; p += n; size -= n;
            if (used == sizeof(buffer)) { transform(buffer); used = 0; }
        }
    }

    std::array<uint8_t, 32> final() const {
        sha256 copy = *this;
        copy.buffer[copy.used++] = 0x80;
        if (copy.used > 56) {
            while (copy.used < 64) copy.buffer[copy.used++] = 0;
            copy.transform(copy.buffer); copy.used = 0;
        }
        while (copy.used < 56) copy.buffer[copy.used++] = 0;
        for (int i = 7; i >= 0; --i) copy.buffer[copy.used++] = uint8_t(copy.bits >> (i * 8));
        copy.transform(copy.buffer);
        std::array<uint8_t, 32> out = {};
        for (int i = 0; i < 8; ++i) {
            out[4*i] = uint8_t(copy.state[i] >> 24); out[4*i + 1] = uint8_t(copy.state[i] >> 16);
            out[4*i + 2] = uint8_t(copy.state[i] >> 8); out[4*i + 3] = uint8_t(copy.state[i]);
        }
        return out;
    }
};

template<typename T> void hash_le(sha256 & hash, T value) {
    static_assert(std::is_integral<T>::value, "integer required");
    uint8_t bytes[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i) bytes[i] = uint8_t(uint64_t(value) >> (8*i));
    hash.update(bytes, sizeof(bytes));
}

server_ssd_cache_digest short_digest(const sha256 & hash) {
    server_ssd_cache_digest result;
    const auto full = hash.final();
    std::copy_n(full.begin(), result.bytes.size(), result.bytes.begin());
    return result;
}

std::string digest_key(const server_ssd_cache_digest & digest) {
    return std::string(reinterpret_cast<const char *>(digest.bytes.data()), digest.bytes.size());
}

std::string prefix_key(uint32_t n_tokens, const server_ssd_cache_digest & digest) {
    return std::to_string(n_tokens) + ":" + digest_key(digest);
}

bool compatible(const server_ssd_cache_compatibility & a, const server_ssd_cache_compatibility & b) {
    return a.model == b.model && a.adapters == b.adapters && a.draft_model == b.draft_model &&
           a.projector == b.projector && a.image_min_tokens == b.image_min_tokens &&
           a.image_max_tokens == b.image_max_tokens && a.projector_settings == b.projector_settings &&
           a.state_layout_version == b.state_layout_version && a.n_ctx == b.n_ctx &&
           a.n_batch == b.n_batch && a.kv_type == b.kv_type &&
           a.speculative_state_version == b.speculative_state_version &&
           a.speculative_kv_topology == b.speculative_kv_topology;
}

#pragma pack(push, 1)
struct ssd_disk_header {
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

struct ssd_lineage_header {
    char magic[8];
    uint32_t format_version;
    uint32_t record_size;
    uint64_t n_records;
};

struct ssd_lineage_record {
    uint64_t id;
    uint64_t parent_id;
    uint32_t n_tokens;
    uint8_t prefix_digest[16];
};
#pragma pack(pop)

static_assert(sizeof(ssd_disk_header) == 208, "SSD cache header must stay fixed-size");
static_assert(sizeof(ssd_lineage_record) == 36, "SSD cache lineage records must stay fixed-size");

server_ssd_cache_compatibility compatibility_from_header(const ssd_disk_header & header) {
    server_ssd_cache_compatibility result;
    memcpy(result.model.bytes.data(), header.model, 16);
    memcpy(result.adapters.bytes.data(), header.adapters, 16);
    memcpy(result.draft_model.bytes.data(), header.draft_model, 16);
    memcpy(result.projector.bytes.data(), header.projector, 16);
    result.image_min_tokens = header.image_min_tokens;
    result.image_max_tokens = header.image_max_tokens;
    result.projector_settings = header.projector_settings;
    result.state_layout_version = header.state_layout_version;
    result.speculative_state_version = header.speculative_state_version;
    result.speculative_kv_topology = header.speculative_kv_topology;
    result.n_ctx = header.n_ctx; result.n_batch = header.n_batch; result.kv_type = header.kv_type;
    return result;
}

std::string entry_filename(uint64_t id) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << id << ".snapshot";
    return out.str();
}

} // namespace

struct server_ssd_prefix_hasher::impl {
    sha256 hash;
    uint32_t n_tokens = 0;

    impl() {
        static constexpr char domain[] = "llama.cpp/ssd-prompt-prefix/v1";
        hash.update(domain, sizeof(domain) - 1);
        hash_le(hash, SSD_CACHE_FORMAT_VERSION);
    }
};

server_ssd_prefix_hasher::server_ssd_prefix_hasher() : impl_(new impl()) {}
server_ssd_prefix_hasher::~server_ssd_prefix_hasher() = default;

void server_ssd_prefix_hasher::update(llama_token token) {
    hash_le(impl_->hash, uint8_t(0));
    hash_le(impl_->hash, uint32_t(token));
    ++impl_->n_tokens;
}

void server_ssd_prefix_hasher::update_image(const std::string & id, uint32_t n_tokens, uint32_t n_pos) {
    hash_le(impl_->hash, uint8_t(1));
    hash_le(impl_->hash, uint32_t(id.size()));
    impl_->hash.update(id.data(), id.size());
    hash_le(impl_->hash, n_tokens);
    hash_le(impl_->hash, n_pos);
    impl_->n_tokens += n_tokens;
}

server_ssd_cache_digest server_ssd_prefix_hasher::finish() const {
    sha256 hash = impl_->hash;
    hash_le(hash, impl_->n_tokens);
    return short_digest(hash);
}

server_ssd_cache_digest server_ssd_prefix_digest(const llama_tokens & tokens) {
    server_ssd_prefix_hasher hash;
    for (const auto token : tokens) hash.update(token);
    return hash.finish();
}

using ssd_prefix = std::pair<uint32_t, server_ssd_cache_digest>;

static std::vector<ssd_prefix> ssd_prefixes(const server_tokens & tokens) {
    std::vector<ssd_prefix> result;
    server_ssd_prefix_hasher hash;
    for (size_t i = 0; i < tokens.size();) {
        if (tokens[i] != LLAMA_TOKEN_NULL) {
            hash.update(tokens[i]);
            result.emplace_back(uint32_t(++i), hash.finish());
            continue;
        }
        const mtmd_input_chunk * chunk;
        try { chunk = tokens.find_chunk(i).get(); } catch (const std::runtime_error &) { return {}; }
        if (!chunk || mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) return {};
        const char * id = mtmd_input_chunk_get_id(chunk);
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
        const llama_pos positions = mtmd_input_chunk_get_n_pos(chunk);
        if (!id || strlen(id) != 64 || !std::all_of(id, id + 64, [](unsigned char c) { return std::isxdigit(c); }) ||
                count == 0 || count > tokens.size() - i || positions <= 0) return {};
        for (size_t j = 0; j < count; ++j) if (tokens[i + j] != LLAMA_TOKEN_NULL) return {};
        hash.update_image(id, uint32_t(count), uint32_t(positions));
        i += count;
        result.emplace_back(uint32_t(i), hash.finish());
    }
    return result;
}

server_ssd_cache_digest server_ssd_cache_file_identity(const std::string & path) {
    sha256 hash;
    static constexpr char domain[] = "llama.cpp/ssd-prompt-file-identity/v1";
    hash.update(domain, sizeof(domain) - 1);
    try {
        const auto canonical = std::filesystem::weakly_canonical(path);
        const auto text = canonical.string();
        hash.update(text.data(), text.size());
        hash_le(hash, uint64_t(std::filesystem::file_size(canonical)));
        hash_le(hash, uint64_t(std::filesystem::last_write_time(canonical).time_since_epoch().count()));
    } catch (const std::exception &) {
        hash.update(path.data(), path.size());
    }
    return short_digest(hash);
}

server_ssd_cache_digest server_ssd_cache_files_identity(const std::vector<std::string> & paths) {
    sha256 hash;
    static constexpr char domain[] = "llama.cpp/ssd-prompt-files-identity/v1";
    hash.update(domain, sizeof(domain) - 1);
    hash_le(hash, uint32_t(paths.size()));
    for (const auto & path : paths) {
        const auto id = server_ssd_cache_file_identity(path);
        hash.update(id.bytes.data(), id.bytes.size());
    }
    return short_digest(hash);
}

server_ssd_cache_digest server_ssd_cache_adapter_identity(const std::vector<common_adapter_lora_info> & adapters) {
    sha256 hash;
    static constexpr char domain[] = "llama.cpp/ssd-prompt-adapters/v1";
    hash.update(domain, sizeof(domain) - 1);
    hash_le(hash, uint32_t(adapters.size()));
    for (const auto & adapter : adapters) {
        const auto id = server_ssd_cache_file_identity(adapter.path);
        hash.update(id.bytes.data(), id.bytes.size());
        uint32_t scale = 0; static_assert(sizeof(scale) == sizeof(adapter.scale), "unexpected float size");
        memcpy(&scale, &adapter.scale, sizeof(scale)); hash_le(hash, scale);
        hash_le(hash, uint32_t(adapter.prompt_prefix.size()));
        hash.update(adapter.prompt_prefix.data(), adapter.prompt_prefix.size());
    }
    return short_digest(hash);
}

bool server_ssd_cache_should_restore(uint32_t active_tokens, uint32_t ram_tokens, uint32_t ssd_tokens, uint32_t prefix_advantage) {
    const uint32_t volatile_tokens = std::max(active_tokens, ram_tokens);
    return ssd_tokens >= volatile_tokens && ssd_tokens - volatile_tokens >= prefix_advantage;
}

bool server_ssd_cache_should_enter(uint32_t candidate_tokens, uint32_t ancestor_tokens, uint32_t prefix_advantage) {
    return candidate_tokens >= ancestor_tokens && candidate_tokens - ancestor_tokens >= prefix_advantage;
}

struct server_ssd_prompt_cache::impl {
    struct entry {
        uint64_t id = 0;
        uint64_t parent_id = 0;
        uint32_t n_tokens = 0;
        server_ssd_cache_digest digest;
        uint64_t target_size = 0;
        uint64_t draft_size = 0;
        uint64_t speculative_size = 0;
        uint64_t hits = 0;
        double priority = 0.0;
    };
    struct staged_entry {
        entry metadata;
        std::vector<uint8_t> target;
        std::vector<uint8_t> draft;
        std::vector<uint8_t> speculative;
    };
    struct lineage_node {
        uint64_t parent_id = 0;
        uint32_t n_tokens = 0;
        server_ssd_cache_digest digest;
    };

    server_ssd_cache_config config;
    server_ssd_cache_compatibility compatibility;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    std::optional<staged_entry> staged;
    bool worker_stop = false;
    bool writer_busy = false;
    bool is_enabled = false;
    uint64_t next_id = 1;
    uint64_t bytes_used = 0;
    uint64_t lineage_bytes = 0;
    double aging_clock = 0.0;
    std::map<uint64_t, entry> entries;
    std::map<uint64_t, lineage_node> lineage;
    std::map<std::string, uint64_t> by_prefix;
    std::set<std::string> in_flight;
    server_ssd_cache_stats statistics;
#ifndef _WIN32
    int lock_fd = -1;
#endif

    impl(const server_ssd_cache_config & config, const server_ssd_cache_compatibility & compatibility)
        : config(config), compatibility(compatibility) {
        initialize();
        if (is_enabled) worker = std::thread([this] { worker_loop(); });
    }

    ~impl() { stop(); }

    std::filesystem::path path_for(uint64_t id) const { return std::filesystem::path(config.directory) / entry_filename(id); }
    std::filesystem::path lineage_path() const { return std::filesystem::path(config.directory) / ".lineage"; }

    static uint64_t entry_bytes(const entry & item) { return sizeof(ssd_disk_header) + item.target_size + item.draft_size + item.speculative_size; }

    void add_lineage(const entry & item) {
        lineage.emplace(item.id, lineage_node { item.parent_id, item.n_tokens, item.digest });
        lineage_bytes = sizeof(ssd_lineage_header) + lineage.size() * sizeof(ssd_lineage_record);
        next_id = std::max(next_id, item.id + 1);
    }

    bool persist_lineage() {
        try {
            const auto final_path = lineage_path();
            const auto tmp_path = final_path.string() + ".tmp";
            ssd_lineage_header header = {};
            memcpy(header.magic, SSD_LINEAGE_MAGIC, sizeof(header.magic));
            header.format_version = SSD_CACHE_FORMAT_VERSION;
            header.record_size = sizeof(ssd_lineage_record);
            header.n_records = lineage.size();
            std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char *>(&header), sizeof(header));
            for (const auto & node : lineage) {
                ssd_lineage_record record = {};
                record.id = node.first; record.parent_id = node.second.parent_id; record.n_tokens = node.second.n_tokens;
                memcpy(record.prefix_digest, node.second.digest.bytes.data(), sizeof(record.prefix_digest));
                output.write(reinterpret_cast<const char *>(&record), sizeof(record));
            }
            output.flush();
            if (!output) return false;
#ifndef _WIN32
            chmod(tmp_path.c_str(), S_IRUSR | S_IWUSR);
#endif
            std::filesystem::rename(tmp_path, final_path);
            lineage_bytes = sizeof(header) + lineage.size() * sizeof(ssd_lineage_record);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    }

    void load_lineage() {
        const auto path = lineage_path();
        if (!std::filesystem::exists(path)) return;
        std::ifstream input(path, std::ios::binary);
        ssd_lineage_header header = {};
        input.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (!input || memcmp(header.magic, SSD_LINEAGE_MAGIC, sizeof(header.magic)) != 0 ||
            header.format_version != SSD_CACHE_FORMAT_VERSION || header.record_size != sizeof(ssd_lineage_record) ||
            std::filesystem::file_size(path) != sizeof(header) + header.n_records * sizeof(ssd_lineage_record)) {
            std::error_code ignored; std::filesystem::remove(path, ignored);
            return;
        }
        for (uint64_t i = 0; i < header.n_records; ++i) {
            ssd_lineage_record record = {};
            input.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!input || record.id == 0 || lineage.count(record.id)) {
                lineage.clear();
                std::error_code ignored; std::filesystem::remove(path, ignored);
                return;
            }
            lineage_node node;
            node.parent_id = record.parent_id; node.n_tokens = record.n_tokens;
            memcpy(node.digest.bytes.data(), record.prefix_digest, sizeof(record.prefix_digest));
            lineage.emplace(record.id, node);
            next_id = std::max(next_id, record.id + 1);
        }
        lineage_bytes = sizeof(header) + lineage.size() * sizeof(ssd_lineage_record);
    }

    void initialize() {
        if (config.directory.empty() || config.max_bytes == 0 || config.prefix_advantage <= 0) return;
        try {
            std::filesystem::create_directories(config.directory);
#ifndef _WIN32
            chmod(config.directory.c_str(), S_IRWXU);
            const auto lock_path = (std::filesystem::path(config.directory) / ".llama-ssd-cache.lock").string();
            lock_fd = open(lock_path.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
            if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
                SRV_WRN("SSD prompt cache disabled: cache directory '%s' is already owned\n", config.directory.c_str());
                if (lock_fd >= 0) { close(lock_fd); lock_fd = -1; }
                return;
            }
#endif
            is_enabled = true;
            load_lineage();
            for (const auto & file : std::filesystem::directory_iterator(config.directory)) {
                if (!file.is_regular_file()) continue;
                const auto extension = file.path().extension().string();
                if (extension == ".tmp") { std::filesystem::remove(file.path()); continue; }
                if (extension != ".snapshot") continue;
                load_metadata(file.path());
            }
            // Old cache directories have no lineage file.  Rebuild it from
            // committed headers before allowing future eviction.
            if (!persist_lineage()) {
                SRV_WRN("%s", "SSD prompt cache could not persist ancestry metadata\n");
            }
            statistics.bytes = bytes_used + lineage_bytes;
            statistics.entries = entries.size();
            SRV_INF("SSD prompt cache enabled: %s, %.2f MiB used by %zu entries\n", config.directory.c_str(), bytes_used / (1024.0 * 1024.0), entries.size());
        } catch (const std::exception & error) {
            SRV_WRN("SSD prompt cache disabled: %s\n", error.what());
            is_enabled = false;
        }
    }

    void load_metadata(const std::filesystem::path & path) {
        std::ifstream input(path, std::ios::binary);
        ssd_disk_header header = {};
        input.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (header.target_size > config.max_bytes || header.draft_size > config.max_bytes ||
                header.speculative_size > config.max_bytes ||
                header.target_size > config.max_bytes - header.draft_size ||
                header.target_size + header.draft_size > config.max_bytes - header.speculative_size) {
            std::filesystem::remove(path);
            return;
        }
        const uint64_t expected = sizeof(header) + header.target_size + header.draft_size + header.speculative_size;
        if (!input || memcmp(header.magic, SSD_CACHE_MAGIC, sizeof(header.magic)) != 0 ||
            header.format_version != SSD_CACHE_FORMAT_VERSION || header.header_size != sizeof(header) ||
            !compatible(compatibility_from_header(header), compatibility) ||
            std::filesystem::file_size(path) != expected || expected > config.max_bytes) {
            std::filesystem::remove(path);
            return;
        }
        entry item;
        item.id = header.id; item.parent_id = header.parent_id; item.n_tokens = header.n_tokens;
        memcpy(item.digest.bytes.data(), header.prefix_digest, item.digest.bytes.size());
        item.target_size = header.target_size; item.draft_size = header.draft_size; item.speculative_size = header.speculative_size;
        const auto key = prefix_key(item.n_tokens, item.digest);
        // Recovery must enforce the same one-payload-per-prefix and capacity
        // rules as live admission.  Leaving rejected files behind would make
        // the configured directory grow past its limit and reintroduce them
        // on every subsequent startup.
        const uint64_t lineage_extra = lineage.count(item.id) ? 0 : sizeof(ssd_lineage_record);
        if (by_prefix.count(key) || bytes_used + lineage_bytes + lineage_extra + entry_bytes(item) > config.max_bytes) {
            std::filesystem::remove(path);
            return;
        }
        add_lineage(item);
        bytes_used += entry_bytes(item); by_prefix.emplace(key, item.id); entries.emplace(item.id, item);
    }

    bool reserve(const entry & candidate) {
        const uint64_t wanted = entry_bytes(candidate);
        const uint64_t required = wanted + sizeof(ssd_lineage_record);
        if (required > config.max_bytes) return false;
        const double candidate_priority = candidate.priority;
        uint64_t projected_bytes = bytes_used;
        std::vector<uint64_t> victims;
        std::vector<const entry *> ordered;
        ordered.reserve(entries.size());
        for (const auto & item : entries) ordered.push_back(&item.second);
        std::sort(ordered.begin(), ordered.end(), [](const entry * a, const entry * b) { return a->priority < b->priority; });
        for (const entry * item : ordered) {
            if (projected_bytes + lineage_bytes + required <= config.max_bytes) break;
            if (item->priority > candidate_priority) return false;
            projected_bytes -= entry_bytes(*item);
            victims.push_back(item->id);
        }
        if (projected_bytes + lineage_bytes + required > config.max_bytes) return false;
        for (const uint64_t id : victims) {
            const auto victim = entries.find(id);
            GGML_ASSERT(victim != entries.end());
            aging_clock = std::max(aging_clock, victim->second.priority);
            std::error_code ignored;
            std::filesystem::remove(path_for(victim->first), ignored);
            bytes_used -= entry_bytes(victim->second);
            by_prefix.erase(prefix_key(victim->second.n_tokens, victim->second.digest));
            entries.erase(victim);
            recompute_priorities();
            ++statistics.evictions;
        }
        return bytes_used + lineage_bytes + required <= config.max_bytes;
    }

    uint32_t retained_ancestor_tokens(const entry & item) const {
        uint64_t id = item.parent_id;
        while (id) {
            const auto retained = entries.find(id);
            if (retained != entries.end()) return retained->second.n_tokens;
            const auto node = lineage.find(id);
            if (node == lineage.end()) break;
            id = node->second.parent_id;
        }
        return 0;
    }

    void recompute_priorities() {
        for (auto & item : entries) {
            const uint32_t fallback = retained_ancestor_tokens(item.second);
            item.second.priority = aging_clock + double((item.second.n_tokens - fallback) * (item.second.hits + 1)) /
                std::max<uint64_t>(1, entry_bytes(item.second));
        }
    }

    uint64_t nearest_parent(const std::vector<ssd_prefix> & prefixes) const {
        uint64_t parent = 0;
        for (const auto & prefix : prefixes) {
            const auto key = prefix_key(prefix.first, prefix.second);
            const auto found = by_prefix.find(key);
            if (found != by_prefix.end()) parent = found->second;
        }
        return parent;
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            cv.wait(lock, [this] { return worker_stop || staged.has_value(); });
            if (worker_stop && !staged) break;
            staged_entry item = std::move(*staged);
            staged.reset();
            lock.unlock();
            const bool committed = write(item);
            lock.lock();
            writer_busy = false;
            in_flight.erase(prefix_key(item.metadata.n_tokens, item.metadata.digest));
            if (committed) {
                bytes_used += entry_bytes(item.metadata);
                by_prefix.emplace(prefix_key(item.metadata.n_tokens, item.metadata.digest), item.metadata.id);
                entries.emplace(item.metadata.id, item.metadata);
                add_lineage(item.metadata);
                if (!persist_lineage()) {
                    SRV_WRN("%s", "SSD prompt cache could not update ancestry metadata\n");
                }
                ++statistics.admissions;
            } else {
                ++statistics.write_failures;
            }
            statistics.bytes = bytes_used + lineage_bytes; statistics.entries = entries.size();
            cv.notify_all();
        }
    }

    bool write(const staged_entry & item) const {
        const auto final_path = path_for(item.metadata.id);
        const auto tmp_path = final_path.string() + ".tmp";
        try {
            sha256 payload_hash;
            payload_hash.update(item.target.data(), item.target.size());
            payload_hash.update(item.draft.data(), item.draft.size());
            payload_hash.update(item.speculative.data(), item.speculative.size());
            const auto payload_sum = payload_hash.final();
            ssd_disk_header header = {};
            memcpy(header.magic, SSD_CACHE_MAGIC, sizeof(header.magic));
            header.format_version = SSD_CACHE_FORMAT_VERSION; header.header_size = sizeof(header);
            header.id = item.metadata.id; header.parent_id = item.metadata.parent_id;
            memcpy(header.model, compatibility.model.bytes.data(), 16); memcpy(header.adapters, compatibility.adapters.bytes.data(), 16);
            memcpy(header.draft_model, compatibility.draft_model.bytes.data(), 16);
            memcpy(header.projector, compatibility.projector.bytes.data(), 16);
            header.image_min_tokens = compatibility.image_min_tokens;
            header.image_max_tokens = compatibility.image_max_tokens;
            header.projector_settings = compatibility.projector_settings;
            header.state_layout_version = compatibility.state_layout_version; header.n_ctx = compatibility.n_ctx;
            header.n_batch = compatibility.n_batch; header.kv_type = compatibility.kv_type;
            header.speculative_state_version = compatibility.speculative_state_version;
            header.speculative_kv_topology = compatibility.speculative_kv_topology;
            header.n_tokens = item.metadata.n_tokens; memcpy(header.prefix_digest, item.metadata.digest.bytes.data(), 16);
            header.target_size = item.target.size(); header.draft_size = item.draft.size(); header.speculative_size = item.speculative.size();
            memcpy(header.payload_sha256, payload_sum.data(), payload_sum.size());
#ifndef _WIN32
            const uint64_t total = sizeof(header) + item.target.size() + item.draft.size() + item.speculative.size();
            if (total > uint64_t(std::numeric_limits<off_t>::max())) throw std::runtime_error("cache snapshot is too large");
            const int fd = open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
            if (fd < 0) throw std::runtime_error("failed to create temporary cache snapshot");
            if (ftruncate(fd, off_t(total)) != 0) { close(fd); throw std::runtime_error("failed to size temporary cache snapshot"); }
            void * mapping = mmap(nullptr, size_t(total), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapping == MAP_FAILED) { close(fd); throw std::runtime_error("failed to map temporary cache snapshot"); }
            auto * bytes = static_cast<uint8_t *>(mapping);
            memcpy(bytes, &header, sizeof(header));
            memcpy(bytes + sizeof(header), item.target.data(), item.target.size());
            memcpy(bytes + sizeof(header) + item.target.size(), item.draft.data(), item.draft.size());
            memcpy(bytes + sizeof(header) + item.target.size() + item.draft.size(), item.speculative.data(), item.speculative.size());
            const int sync_result = msync(mapping, size_t(total), MS_SYNC);
            munmap(mapping, size_t(total));
            const int flush_result = fsync(fd);
            close(fd);
            if (sync_result != 0 || flush_result != 0) throw std::runtime_error("failed to synchronize temporary cache snapshot");
#else
            std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char *>(&header), sizeof(header));
            output.write(reinterpret_cast<const char *>(item.target.data()), item.target.size());
            output.write(reinterpret_cast<const char *>(item.draft.data()), item.draft.size());
            output.write(reinterpret_cast<const char *>(item.speculative.data()), item.speculative.size());
            output.flush();
            if (!output) throw std::runtime_error("failed to write cache snapshot");
#endif
            std::filesystem::rename(tmp_path, final_path);
            return true;
        } catch (const std::exception & error) {
            SRV_WRN("SSD prompt cache write failed: %s\n", error.what());
            std::error_code ignored; std::filesystem::remove(tmp_path, ignored);
            return false;
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (worker_stop) return;
            worker_stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
#ifndef _WIN32
        if (lock_fd >= 0) {
            flock(lock_fd, LOCK_UN);
            close(lock_fd);
            lock_fd = -1;
        }
#endif
        is_enabled = false;
    }
};

server_ssd_prompt_cache::server_ssd_prompt_cache(const server_ssd_cache_config & config, const server_ssd_cache_compatibility & compatibility)
    : impl_(new impl(config, compatibility)) {}
server_ssd_prompt_cache::~server_ssd_prompt_cache() = default;
bool server_ssd_prompt_cache::enabled() const { return impl_->is_enabled; }

server_ssd_cache_handle server_ssd_prompt_cache::lookup(const llama_tokens & tokens, bool require_strict_prefix) {
    server_tokens input(tokens, false);
    return lookup(input, require_strict_prefix);
}

server_ssd_cache_handle server_ssd_prompt_cache::lookup(const server_tokens & tokens, bool require_strict_prefix) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->is_enabled) return {};
    server_ssd_cache_handle best;
    const auto prefixes = ssd_prefixes(tokens);
    for (const auto & prefix : prefixes) {
        if (require_strict_prefix && prefix.first >= tokens.size()) break;
        const auto found = impl_->by_prefix.find(prefix_key(prefix.first, prefix.second));
        if (found != impl_->by_prefix.end()) best = { found->second, prefix.first, prefix.second };
    }
    if (best) ++impl_->statistics.lookup_hits; else ++impl_->statistics.lookup_misses;
    return best;
}

void server_ssd_prompt_cache::enter(const llama_tokens & tokens, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id) {
    server_tokens input(tokens, false);
    enter(input, target, draft, speculative, sequence_id);
}

void server_ssd_prompt_cache::enter(const server_tokens & tokens, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id) {
    if (tokens.empty() || !target) return;
    const int64_t capture_start = ggml_time_us();
    const auto prefixes = ssd_prefixes(tokens);
    if (prefixes.empty() || prefixes.back().first != tokens.size()) return;
    const auto digest = prefixes.back().second;
    const auto key = prefix_key(uint32_t(tokens.size()), digest);
    impl::entry metadata;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->is_enabled) return;
        if (impl_->writer_busy) { ++impl_->statistics.busy_drops; return; }
        if (impl_->by_prefix.count(key) || impl_->in_flight.count(key)) return;
        metadata.parent_id = impl_->nearest_parent(prefixes);
        const uint32_t previous = metadata.parent_id ? impl_->entries.at(metadata.parent_id).n_tokens : 0;
        if (!server_ssd_cache_should_enter(uint32_t(tokens.size()), previous, impl_->config.prefix_advantage)) return;
        metadata.id = impl_->next_id++; metadata.n_tokens = uint32_t(tokens.size()); metadata.digest = digest;
    }
    const auto capability = common_speculative_get_ssd_state_capability(speculative);
    if (speculative && !capability.supported) return;
    if (capability.kv_topology == COMMON_SPECULATIVE_SSD_KV_SEPARATE && draft == nullptr) return;
    std::vector<uint8_t> speculative_data;
    if (capability.supported && !common_speculative_capture_ssd_state(speculative, sequence_id, speculative_data)) return;
    metadata.target_size = llama_state_seq_get_size_ext(target, sequence_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    metadata.draft_size = capability.kv_topology == COMMON_SPECULATIVE_SSD_KV_SEPARATE ?
        llama_state_seq_get_size_ext(draft, sequence_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    metadata.speculative_size = speculative_data.size();
    if (metadata.target_size == 0) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const uint32_t parent_tokens = metadata.parent_id ? impl_->entries.at(metadata.parent_id).n_tokens : 0;
        metadata.priority = impl_->aging_clock + double(metadata.n_tokens - parent_tokens) /
            std::max<uint64_t>(1, impl::entry_bytes(metadata));
        if (!impl_->is_enabled || impl_->writer_busy || !impl_->reserve(metadata)) return;
        impl_->writer_busy = true; impl_->in_flight.insert(key);
    }
    impl::staged_entry staged;
    staged.metadata = metadata;
    try {
#if defined(LLAMA_SERVER_SSD_CACHE_TESTING)
        // Deterministic failure injection for the server integration test.
        // This is intentionally unavailable in production builds.
        if (std::getenv("LLAMA_SSD_CACHE_TEST_FAIL_STAGING")) throw std::bad_alloc();
#endif
        staged.target.resize(metadata.target_size); staged.draft.resize(metadata.draft_size); staged.speculative = std::move(speculative_data);
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->writer_busy = false; impl_->in_flight.erase(key); impl_->is_enabled = false;
        SRV_WRN("%s", "SSD prompt cache disabled: staging memory allocation failed\n");
        return;
    }
    const size_t target_written = llama_state_seq_get_data_ext(target, staged.target.data(), staged.target.size(), sequence_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t draft_written = draft ? llama_state_seq_get_data_ext(draft, staged.draft.data(), staged.draft.size(), sequence_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    if (target_written != staged.target.size() || draft_written != staged.draft.size()) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->writer_busy = false; impl_->in_flight.erase(key);
        ++impl_->statistics.write_failures;
        return;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->statistics.capture_us += ggml_time_us() - capture_start;
        impl_->staged.emplace(std::move(staged));
    }
    impl_->cv.notify_one();
}

bool server_ssd_prompt_cache::restore(const server_ssd_cache_handle & handle, llama_context * target, llama_context * draft, common_speculative * speculative, int32_t sequence_id) {
    const int64_t start = ggml_time_us();
    const auto capability = common_speculative_get_ssd_state_capability(speculative);
    impl::entry metadata;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto found = impl_->entries.find(handle.id);
        if (!impl_->is_enabled || found == impl_->entries.end() || found->second.n_tokens != handle.n_tokens || !(found->second.digest == handle.digest)) return false;
        metadata = found->second;
    }
    bool success = false;
    try {
        std::ifstream input(impl_->path_for(metadata.id), std::ios::binary);
        ssd_disk_header header = {};
        input.read(reinterpret_cast<char *>(&header), sizeof(header));
        const uint64_t expected_size = sizeof(header) + metadata.target_size + metadata.draft_size + metadata.speculative_size;
        if (!input || memcmp(header.magic, SSD_CACHE_MAGIC, sizeof(header.magic)) != 0 ||
                header.format_version != SSD_CACHE_FORMAT_VERSION || header.header_size != sizeof(header) ||
                header.id != metadata.id || header.n_tokens != metadata.n_tokens ||
                header.target_size != metadata.target_size || header.draft_size != metadata.draft_size ||
                header.speculative_size != metadata.speculative_size ||
                std::filesystem::file_size(impl_->path_for(metadata.id)) != expected_size) {
            throw std::runtime_error("invalid SSD cache snapshot header");
        }
        std::vector<uint8_t> target_data(header.target_size), draft_data(header.draft_size), speculative_data(header.speculative_size);
        input.read(reinterpret_cast<char *>(target_data.data()), target_data.size());
        input.read(reinterpret_cast<char *>(draft_data.data()), draft_data.size());
        input.read(reinterpret_cast<char *>(speculative_data.data()), speculative_data.size());
        sha256 hash; hash.update(target_data.data(), target_data.size()); hash.update(draft_data.data(), draft_data.size()); hash.update(speculative_data.data(), speculative_data.size());
        const auto checksum = hash.final();
        if (input && memcmp(checksum.data(), header.payload_sha256, checksum.size()) == 0 &&
            llama_state_seq_set_data_ext(target, target_data.data(), target_data.size(), sequence_id, 0) == target_data.size() &&
            (!draft_data.size() || (draft && llama_state_seq_set_data_ext(draft, draft_data.data(), draft_data.size(), sequence_id, 0) == draft_data.size())) &&
            (!speculative_data.size() || (capability.supported && common_speculative_restore_ssd_state(speculative, sequence_id, speculative_data))) ) {
            success = true;
        }
    } catch (const std::exception &) { success = false; }
    if (!success) {
        if (target) llama_memory_seq_rm(llama_get_memory(target), sequence_id, 0, -1);
        if (draft && capability.kv_topology != COMMON_SPECULATIVE_SSD_KV_TARGET_SHARED) {
            llama_memory_seq_rm(llama_get_memory(draft), sequence_id, 0, -1);
        }
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->statistics.restore_us += ggml_time_us() - start;
    if (success) {
        auto & item = impl_->entries.at(metadata.id);
        ++item.hits;
        const uint32_t parent_tokens = impl_->retained_ancestor_tokens(item);
        item.priority = impl_->aging_clock + double((item.n_tokens - parent_tokens) * (item.hits + 1)) / std::max<uint64_t>(1, impl::entry_bytes(item));
    } else {
        std::error_code ignored; std::filesystem::remove(impl_->path_for(metadata.id), ignored);
        impl_->bytes_used -= impl::entry_bytes(metadata); impl_->by_prefix.erase(prefix_key(metadata.n_tokens, metadata.digest)); impl_->entries.erase(metadata.id);
    }
    impl_->statistics.bytes = impl_->bytes_used + impl_->lineage_bytes; impl_->statistics.entries = impl_->entries.size();
    return success;
}

server_ssd_cache_stats server_ssd_prompt_cache::stats() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->statistics; }
void server_ssd_prompt_cache::shutdown() { impl_->stop(); }

#if defined(LLAMA_SERVER_SSD_CACHE_TESTING)
bool server_ssd_prompt_cache::test_admit(const llama_tokens & tokens, uint64_t target_size) {
    if (tokens.empty() || target_size == 0) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->is_enabled) return false;
    const auto digest = server_ssd_prefix_digest(tokens);
    const auto key = prefix_key(uint32_t(tokens.size()), digest);
    if (impl_->by_prefix.count(key)) return false;

    impl::entry metadata;
    metadata.id = impl_->next_id++;
    metadata.parent_id = impl_->nearest_parent(ssd_prefixes(server_tokens(tokens, false)));
    metadata.n_tokens = uint32_t(tokens.size());
    metadata.digest = digest;
    metadata.target_size = target_size;
    const uint32_t parent_tokens = metadata.parent_id ? impl_->entries.at(metadata.parent_id).n_tokens : 0;
    metadata.priority = impl_->aging_clock + double(metadata.n_tokens - parent_tokens) /
        std::max<uint64_t>(1, impl::entry_bytes(metadata));
    if (!impl_->reserve(metadata)) return false;
    impl_->entries.emplace(metadata.id, metadata);
    impl_->by_prefix.emplace(key, metadata.id);
    impl_->add_lineage(metadata);
    impl_->persist_lineage();
    ++impl_->statistics.admissions;
    impl_->statistics.bytes = impl_->bytes_used + impl_->lineage_bytes;
    impl_->statistics.entries = impl_->entries.size();
    impl_->bytes_used += impl::entry_bytes(metadata);
    impl_->statistics.bytes = impl_->bytes_used + impl_->lineage_bytes;
    return true;
}

uint64_t server_ssd_prompt_cache::test_parent_id(const llama_tokens & tokens) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->by_prefix.find(prefix_key(uint32_t(tokens.size()), server_ssd_prefix_digest(tokens)));
    return found == impl_->by_prefix.end() ? 0 : impl_->entries.at(found->second).parent_id;
}

bool server_ssd_prompt_cache::test_record_restore(const llama_tokens & tokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->by_prefix.find(prefix_key(uint32_t(tokens.size()), server_ssd_prefix_digest(tokens)));
    if (found == impl_->by_prefix.end()) return false;
    auto & item = impl_->entries.at(found->second);
    ++item.hits;
    const uint32_t parent_tokens = impl_->retained_ancestor_tokens(item);
    item.priority = impl_->aging_clock + double((item.n_tokens - parent_tokens) * (item.hits + 1)) /
        std::max<uint64_t>(1, impl::entry_bytes(item));
    return true;
}
#endif

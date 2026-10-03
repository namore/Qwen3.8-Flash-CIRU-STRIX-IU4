#include "speculative.h"

#include <cstring>
#include <vector>

int main() {
    const std::vector<float> expected = { 1.0f, -2.5f, 3.25f, 0.0f };
    const llama_pos expected_pos = 123;
    const llama_pos expected_image_pos = 117;
    std::vector<uint8_t> payload;
    if (!common_speculative_mtp_ssd_state_capture(expected_pos, expected_image_pos, expected, payload)) return 1;

    std::vector<float> restored = { 9.0f };
    llama_pos restored_pos = -1;
    llama_pos restored_image_pos = -1;
    if (!common_speculative_mtp_ssd_state_restore(uint32_t(expected.size()), payload, restored_pos, restored_image_pos, restored) ||
            restored_pos != expected_pos || restored_image_pos != expected_image_pos || restored != expected) return 1;

    const std::vector<float> unchanged = restored;
    auto malformed = payload;
    malformed[0] ^= 1;
    if (common_speculative_mtp_ssd_state_restore(uint32_t(expected.size()), malformed, restored_pos, restored_image_pos, restored) || restored != unchanged) return 1;

    malformed = payload;
    malformed.resize(malformed.size() - 1);
    if (common_speculative_mtp_ssd_state_restore(uint32_t(expected.size()), malformed, restored_pos, restored_image_pos, restored) || restored != unchanged) return 1;

    malformed = payload;
    const uint32_t wrong_width = uint32_t(expected.size() + 1);
    memcpy(malformed.data() + 4, &wrong_width, sizeof(wrong_width));
    if (common_speculative_mtp_ssd_state_restore(uint32_t(expected.size()), malformed, restored_pos, restored_image_pos, restored) || restored != unchanged) return 1;

    malformed = payload;
    const uint32_t wrong_magic = 0;
    memcpy(malformed.data(), &wrong_magic, sizeof(wrong_magic));
    if (common_speculative_mtp_ssd_state_restore(uint32_t(expected.size()), malformed, restored_pos, restored_image_pos, restored) || restored != unchanged) return 1;

    return 0;
}

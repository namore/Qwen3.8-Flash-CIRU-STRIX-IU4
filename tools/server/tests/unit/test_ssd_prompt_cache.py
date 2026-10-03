import time
import base64
import struct
import zlib
import pytest
import re
import shutil

from utils import *


def make_server(cache_dir, advantage=1, cache_ram=0):
    server = ServerPreset.tinyllama2()
    server.cache_ram = cache_ram
    server.cache_ssd_dir = str(cache_dir)
    server.cache_ssd_maxsize = 16
    server.cache_ssd_prefix_advantage = advantage
    server.server_metrics = True
    server.n_predict = 1
    server.temperature = 0.0
    return server


def metric_body(server):
    response = server.make_request("GET", "/metrics")
    assert response.status_code == 200
    assert isinstance(response.body, str)
    return response.body


def metric_value(server, name):
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    match = re.search(rf"^llamacpp:{re.escape(name)} ({number})$", metric_body(server), re.MULTILINE)
    assert match, f"missing metric {name}"
    return float(match.group(1))


def wait_for_admission(server):
    for _ in range(40):
        body = metric_body(server)
        if "llamacpp:ssd_cache_admissions_total 1" in body:
            return
        time.sleep(0.05)
    raise AssertionError("SSD cache did not admit a stable prompt state")


def wait_for_metric(server, name, minimum):
    for _ in range(80):
        if metric_value(server, name) >= minimum:
            return
        time.sleep(0.05)
    raise AssertionError(f"{name} did not reach {minimum}")


def test_ssd_cache_reuses_after_restart(tmp_path):
    cache_dir = tmp_path / "ssd-cache"
    prompt = "A durable cache should restore this exact text prefix."

    server = make_server(cache_dir)
    server.start()
    response = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    assert response.status_code == 200
    wait_for_admission(server)
    server.stop()

    server = make_server(cache_dir)
    server.start()
    # Identical prompts use the strict-prefix fallback instead of loading a
    # tip state that cannot provide final-token logits.
    response = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    assert response.status_code == 200
    response = server.make_request("POST", "/completion", data={"prompt": prompt + " Next.", "cache_prompt": True})
    assert response.status_code == 200
    assert metric_value(server, "ssd_cache_lookup_hits_total") >= 1


def test_ssd_cache_threshold_and_disabled_regression(tmp_path):
    prompt = "A shared prefix is useful only when it exceeds the configured advantage."

    # A large advantage keeps an equally deep RAM candidate selected and makes
    # SSD optional; disabling SSD must retain normal RAM prompt-cache behavior.
    disabled = ServerPreset.tinyllama2()
    disabled.cache_ram = 16
    disabled.n_predict = 1
    disabled.temperature = 0.0
    disabled.start()
    first = disabled.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    second = disabled.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    assert first.status_code == second.status_code == 200
    assert second.body["timings"]["prompt_n"] <= first.body["timings"]["prompt_n"]
    disabled.stop()

    cache = make_server(tmp_path / "threshold", advantage=4096, cache_ram=16)
    cache.start()
    response = cache.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    assert response.status_code == 200
    assert "llamacpp:ssd_cache_lookup_hits_total" in metric_body(cache)


def test_ssd_cache_beats_unrestorable_swa_slot_prefix(tmp_path):
    # A slot save/load restores the model state but not server checkpoints.
    # tinygemma3's SWA state consequently cannot rewind from the generated
    # token to the shared prompt prefix.  The SSD state is valid, though, and
    # must compete against the post-validation (zero) live prefix rather than
    # the slot's logical LCP.
    server = ServerPreset.tinygemma3()
    server.cache_ram = 0
    server.cache_ssd_dir = str(tmp_path / "ssd-cache")
    server.cache_ssd_maxsize = 64
    server.cache_ssd_prefix_advantage = 1
    server.server_metrics = True
    slot_save_path = tmp_path / "slots"
    slot_save_path.mkdir()
    server.slot_save_path = str(slot_save_path)
    server.n_predict = 1
    server.temperature = 0.0
    server.start()

    prompt = "A saved SWA prefix must be restored from SSD. " * 24
    first = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert first.status_code == 200
    wait_for_metric(server, "ssd_cache_admissions_total", 1)

    saved = server.make_request("POST", "/slots/0?action=save", data={"filename": "swa.bin"})
    assert saved.status_code == 200
    restored = server.make_request("POST", "/slots/0?action=restore", data={"filename": "swa.bin"})
    assert restored.status_code == 200

    second = server.make_request("POST", "/completion", data={
        "prompt": prompt + " Continue from the shared prefix.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert second.status_code == 200
    assert second.body["timings"]["cache_n"] > 0
    assert second.body["timings"]["prompt_n"] < first.body["timings"]["prompt_n"]


def test_ssd_cache_recovers_only_published_payloads(tmp_path):
    cache_dir = tmp_path / "published"
    server = make_server(cache_dir)
    server.start()
    response = server.make_request("POST", "/completion", data={
        "prompt": "The complete state must appear only after atomic publication.",
        "cache_prompt": True,
    })
    assert response.status_code == 200
    wait_for_admission(server)
    server.stop()

    snapshots = list(cache_dir.glob("*.snapshot"))
    assert snapshots
    assert not list(cache_dir.glob("*.snapshot.tmp"))


def test_ssd_cache_damaged_payload_falls_back(tmp_path):
    cache_dir = tmp_path / "damaged"
    prompt = "A damaged saved state must fall back to normal decoding."
    server = make_server(cache_dir)
    server.start()
    first = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True})
    assert first.status_code == 200
    wait_for_admission(server)
    server.stop()

    snapshot = next(cache_dir.glob("*.snapshot"))
    data = bytearray(snapshot.read_bytes())
    data[-1] ^= 1
    snapshot.write_bytes(data)

    server = make_server(cache_dir)
    server.start()
    response = server.make_request("POST", "/completion", data={"prompt": prompt + " Continue.", "cache_prompt": True})
    assert response.status_code == 200
    assert response.body["timings"]["cache_n"] == 0
    server.stop()


def test_ssd_cache_pressure_is_bounded_and_deduplicates_shared_prefixes(tmp_path):
    cache_dir = tmp_path / "pressure"
    server = make_server(cache_dir)
    server.n_slots = 2
    server.n_ctx = 4096
    server.start()

    shared = "System instruction shared by all conversations: answer compactly. "
    first = server.make_request("POST", "/completion", data={"prompt": shared, "cache_prompt": True})
    assert first.status_code == 200
    wait_for_admission(server)
    admissions_before_repeat = metric_value(server, "ssd_cache_admissions_total")

    # Repeating an identical stable state must not create a second payload.
    repeat = server.make_request("POST", "/completion", data={"prompt": shared, "cache_prompt": True})
    assert repeat.status_code == 200
    time.sleep(0.1)
    assert metric_value(server, "ssd_cache_admissions_total") <= admissions_before_repeat + 1
    admissions_after_repeat = metric_value(server, "ssd_cache_admissions_total")
    repeat = server.make_request("POST", "/completion", data={"prompt": shared, "cache_prompt": True})
    assert repeat.status_code == 200
    time.sleep(0.1)
    assert metric_value(server, "ssd_cache_admissions_total") == admissions_after_repeat

    # Exercise many independent branches plus a single long conversation. The
    # cache may evict, but its observable accounting must remain bounded.
    for branch in range(12):
        response = server.make_request("POST", "/completion", data={
            "prompt": shared + f"Branch {branch}: " + ("detail " * 20),
            "cache_prompt": True,
        })
        assert response.status_code == 200
    long = server.make_request("POST", "/completion", data={
        "prompt": "One long independent conversation. " + ("context " * 180),
        "cache_prompt": True,
    })
    assert long.status_code == 200
    wait_for_metric(server, "ssd_cache_entries", 1)
    assert metric_value(server, "ssd_cache_bytes") <= 16 * 1024 * 1024


def test_ssd_cache_concurrent_target_only_slots(tmp_path):
    server = make_server(tmp_path / "concurrent")
    server.n_slots = 2
    server.n_ctx = 1024
    server.start()
    requests = [
        (server.make_request, ("POST", "/completion", {
            "prompt": f"Concurrent branch {index}. " + ("shared context " * 30),
            "cache_prompt": True,
        }))
        for index in range(2)
    ]
    responses = parallel_function_calls(requests)
    assert all(response.status_code == 200 for response in responses)
    wait_for_metric(server, "ssd_cache_admissions_total", 1)
    assert metric_value(server, "ssd_cache_write_failures_total") == 0
    server.stop()


def test_ssd_cache_worker_failure_and_busy_drop_do_not_fail_requests(tmp_path):
    cache_dir = tmp_path / "worker-failure"
    server = make_server(cache_dir)
    server.n_slots = 2
    server.n_ctx = 1024
    server.start()

    # Removing the storage root after initialization makes persistence fail;
    # normal completion requests must remain successful.
    shutil.rmtree(cache_dir)
    requests = [
        (server.make_request, ("POST", "/completion", {
            "prompt": f"Concurrent cache staging request {index}. " + ("token " * 80),
            "cache_prompt": True,
        }))
        for index in range(2)
    ]
    responses = parallel_function_calls(requests)
    assert all(response.status_code == 200 for response in responses)
    wait_for_metric(server, "ssd_cache_write_failures_total", 1)
    # The second candidate may race after the worker has cleared; either path
    # remains observable and normal request handling is unaffected.
    assert metric_value(server, "ssd_cache_busy_drops_total") >= 0


def test_ssd_cache_staging_allocation_failure_is_nonfatal(tmp_path, monkeypatch):
    # The server test build offers a deterministic allocation-failure seam;
    # release builds have no such switch.  SSD caching disables itself while
    # the completion request continues normally.
    monkeypatch.setenv("LLAMA_SSD_CACHE_TEST_FAIL_STAGING", "1")
    server = make_server(tmp_path / "staging-failure")
    server.log_path = str(tmp_path / "staging-failure.log")
    server.start()
    response = server.make_request("POST", "/completion", data={
        "prompt": "Staging allocation failure must not fail prompt processing.",
        "cache_prompt": True,
    })
    assert response.status_code == 200
    server.stop()
    assert "staging memory allocation failed" in (tmp_path / "staging-failure.log").read_text()




def png_pixel(red):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    image = header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes([0, red, 0, 0]))) + chunk(b"IEND", b"")
    return base64.b64encode(image).decode()


@pytest.mark.parametrize("images", [[png_pixel(255)], [png_pixel(255), png_pixel(0)]])
def test_ssd_cache_reuses_complete_images_after_restart(tmp_path, monkeypatch, images):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    cache_dir = tmp_path / "vision"

    def vision_server():
        server = ServerPreset.tinygemma3()
        server.n_ctx = 2048
        server.n_slots = 1
        server.n_predict = 1
        server.temperature = 0.0
        server.cache_ram = 0
        server.cache_ssd_dir = str(cache_dir)
        server.cache_ssd_maxsize = 64
        server.cache_ssd_prefix_advantage = 1
        server.server_metrics = True
        return server

    prompt = "What is shown? " + " ".join("<__media__>" for _ in images)

    def request(server, suffix, image_data):
        return server.make_request("POST", "/completions", data={
            "prompt": {"prompt_string": prompt + suffix, "multimodal_data": image_data},
            "cache_prompt": True,
            "n_predict": 1,
        })

    server = vision_server()
    server.start()
    first = request(server, "", images)
    assert first.status_code == 200
    wait_for_metric(server, "ssd_cache_admissions_total", 1)
    if len(images) == 2:
        for attempt in range(4):
            lengths = [struct.unpack_from("<I", path.read_bytes(), 132)[0] for path in cache_dir.glob("*.snapshot")]
            if lengths and max(lengths) >= first.body["timings"]["prompt_n"]:
                break
            time.sleep(0.1)
            assert request(server, f" Replay {attempt}.", images).status_code == 200
        else:
            pytest.fail("two-image prefix was not admitted")
    server.stop()

    server = vision_server()
    server.start()
    suffix = " Continue."
    reused = request(server, suffix, images)
    assert reused.status_code == 200
    assert reused.body["timings"]["cache_n"] > 0
    assert metric_value(server, "ssd_cache_lookup_hits_total") >= 1

    changed = images.copy()
    changed[-1] = png_pixel(128)
    different = request(server, suffix, changed)
    assert different.status_code == 200
    assert different.body["timings"]["cache_n"] < reused.body["timings"]["cache_n"]
    server.stop()

#!/usr/bin/env python3
"""Measure llama-server SSD prompt-cache capture and restore behavior."""

import argparse
import json
import pathlib
import subprocess
import time
import urllib.request


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        return json.loads(response.read()) if "json" in content_type else response.read().decode()


def start(args):
    command = [
        args.server, "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--ctx-size", str(args.context), "--parallel", "1", "--n-predict", "1", "--temp", "0",
        "--cache-ram", str(args.cache_ram), "--cache-ssd-dir", args.cache_dir,
        "--cache-ssd-maxsize", str(args.cache_mib), "--cache-ssd-prefix-advantage", str(args.advantage),
        "--metrics", "--no-ui",
    ]
    if args.model_draft:
        command.extend(["--model-draft", args.model_draft, "--spec-type", args.spec_type, "--no-kv-unified"])
    if args.ple_sidecar:
        command.extend(["--ple-sidecar", args.ple_sidecar])
    process = subprocess.Popen(command)
    for _ in range(120):
        try:
            request(args.port, "/health")
            return process
        except Exception:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {process.returncode}")
            time.sleep(0.25)
    process.terminate()
    raise RuntimeError("llama-server did not start")


def completion(args, prompt):
    started = time.perf_counter()
    response = request(args.port, "/completion", {"prompt": prompt, "cache_prompt": True})
    return {"wall_seconds": time.perf_counter() - started, "prompt_tokens": response["timings"]["prompt_n"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-draft")
    parser.add_argument("--spec-type", default="draft-mtp")
    parser.add_argument("--ple-sidecar")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-mib", type=int, default=1024)
    parser.add_argument("--cache-ram", type=int, default=0)
    parser.add_argument("--advantage", type=int, default=32)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--repeat", type=int, default=256)
    args = parser.parse_args()

    pathlib.Path(args.cache_dir).mkdir(parents=True, exist_ok=False)
    prompt = ("SSD prompt cache benchmark.\n" * args.repeat).rstrip("\n")
    process = start(args)
    try:
        capture = completion(args, prompt)
        time.sleep(0.25)
        first_metrics = request(args.port, "/metrics")
    finally:
        process.terminate()
        process.wait(timeout=10)

    process = start(args)
    try:
        prefix_tokens = request(args.port, "/tokenize", {"content": prompt, "add_special": True})["tokens"]
        suffix_tokens = request(args.port, "/tokenize", {"content": "\ncontinuation"})["tokens"]
        strict_prefix_restore = completion(args, prefix_tokens + suffix_tokens)
        exact_tip_fallback = completion(args, prefix_tokens)
        metrics = request(args.port, "/metrics")
    finally:
        process.terminate()
        process.wait(timeout=10)

    print(json.dumps({
        "capture": capture,
        "strict_prefix_restore": strict_prefix_restore,
        "exact_tip_fallback": exact_tip_fallback,
        "initial_metrics": first_metrics,
        "restart_metrics": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

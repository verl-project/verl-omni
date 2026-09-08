# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import socket


def ephemeral_port_range() -> tuple[int, int]:
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            lo, hi = f.read().split()
        return int(lo), int(hi)
    except (OSError, ValueError):
        return 32768, 60999  # Linux default


def get_non_ephemeral_free_port(address: str = "127.0.0.1") -> int:
    """Pick a free port below the kernel's ephemeral port range.

    The consumer binds the port only after worker spawn + model load, and ports
    inside the ephemeral range can be claimed as source ports by unrelated
    connections in that window (later failing to listen with ``EADDRINUSE``).
    """
    lo, _ = ephemeral_port_range()
    if lo <= 1024:
        raise RuntimeError(f"Ephemeral port range starts at {lo}; no non-privileged candidate ports below it.")
    candidates = range(1024, lo)
    start = random.randrange(len(candidates))
    for offset in range(len(candidates)):
        port = candidates[(start + offset) % len(candidates)]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free non-ephemeral port on {address} below {lo}.")

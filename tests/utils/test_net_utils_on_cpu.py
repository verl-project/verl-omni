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
"""CPU tests for the non-ephemeral MASTER_PORT picker."""

import socket

import pytest

from verl_omni.utils import net_utils


def _bind(address: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((address, port))
    return sock


def test_returns_bindable_port_below_ephemeral_range():
    port = net_utils.get_non_ephemeral_free_port("127.0.0.1")
    lo, _ = net_utils.ephemeral_port_range()
    assert 1024 <= port < lo
    holder = _bind("127.0.0.1", port)  # raises if the picker handed out a busy port
    holder.close()


def test_skips_ports_already_bound(monkeypatch):
    monkeypatch.setattr(net_utils, "ephemeral_port_range", lambda: (1030, 60999))
    holders = [_bind("127.0.0.1", port) for port in range(1024, 1029)]
    try:
        assert net_utils.get_non_ephemeral_free_port("127.0.0.1") == 1029
    finally:
        for holder in holders:
            holder.close()


def test_raises_when_all_candidates_are_bound(monkeypatch):
    monkeypatch.setattr(net_utils, "ephemeral_port_range", lambda: (1027, 60999))
    holders = [_bind("127.0.0.1", port) for port in range(1024, 1027)]
    try:
        with pytest.raises(RuntimeError, match="No free non-ephemeral port"):
            net_utils.get_non_ephemeral_free_port("127.0.0.1")
    finally:
        for holder in holders:
            holder.close()


def test_raises_when_ephemeral_range_leaves_no_candidates(monkeypatch):
    monkeypatch.setattr(net_utils, "ephemeral_port_range", lambda: (1024, 60999))
    with pytest.raises(RuntimeError, match="no non-privileged candidate ports"):
        net_utils.get_non_ephemeral_free_port("127.0.0.1")

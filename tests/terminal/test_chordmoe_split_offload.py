"""Unit tests for ChordMoE All-Cards + Host-RAM hybrid split offload admission (`STRATA_CHORD_SPLIT_OFFLOAD=1`)."""
from __future__ import annotations

from strata_control import split_engine as se


def _qwen38_q4_facts(routed_gib: float = 72.0) -> se.ModelFacts:
    return se.ModelFacts(
        routed_types={"Q4_K": 144},
        routed_bytes=int(routed_gib * se.GIB),
        embd_type="Q8_0",
        shexp_types={"Q8_0": 144, "F32": 48},
        embd_ram_bytes=int(27.5 * se.GIB),
        layers=48,
        experts=512,
        ple_type="IQ4_NL",
        output_type="Q8_0",
    )


def test_default_split_refuses_partial_vram_fit(monkeypatch):
    monkeypatch.delenv("STRATA_CHORD_SPLIT_OFFLOAD", raising=False)
    cards = [
        se.Card(index=0, name="NVIDIA GeForce RTX 4090", vram_mib=24564, cc=89),
        se.Card(index=1, name="NVIDIA GeForce RTX 4090", vram_mib=24564, cc=89),
    ]
    dec = se.decide(
        enabled=True,
        system="linux",
        cards=cards,
        ram_total_gib=256.0,
        ram_available_gib=220.0,
        model=_qwen38_q4_facts(72.0),
    )
    assert not dec.use_split
    assert dec.offload_ram_gib == 0
    assert any("do not fit" in r for r in dec.reasons)


def test_chord_split_offload_unlocks_2x4090_and_4x3090(monkeypatch):
    monkeypatch.setenv("STRATA_CHORD_SPLIT_OFFLOAD", "1")
    model = _qwen38_q4_facts(72.0)

    # 2x RTX 4090 (24 GB each, sm_89): ~15 GB expert room per card (~30 GB total), ~42-43 GiB offloaded to host RAM!
    cards_2x4090 = [
        se.Card(index=0, name="NVIDIA GeForce RTX 4090", vram_mib=24576, cc=89),
        se.Card(index=1, name="NVIDIA GeForce RTX 4090", vram_mib=24576, cc=89),
    ]
    dec_2 = se.decide(
        enabled=True,
        system="linux",
        cards=cards_2x4090,
        ram_total_gib=256.0,
        ram_available_gib=192.0,
        model=model,
    )
    assert dec_2.use_split, dec_2.reasons
    assert dec_2.offload_ram_gib >= 40
    cfg_2 = se.split_config(
        {"port": 8080},
        dec_2,
        exe="/bin/strata-split-2",
        native="/models/q38.gguf",
        mtp_dir="/models/mtp",
        draft_vocab="/models/vocab.gguf",
        context=65536,
    )
    assert cfg_2["env"]["FNX_EXPERT_RAM_GIB"] == str(dec_2.offload_ram_gib)
    assert cfg_2["env"]["FNX_ARENA"] == "0"

    # 4x RTX 3090 (24 GB each, sm_86): ~60 GiB expert room across 4 cards, ~12 GiB offloaded to host RAM!
    cards_4x3090 = [
        se.Card(index=i, name="NVIDIA GeForce RTX 3090", vram_mib=24576, cc=86)
        for i in range(4)
    ]
    dec_4 = se.decide(
        enabled=True,
        system="linux",
        cards=cards_4x3090,
        ram_total_gib=256.0,
        ram_available_gib=128.0,
        model=model,
    )
    assert dec_4.use_split, dec_4.reasons
    assert 1 <= dec_4.offload_ram_gib < dec_2.offload_ram_gib


def test_chord_split_offload_still_checks_host_ram():
    model = _qwen38_q4_facts(72.0)
    cards_2x4090 = [
        se.Card(index=0, name="NVIDIA GeForce RTX 4090", vram_mib=24576, cc=89),
        se.Card(index=1, name="NVIDIA GeForce RTX 4090", vram_mib=24576, cc=89),
    ]
    # Only 48 GiB free RAM is not enough for 27.5 GiB embeddings + 42 GiB offloaded experts + 8 GiB spare!
    dec = se.decide(
        enabled=True,
        system="linux",
        cards=cards_2x4090,
        ram_total_gib=64.0,
        ram_available_gib=48.0,
        model=model,
        chord_offload=True,
    )
    assert not dec.use_split
    assert any("not enough free RAM" in r for r in dec.reasons)

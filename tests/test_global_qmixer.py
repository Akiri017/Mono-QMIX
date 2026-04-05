"""
Phase 2 Gate — GlobalQMixer Unit Tests

Tests shape correctness, NaN safety, and gradient flow.
Run from repo root:
    python tests/test_global_qmixer.py
"""

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

import torch
from modules.mixers.global_qmixer import GlobalQMixer

# ---------------------------------------------------------------------------
# Shared config — matches civiq_sumo.yaml values
# ---------------------------------------------------------------------------
ARGS = {
    "max_rsus": 12,
    "global_state_dim": 4485,
    "global_mixing_embed_dim": 32,
    "hypernet_layers": 2,
    "hypernet_embed": 64,
}

BATCH_SIZE = 4
MAX_RSUS = ARGS["max_rsus"]
GLOBAL_STATE_DIM = ARGS["global_state_dim"]


def build_mixer():
    return GlobalQMixer(ARGS)


# ---------------------------------------------------------------------------
# Test 1 — Full batch, all RSUs active
# ---------------------------------------------------------------------------
def test_full_batch():
    mixer = build_mixer()
    rsu_qtots = torch.randn(BATCH_SIZE, MAX_RSUS)
    global_states = torch.randn(BATCH_SIZE, GLOBAL_STATE_DIM)
    rsu_mask = torch.ones(BATCH_SIZE, MAX_RSUS)

    out = mixer(rsu_qtots, global_states, rsu_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in full-batch output"
    assert not torch.isinf(out).any(), "Inf in full-batch output"
    print("  Test 1 PASSED — full batch, all 12 RSUs active")


# ---------------------------------------------------------------------------
# Test 2 — Partial RSUs active (8 of 12)
# ---------------------------------------------------------------------------
def test_partial_rsus():
    mixer = build_mixer()
    global_states = torch.randn(BATCH_SIZE, GLOBAL_STATE_DIM)

    rsu_mask = torch.zeros(BATCH_SIZE, MAX_RSUS)
    rsu_mask[:, :8] = 1.0

    rsu_qtots = torch.randn(BATCH_SIZE, MAX_RSUS)
    rsu_qtots[:, 8:] = 0.0  # padded slots zeroed

    out = mixer(rsu_qtots, global_states, rsu_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in partial-RSU output"
    print("  Test 2 PASSED — partial batch (8/12 active RSUs)")


# ---------------------------------------------------------------------------
# Test 3 — Zero mask: no active RSUs (edge case)
# ---------------------------------------------------------------------------
def test_empty_network():
    mixer = build_mixer()
    global_states = torch.randn(BATCH_SIZE, GLOBAL_STATE_DIM)

    rsu_mask = torch.zeros(BATCH_SIZE, MAX_RSUS)
    rsu_qtots = torch.zeros(BATCH_SIZE, MAX_RSUS)

    out = mixer(rsu_qtots, global_states, rsu_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in empty-network output"
    print("  Test 3 PASSED — empty network (all-zero mask), no NaN")
    print(f"           Empty-network output sample: {out[0].item():.4f} (bias contribution)")


# ---------------------------------------------------------------------------
# Test 4 — Gradient flow through all parameters
# ---------------------------------------------------------------------------
def test_gradient_flow():
    mixer = build_mixer()

    rsu_qtots = torch.randn(BATCH_SIZE, MAX_RSUS, requires_grad=True)
    global_states = torch.randn(BATCH_SIZE, GLOBAL_STATE_DIM, requires_grad=True)
    rsu_mask = torch.ones(BATCH_SIZE, MAX_RSUS)

    out = mixer(rsu_qtots, global_states, rsu_mask)
    loss = out.sum()
    loss.backward()

    assert rsu_qtots.grad is not None, "No gradient on rsu_qtots"
    assert not torch.isnan(rsu_qtots.grad).any(), "NaN gradient on rsu_qtots"

    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"

    print("  Test 4 PASSED — gradients flow to all parameters without NaN")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nGlobalQMixer unit tests")
    print(f"  max_rsus         = {MAX_RSUS}")
    print(f"  global_state_dim = {GLOBAL_STATE_DIM}  (peak 69 agents * obs_dim 65 — LOS E validated)")
    print(f"  embed_dim        = {ARGS['global_mixing_embed_dim']}")
    print(f"  hypernet_layers  = {ARGS['hypernet_layers']}")
    print()

    test_full_batch()
    test_partial_rsus()
    test_empty_network()
    test_gradient_flow()

    print("\nPHASE 2 GATE PASSED")

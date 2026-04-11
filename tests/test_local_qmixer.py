"""
Phase 1 Gate — LocalQMixer Unit Tests

Tests shape correctness, NaN safety, and gradient flow.
Run from repo root:
    python tests/test_local_qmixer.py
"""

import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

import torch
from modules.mixers.local_qmixer import LocalQMixer

# ---------------------------------------------------------------------------
# Shared config — matches civiq_sumo.yaml values
# ---------------------------------------------------------------------------
ARGS = {
    "max_agents_per_rsu": 28,
    "obs_dim": 65,
    "local_mixing_embed_dim": 32,
    "hypernet_layers": 2,
    "hypernet_embed": 64,
}

BATCH_SIZE = 4
MAX_AGENTS = ARGS["max_agents_per_rsu"]
LOCAL_STATE_DIM = MAX_AGENTS * ARGS["obs_dim"]  # 28 * 65 = 1820


def build_mixer():
    return LocalQMixer(ARGS)


# ---------------------------------------------------------------------------
# Test 1 — Full batch, all agents real
# ---------------------------------------------------------------------------
def test_full_batch():
    mixer = build_mixer()
    agent_qs = torch.randn(BATCH_SIZE, MAX_AGENTS)
    local_states = torch.randn(BATCH_SIZE, LOCAL_STATE_DIM)
    agent_mask = torch.ones(BATCH_SIZE, MAX_AGENTS)

    out = mixer(agent_qs, local_states, agent_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in full-batch output"
    assert not torch.isinf(out).any(), "Inf in full-batch output"
    print("  Test 1 PASSED — full batch, all real agents")


# ---------------------------------------------------------------------------
# Test 2 — Partial batch: 10 real agents, 18 padded
# ---------------------------------------------------------------------------
def test_partial_agents():
    mixer = build_mixer()
    local_states = torch.randn(BATCH_SIZE, LOCAL_STATE_DIM)

    agent_mask = torch.zeros(BATCH_SIZE, MAX_AGENTS)
    agent_mask[:, :10] = 1.0

    agent_qs = torch.randn(BATCH_SIZE, MAX_AGENTS)
    agent_qs[:, 10:] = 0.0  # padded slots zeroed

    out = mixer(agent_qs, local_states, agent_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in partial-agent output"
    print("  Test 2 PASSED — partial batch (10/28 real agents)")


# ---------------------------------------------------------------------------
# Test 3 — Zero mask: empty zone edge case
# ---------------------------------------------------------------------------
def test_empty_zone():
    mixer = build_mixer()
    local_states = torch.randn(BATCH_SIZE, LOCAL_STATE_DIM)

    agent_mask = torch.zeros(BATCH_SIZE, MAX_AGENTS)
    agent_qs = torch.zeros(BATCH_SIZE, MAX_AGENTS)

    out = mixer(agent_qs, local_states, agent_mask)

    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in empty-zone output"
    # Output should NOT be zero — bias terms (b1, V) still contribute
    print("  Test 3 PASSED — empty zone (all-zero mask), no NaN")
    print(f"           Empty-zone output sample: {out[0].item():.4f} (bias contribution)")


# ---------------------------------------------------------------------------
# Test 4 — Gradient flow through all parameters
# ---------------------------------------------------------------------------
def test_gradient_flow():
    mixer = build_mixer()

    agent_qs = torch.randn(BATCH_SIZE, MAX_AGENTS, requires_grad=True)
    local_states = torch.randn(BATCH_SIZE, LOCAL_STATE_DIM, requires_grad=True)
    agent_mask = torch.ones(BATCH_SIZE, MAX_AGENTS)

    out = mixer(agent_qs, local_states, agent_mask)
    loss = out.sum()
    loss.backward()

    assert agent_qs.grad is not None, "No gradient on agent_qs"
    assert not torch.isnan(agent_qs.grad).any(), "NaN gradient on agent_qs"

    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"

    print("  Test 4 PASSED — gradients flow to all parameters without NaN")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nLocalQMixer unit tests")
    print(f"  max_agents_per_rsu = {MAX_AGENTS}")
    print(f"  obs_dim            = {ARGS['obs_dim']}")
    print(f"  local_state_dim    = {LOCAL_STATE_DIM}")
    print(f"  embed_dim          = {ARGS['local_mixing_embed_dim']}")
    print(f"  hypernet_layers    = {ARGS['hypernet_layers']}")
    print()

    test_full_batch()
    test_partial_agents()
    test_empty_zone()
    test_gradient_flow()

    print("\nPHASE 1 GATE PASSED")

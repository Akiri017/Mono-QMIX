"""
Phase 3 Gate — HierarchicalQLearner Unit Tests

Tests construction, parameter grouping, target network independence,
and save/load round-trip. Does NOT run a full training step (Phase 4
batch fields not yet available).

Run from repo root:
    python tests/test_hierarchical_learner.py
"""

import os
import sys
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "pymarl", "src"))

from learners.hierarchical_q_learner import HierarchicalQLearner

# ---------------------------------------------------------------------------
# Minimal MAC stub
# ---------------------------------------------------------------------------
class StubAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class StubMAC:
    """Minimal MAC stub matching BasicMAC's interface used by HierarchicalQLearner."""

    def __init__(self):
        self.agent = StubAgent()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def init_hidden(self, batch_size):
        pass

    def forward(self, batch, t):
        return torch.zeros(1, 1, 1)

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")

    def load_models(self, path):
        self.agent.load_state_dict(
            torch.load(f"{path}/agent.pth", map_location="cpu")
        )


# ---------------------------------------------------------------------------
# Shared args — matches civiq_sumo.yaml values
# ---------------------------------------------------------------------------
RSU_CONFIG_PATH = os.path.join(repo_root, "config", "envs", "rsu", "synthetic_4x4.yaml")

ARGS = {
    # Core
    "n_agents": 32,
    "n_actions": 4,
    "use_cuda": False,
    # Learning
    "gamma": 0.99,
    "td_lambda": 0.8,
    "double_q": True,
    "grad_norm_clip": 10,
    "lr": 0.0005,
    "optim_eps": 1e-5,
    # Target network
    "target_update_interval": 200,
    "target_update_mode": "hard",
    "tau": 0.001,
    # LocalQMixer
    "max_agents_per_rsu": 28,
    "obs_dim": 65,
    "local_mixing_embed_dim": 32,
    "hypernet_layers": 2,
    "hypernet_embed": 64,
    # GlobalQMixer
    "max_rsus": 12,
    "global_state_dim": 4485,
    "global_mixing_embed_dim": 32,
    # RSU zone manager
    "rsu_config": RSU_CONFIG_PATH,
}


def build_learner():
    mac = StubMAC()
    scheme = {"state": {"vshape": 4485}}
    logger = None
    return HierarchicalQLearner(mac, scheme, logger, ARGS)


# ---------------------------------------------------------------------------
# Test 1 — Construction: all components present
# ---------------------------------------------------------------------------
def test_construction():
    learner = build_learner()

    assert learner.local_mixer is not None,        "local_mixer missing"
    assert learner.target_local_mixer is not None, "target_local_mixer missing"
    assert learner.global_mixer is not None,       "global_mixer missing"
    assert learner.target_global_mixer is not None, "target_global_mixer missing"
    assert learner.zone_manager is not None,       "zone_manager missing"
    assert len(list(learner.params)) > 0,          "params list is empty"

    print("  Test 1 PASSED — all components constructed")
    return learner


# ---------------------------------------------------------------------------
# Test 2 — Parameter count: all three groups present in self.params
# ---------------------------------------------------------------------------
def test_parameter_count(learner):
    mac_params     = set(id(p) for p in learner.mac.parameters())
    local_params   = set(id(p) for p in learner.local_mixer.parameters())
    global_params  = set(id(p) for p in learner.global_mixer.parameters())
    all_params     = set(id(p) for p in learner.params)

    assert mac_params.issubset(all_params),    "MAC params missing from self.params"
    assert local_params.issubset(all_params),  "LocalQMixer params missing from self.params"
    assert global_params.issubset(all_params), "GlobalQMixer params missing from self.params"

    mac_count    = sum(p.numel() for p in learner.mac.parameters())
    local_count  = sum(p.numel() for p in learner.local_mixer.parameters())
    global_count = sum(p.numel() for p in learner.global_mixer.parameters())
    total_count  = sum(p.numel() for p in learner.params)

    print("  Test 2 PASSED — all parameter groups present in self.params")
    print(f"           MAC params:          {mac_count:,}")
    print(f"           LocalQMixer params:  {local_count:,}")
    print(f"           GlobalQMixer params: {global_count:,}")
    print(f"           Total in optimizer:  {total_count:,}")


# ---------------------------------------------------------------------------
# Test 3 — Target network independence
# ---------------------------------------------------------------------------
def test_target_independence(learner):
    # Modify first param of local_mixer — target must be unaffected
    local_param = next(iter(learner.local_mixer.parameters()))
    target_local_param = next(iter(learner.target_local_mixer.parameters()))
    original_target_val = target_local_param.data.clone()
    local_param.data.add_(1.0)
    assert torch.allclose(target_local_param.data, original_target_val), \
        "target_local_mixer changed when local_mixer was modified"
    local_param.data.sub_(1.0)  # restore

    # Modify first param of global_mixer — target must be unaffected
    global_param = next(iter(learner.global_mixer.parameters()))
    target_global_param = next(iter(learner.target_global_mixer.parameters()))
    original_target_val = target_global_param.data.clone()
    global_param.data.add_(1.0)
    assert torch.allclose(target_global_param.data, original_target_val), \
        "target_global_mixer changed when global_mixer was modified"
    global_param.data.sub_(1.0)  # restore

    print("  Test 3 PASSED — target networks are independent copies")


# ---------------------------------------------------------------------------
# Test 4 — save_models / load_models round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(learner):
    checkpoint_path = "/tmp/civiq_test_checkpoint"

    # Save
    learner.save_models(checkpoint_path)

    assert os.path.exists(f"{checkpoint_path}/local_mixer.th"),  "local_mixer.th not saved"
    assert os.path.exists(f"{checkpoint_path}/global_mixer.th"), "global_mixer.th not saved"
    assert os.path.exists(f"{checkpoint_path}/agent.pth"),       "agent.pth not saved"

    # Load into a fresh learner
    learner2 = build_learner()
    learner2.load_models(checkpoint_path)

    # Compare local_mixer weights
    for (n, p1), (_, p2) in zip(
        learner.local_mixer.named_parameters(),
        learner2.local_mixer.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"local_mixer weight mismatch: {n}"

    # Compare global_mixer weights
    for (n, p1), (_, p2) in zip(
        learner.global_mixer.named_parameters(),
        learner2.global_mixer.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"global_mixer weight mismatch: {n}"

    print("  Test 4 PASSED — save/load round-trip weights match")
    print(f"           Checkpoint path: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nHierarchicalQLearner unit tests")
    print(f"  max_agents_per_rsu  = {ARGS['max_agents_per_rsu']}")
    print(f"  max_rsus            = {ARGS['max_rsus']}")
    print(f"  global_state_dim    = {ARGS['global_state_dim']}")
    print(f"  rsu_config          = {ARGS['rsu_config']}")
    print()

    learner = test_construction()
    test_parameter_count(learner)
    test_target_independence(learner)
    test_save_load_roundtrip(learner)

    print("\nPHASE 3 GATE PASSED")

"""
Integration Test for PyMARL QMIX Implementation

Tests that all components work together correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np


def test_agent():
    """Test RNN agent module."""
    print("\n=== Testing RNN Agent ===")
    from modules.agents.rnn_agent import RNNAgent

    args = {
        "n_actions": 5,
        "agent_hidden_dim": 64,
        "agent_rnn_type": "gru"
    }

    agent = RNNAgent(input_shape=20, args=args)
    hidden = agent.init_hidden()

    # Forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 20)  # (batch=4, obs_dim=20)
    hidden = hidden.expand(batch_size, -1)  # Expand to match batch size
    q_vals, hidden = agent(obs, hidden)

    assert q_vals.shape == (batch_size, 5), f"Expected ({batch_size}, 5), got {q_vals.shape}"
    print(f"[OK] Agent output shape: {q_vals.shape}")
    print(f"[OK] Q-values range: [{q_vals.min():.2f}, {q_vals.max():.2f}]")
    return True


def test_mixer():
    """Test QMIX mixer network."""
    print("\n=== Testing QMIX Mixer ===")
    from modules.mixers.qmix import QMixer

    args = {
        "n_agents": 3,
        "state_shape": 50,
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64
    }

    mixer = QMixer(args)

    # Forward pass
    agent_qs = torch.randn(4, 3)  # (batch=4, n_agents=3)
    states = torch.randn(4, 50)  # (batch=4, state_dim=50)
    q_tot = mixer(agent_qs, states)

    assert q_tot.shape == (4, 1), f"Expected (4, 1), got {q_tot.shape}"
    print(f"[OK] Mixer output shape: {q_tot.shape}")
    print(f"[OK] Q_tot range: [{q_tot.min():.2f}, {q_tot.max():.2f}]")
    return True


def test_episode_buffer():
    """Test episode buffer."""
    print("\n=== Testing Episode Buffer ===")
    from components.episode_buffer import EpisodeBatch, ReplayBuffer

    # Create scheme
    scheme = {
        "state": {"vshape": 50},
        "obs": {"vshape": 20, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (5,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.uint8}
    }
    groups = {"agents": 3}

    # Create buffer
    buffer = ReplayBuffer(scheme, groups, buffer_size=10, max_seq_length=21)

    # Create episode batch
    batch = EpisodeBatch(scheme, groups, batch_size=2, max_seq_length=21)

    # Fill with dummy data
    batch.data.transition_data["state"][:, :10] = torch.randn(2, 10, 50)
    batch.data.transition_data["obs"][:, :10] = torch.randn(2, 10, 3, 20)
    batch.data.transition_data["actions"][:, :10] = torch.randint(0, 5, (2, 10, 3, 1))
    batch.data.transition_data["avail_actions"][:, :10] = torch.ones(2, 10, 3, 5, dtype=torch.int)
    batch.data.transition_data["reward"][:, :10] = torch.randn(2, 10, 1)
    batch.data.transition_data["terminated"][:, :10] = torch.zeros(2, 10, 1, dtype=torch.uint8)
    batch.data.transition_data["filled"][:, :10] = torch.ones(2, 10, 1, dtype=torch.uint8)

    # Insert to buffer
    buffer.insert_episode_batch(batch)
    assert len(buffer) == 2, f"Expected 2 episodes in buffer, got {len(buffer)}"

    print(f"[OK] Buffer has {len(buffer)} episodes")

    # Sample from buffer
    sampled = buffer.sample(2)
    assert sampled["state"].shape == (2, 21, 50)
    print(f"[OK] Sampled batch shape: {sampled['state'].shape}")

    return True


def test_controller():
    """Test Multi-Agent Controller."""
    print("\n=== Testing Multi-Agent Controller ===")
    from controllers.basic_controller import BasicMAC

    args = {
        "n_agents": 3,
        "n_actions": 5,
        "agent_hidden_dim": 64,
        "agent_rnn_type": "gru",
        "use_cuda": False,
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 50000
    }

    scheme = {
        "obs": {"vshape": 20},
    }
    groups = {"agents": 3}

    mac = BasicMAC(scheme, groups, args)
    mac.init_hidden(batch_size=2)

    print(f"[OK] MAC initialized with {args['n_agents']} agents")
    print(f"[OK] Agent parameters: {sum(p.numel() for p in mac.parameters())} params")

    return True


def test_integration():
    """Test full integration."""
    print("\n=== Testing Full Integration ===")

    # Mock environment info
    env_info = {
        "n_agents": 3,
        "n_actions": 5,
        "state_shape": 50,
        "obs_shape": 20,
        "episode_limit": 20
    }

    args = {
        **env_info,
        "agent_hidden_dim": 64,
        "agent_rnn_type": "gru",
        "use_cuda": False,
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": 50000,
        "mixing_embed_dim": 32,
        "hypernet_layers": 2,
        "hypernet_embed": 64,
        "gamma": 0.99,
        "td_lambda": 0.8,
        "double_q": True,
        "grad_norm_clip": 10,
        "target_update_interval": 200,
        "lr": 0.0005,
        "log_interval": 5000
    }

    # Create scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
        "filled": {"vshape": (1,), "dtype": torch.uint8}
    }
    groups = {"agents": env_info["n_agents"]}

    # Create components
    from controllers.basic_controller import BasicMAC
    from learners.q_learner import QLearner
    from utils.logging import Logger
    from components.episode_buffer import EpisodeBatch

    logger = Logger()
    mac = BasicMAC(scheme, groups, args)
    learner = QLearner(mac, scheme, logger, args)

    print(f"[OK] MAC created with {args['n_agents']} agents")
    print(f"[OK] Learner created with mixer network")

    # Create dummy episode batch
    batch = EpisodeBatch(scheme, groups, batch_size=4, max_seq_length=21)

    # Fill with dummy data (10 timesteps)
    T = 10
    batch.data.transition_data["state"][:, :T+1] = torch.randn(4, T+1, 50)
    batch.data.transition_data["obs"][:, :T+1] = torch.randn(4, T+1, 3, 20)
    batch.data.transition_data["actions"][:, :T] = torch.randint(0, 5, (4, T, 3, 1))
    batch.data.transition_data["avail_actions"][:, :T+1] = torch.ones(4, T+1, 3, 5, dtype=torch.int)
    batch.data.transition_data["reward"][:, :T] = torch.randn(4, T, 1) * 0.1
    batch.data.transition_data["terminated"][:, :T] = torch.zeros(4, T, 1, dtype=torch.uint8)
    batch.data.transition_data["filled"][:, :T] = torch.ones(4, T, 1, dtype=torch.uint8)

    # Train on batch
    print("\nTraining on dummy batch...")
    stats = learner.train(batch, t_env=0, episode_num=1)

    print(f"[OK] Training step completed")
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Grad norm: {stats['grad_norm']:.4f}")
    print(f"  Mean Q-value: {stats['q_mean']:.4f}")
    print(f"  Mean target: {stats['target_mean']:.4f}")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("PyMARL QMIX Integration Test")
    print("="*60)

    tests = [
        ("RNN Agent", test_agent),
        ("QMIX Mixer", test_mixer),
        ("Episode Buffer", test_episode_buffer),
        ("Multi-Agent Controller", test_controller),
        ("Full Integration", test_integration)
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] {name} FAILED:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "[OK]" if success else "[FAIL]"
        print(f"{symbol} {name}: {status}")

    all_passed = all(success for _, success in results)
    print("="*60)
    if all_passed:
        print("\n[OK] All tests passed! PyMARL QMIX implementation is ready.")
    else:
        print("\n[FAIL] Some tests failed. Please review errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

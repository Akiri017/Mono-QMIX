"""
Episode Buffer Components

EpisodeBatch: Stores a batch of episodes
ReplayBuffer: Stores and samples batches of episodes for training
"""

import torch
import numpy as np
from types import SimpleNamespace as SN


class EpisodeBatch:
    """
    Stores a batch of episodes for training.

    Args:
        scheme: Data scheme defining what data to store
        groups: Agent groups (not used, kept for compatibility)
        batch_size: Number of episodes in batch
        max_seq_length: Maximum episode length
        preprocess: Preprocessing functions
        device: torch device
    """

    def __init__(self, scheme, groups, batch_size, max_seq_length, preprocess=None, device="cpu"):
        self.scheme = scheme
        self.groups = groups or {}
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = preprocess or {}
        self.device = device

        # Create data structure
        self.data = SN()
        self.data.transition_data = {}
        self.data.episode_data = {}

        # Initialize tensors for each field in scheme
        for field_key, field_info in scheme.items():
            shape = field_info.get("vshape", 1)
            dtype = field_info.get("dtype", torch.float32)
            group = field_info.get("group", None)

            if isinstance(shape, int):
                shape = (shape,)

            # Determine total shape
            if group:
                # Agent-level data: (batch, time, n_agents, *shape)
                n_agents = groups[group]
                full_shape = (batch_size, max_seq_length, n_agents, *shape)
            else:
                # Global data: (batch, time, *shape)
                full_shape = (batch_size, max_seq_length, *shape)

            # Create tensor
            if field_info.get("episode_const", False):
                # Episode-level constant (not per-timestep)
                full_shape = (batch_size, *shape)
                self.data.episode_data[field_key] = torch.zeros(full_shape, dtype=dtype, device=device)
            else:
                # Transition-level (per-timestep)
                self.data.transition_data[field_key] = torch.zeros(full_shape, dtype=dtype, device=device)

    def __getitem__(self, item):
        """Get data by key."""
        if item in self.data.transition_data:
            return self.data.transition_data[item]
        elif item in self.data.episode_data:
            return self.data.episode_data[item]
        else:
            raise KeyError(f"Key {item} not found in episode batch")

    def update(self, data, ts=None):
        """
        Update batch with new data.

        Args:
            data: Dict of data to update
            ts: Timestep to update (if None, updates episode data)
        """
        for k, v in data.items():
            if k in self.data.transition_data:
                # Convert to tensor
                v_tensor = self._to_tensor(v, k)
                if ts is not None:
                    self.data.transition_data[k][:, ts] = v_tensor
                else:
                    self.data.transition_data[k][:] = v_tensor
            elif k in self.data.episode_data:
                v_tensor = self._to_tensor(v, k)
                self.data.episode_data[k][:] = v_tensor

    def _to_tensor(self, data, key):
        """Convert data to tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)

        # Convert list/numpy to tensor
        scheme_info = self.scheme[key]
        dtype = scheme_info.get("dtype", torch.float32)

        if isinstance(data, list):
            data = np.array(data)

        tensor = torch.tensor(data, dtype=dtype, device=self.device)

        # Apply preprocessing if available
        if key in self.preprocess:
            tensor = self.preprocess[key](tensor)

        return tensor

    def to(self, device):
        """Move batch to device."""
        for k in self.data.transition_data:
            self.data.transition_data[k] = self.data.transition_data[k].to(device)
        for k in self.data.episode_data:
            self.data.episode_data[k] = self.data.episode_data[k].to(device)
        self.device = device
        return self


class ReplayBuffer:
    """
    Replay buffer for storing and sampling episodes.

    Args:
        scheme: Data scheme
        groups: Agent groups
        buffer_size: Maximum number of episodes to store
        max_seq_length: Maximum episode length
        preprocess: Preprocessing functions
        device: torch device
    """

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        self.scheme = scheme
        self.groups = groups
        self.buffer_size = buffer_size
        self.max_seq_length = max_seq_length
        self.preprocess = preprocess
        self.device = device

        # Create buffer as one large EpisodeBatch
        self.buffer = EpisodeBatch(scheme, groups, buffer_size, max_seq_length,
                                   preprocess=preprocess, device=device)

        # Buffer management
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, episode_batch):
        """
        Insert a batch of episodes into the buffer.

        Args:
            episode_batch: EpisodeBatch to insert
        """
        batch_size = episode_batch.batch_size

        # Handle buffer overflow
        if self.buffer_index + batch_size <= self.buffer_size:
            # Fits in buffer
            indices = slice(self.buffer_index, self.buffer_index + batch_size)
        else:
            # Wrap around
            # For simplicity, just use the first episodes that fit
            remaining = self.buffer_size - self.buffer_index
            if remaining > 0:
                indices = slice(self.buffer_index, self.buffer_size)
                batch_size = remaining
            else:
                # Buffer full, start from beginning
                self.buffer_index = 0
                indices = slice(0, batch_size)

        # Copy data
        for k in self.buffer.data.transition_data:
            self.buffer.data.transition_data[k][indices] = episode_batch[k][:batch_size].to(self.device)

        for k in self.buffer.data.episode_data:
            self.buffer.data.episode_data[k][indices] = episode_batch[k][:batch_size].to(self.device)

        # Update indices
        self.buffer_index = (self.buffer_index + batch_size) % self.buffer_size
        self.episodes_in_buffer = min(self.episodes_in_buffer + batch_size, self.buffer_size)

    def sample(self, batch_size):
        """
        Sample a batch of episodes from the buffer.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            EpisodeBatch with sampled episodes
        """
        assert self.can_sample(batch_size), "Not enough episodes in buffer"

        # Random sampling
        indices = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)

        # Create new batch
        sampled_batch = EpisodeBatch(self.scheme, self.groups, batch_size,
                                     self.max_seq_length, preprocess=self.preprocess,
                                     device=self.device)

        # Copy sampled data
        for k in sampled_batch.data.transition_data:
            sampled_batch.data.transition_data[k][:] = self.buffer.data.transition_data[k][indices]

        for k in sampled_batch.data.episode_data:
            sampled_batch.data.episode_data[k][:] = self.buffer.data.episode_data[k][indices]

        return sampled_batch

    def can_sample(self, batch_size):
        """Check if buffer has enough episodes to sample."""
        return self.episodes_in_buffer >= batch_size

    def __len__(self):
        """Return number of episodes in buffer."""
        return self.episodes_in_buffer

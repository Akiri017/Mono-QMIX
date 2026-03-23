"""
Simple Logger for PyMARL

Logs training statistics to console and optionally to TensorBoard.
"""

import os
from collections import defaultdict


class Logger:
    """
    Simple logger for training metrics.

    Args:
        console_logger: Console logger instance (optional)
        use_tensorboard: Whether to use TensorBoard logging
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, console_logger=None, use_tensorboard=False, log_dir="results/logs"):
        self.console_logger = console_logger
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir

        self.stats = defaultdict(list)
        self.writer = None

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard logging enabled. Run: tensorboard --logdir={log_dir}")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

    def log_stat(self, key, value, t):
        """
        Log a statistic.

        Args:
            key: Stat name
            value: Stat value
            t: Timestep
        """
        self.stats[key].append((t, value))

        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar(key, value, t)

    def print_recent_stats(self):
        """Print recent statistics."""
        if not self.stats:
            return

        print("\n--- Recent Stats ---")
        for key, values in self.stats.items():
            if values:
                t, value = values[-1]
                print(f"{key}: {value:.4f} (t={t})")
        print("-------------------\n")

        # Clear stats after printing
        self.stats.clear()

    def log_episode_stats(self, episode_num, t_env, episode_return):
        """Log episode statistics."""
        print(f"[Episode {episode_num}] t_env={t_env}, return={episode_return:.2f}")

    def close(self):
        """Close logger (and TensorBoard writer)."""
        if self.writer is not None:
            self.writer.close()

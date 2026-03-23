"""
RNN Agent Module for QMIX

Individual agent Q-network with RNN (GRU) for handling partial observability.
Each agent outputs Q-values for its available actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """
    RNN-based agent network that outputs Q-values for each action.

    Args:
        input_shape: Dimension of observation (obs_dim)
        args: Configuration dict with:
            - n_actions: Number of actions
            - agent_hidden_dim: Hidden dimension of RNN
            - agent_rnn_type: 'gru' or 'lstm'
    """

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.n_actions = args["n_actions"]
        self.hidden_dim = args.get("agent_hidden_dim", 64)
        self.rnn_type = args.get("agent_rnn_type", "gru").lower()

        # Input layer: observation -> hidden
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)

        # RNN layer
        if self.rnn_type == "gru":
            self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")

        # Output layer: hidden -> Q-values
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)

    def init_hidden(self):
        """Initialize hidden state for RNN."""
        # For GRU: single hidden state
        # For LSTM: (hidden, cell) tuple
        if self.rnn_type == "gru":
            return self.fc1.weight.new(1, self.hidden_dim).zero_()
        else:  # lstm
            return (
                self.fc1.weight.new(1, self.hidden_dim).zero_(),
                self.fc1.weight.new(1, self.hidden_dim).zero_()
            )

    def forward(self, inputs, hidden_state):
        """
        Forward pass through agent network.

        Args:
            inputs: Observation tensor, shape (batch, input_shape)
            hidden_state: RNN hidden state

        Returns:
            q_values: Q-values for each action, shape (batch, n_actions)
            hidden_state: Updated RNN hidden state
        """
        # Input layer with ReLU
        x = F.relu(self.fc1(inputs))

        # RNN update
        if self.rnn_type == "gru":
            h = self.rnn(x, hidden_state)
            hidden_state = h
        else:  # lstm
            h, c = self.rnn(x, (hidden_state[0], hidden_state[1]))
            hidden_state = (h, c)
            x = h  # Use hidden state for output

        # Output layer (Q-values)
        q = self.fc2(h if self.rnn_type == "gru" else hidden_state[0])

        return q, hidden_state

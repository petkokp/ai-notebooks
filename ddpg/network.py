import os
import pathlib
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):

    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir='ckpt'):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(
            ckpt_dir, self.name + 'actor_ddpg_ckpt')

        layers = [
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.n_actions),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

        for layer in self.model[:-2]:
            f = 1 / np.sqrt(layer[0].weight.data.size()[0])
            T.nn.init.uniform_(layer[0].weight.data, -f, f)
            T.nn.init.uniform_(layer[0].bias.data, -f, f)

        f3 = 0.003
        T.nn.init.uniform_(self.model[-2].weight.data, -f3, f3)
        T.nn.init.uniform_(self.model[-2].bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if not os.path.exists(ckpt_dir):
            pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    def forward(self, state):
        return self.model(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print("Model saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print("Model loaded")


class CriticNetwork(nn.Module):

    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir='ckpt'):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(
            ckpt_dir, self.name + 'critic_ddpg_ckpt')

        layers = [
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.LayerNorm(self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.LayerNorm(self.fc2_dims),
            nn.Linear(self.n_actions, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        ]

        self.model = nn.Sequential(*layers)

        for layer in self.model[:-1]:
            f = 1 / np.sqrt(layer[0].weight.data.size()[0])
            T.nn.init.uniform_(layer[0].weight.data, -f, f)
            T.nn.init.uniform_(layer[0].bias.data, -f, f)

        f3 = 0.0003
        T.nn.init.uniform_(self.model[-1].weight.data, -f3, f3)
        T.nn.init.uniform_(self.model[-1].bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.model[:6](state)
        action_value = self.model[6](action)
        state_action_value = F.relu(T.add(state_value, action_value))
        return self.model[7](state_action_value)

    def save_checkpoint(self):
        print("Saving Checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading Checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))

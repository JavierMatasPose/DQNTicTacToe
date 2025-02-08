import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for value function approximation.
    """

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size)
        self.fc3 = nn.Linear(4 * input_size, 7 * output_size)
        self.fc4 = nn.Linear(7 * output_size, 5 * output_size)
        self.fc5 = nn.Linear(5 * output_size, 3 * output_size)
        self.fc6 = nn.Linear(3 * output_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class DDQNAgent:
    """
    Double Deep Q-Learning Agent.
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Neural networks
        self.model = MLP(self.state_size, self.action_size)
        self.target_model = MLP(self.state_size, self.action_size)

        # Synchronize target model with main model
        self.update_target_model()

    def update_target_model(self):
        """
        Synchronizes target model parameters with the main model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """
        Returns an action that considers only valid moves.
        """
        # state is expected to be a 2D array with shape (1, state_size)
        valid_actions = [
            i for i in range(self.action_size) if state.flatten()[i] == 0
        ]

        state_tensor = torch.FloatTensor(state)
        q_values = self.model(state_tensor).detach().numpy()
        valid_q_values = {
            action: q_values[0][action]
            for action in valid_actions
        }
        return max(valid_q_values, key=valid_q_values.get)

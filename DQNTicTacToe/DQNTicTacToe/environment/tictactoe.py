import gymnasium as gym
import numpy as np

# Reward values for specific game outcomes
from config.config import REWARD_SCHEME


class TicTacToeEnv(gym.Env):

    def __init__(self):
        """
        Initialize the Tic Tac Toe environment.
        """
        super(TicTacToeEnv, self).__init__()

        # Observation space: 3x3 grid where -1, 0, or 1 represents O, empty, or X
        self.observation_space = gym.spaces.Box(low=-1,
                                                high=1,
                                                shape=(3, 3),
                                                dtype=np.int32)

        # Action space: 9 discrete actions representing the cells of the grid
        self.action_space = gym.spaces.Discrete(9)

        # Initialize the game state
        self.board = np.zeros((3, 3), dtype=np.int32)
        self.current_player = 1  # Player 1 starts (X)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            observation (ndarray): The empty board state.
            info (dict): Additional info (empty here).
        """

        self.board = np.zeros((3, 3), dtype=np.int32)
        self.current_player = 1
        return self.board, {}

    def step(self, action):
        """
        Apply an action (place a marker) and return the updated state.

        Args:
            action (int): The index (0-8) of the cell to place the marker.

        Returns:
            observation (ndarray): Updated board state.
            reward (float): Reward for the action.
            done (bool): Whether the game has ended.
            info (dict): Additional game info (e.g., winner or error).
        """

        # Get the row and the column based on the action
        row, col = divmod(action,
                          3)  # Action = 2 => (0, 2); Action = 7 => (2, 1) ...

        # print("board \n", self.board)
        # print("action", action)
        # Handle invalid moves
        if self._checkinvalidmove(row, col):
            return (
                self.board,
                REWARD_SCHEME["invalid_move"],
                True,
                {
                    "error": "Invalid move"
                },
            )

        # Place the current player's marker, either -1 or 1
        self.board[row, col] = self.current_player
        # print("board \n", self.board)
        # Check for a winner
        if self._check_winner():
            return (
                self.board,
                REWARD_SCHEME["win"],
                True,
                {
                    "winner": self.current_player
                },
            )

        # Check for a draw
        if self._checkdraw():
            return self.board, REWARD_SCHEME["draw"], True, {"draw": True}

        # If the game is not finished yet, then switch player
        self.current_player *= -1

        return self.board, -0.4, False, {}

    def get_obs(self):
        """Return the current observation (board state)."""
        return self.board

    def _checkdraw(self):
        """Check if the game ends in a draw. All cells are occupied."""
        return np.all(self.board != 0)

    def _checkinvalidmove(self, row, col):
        """Check if the move is invalid. Not 0 cells are already occupied"""
        return self.board[row, col] != 0

    def _check_winner(self):
        """Check if the current player has won."""
        # Check rows and columns
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player):  # Check row i
                return True
            if np.all(self.board[:,
                                 i] == self.current_player):  # Check column i
                return True

        # Check diagonals
        if np.all(np.diagonal(self.board) ==
                  self.current_player):  # Main diagonal
            return True
        if np.all(np.diagonal(np.fliplr(self.board)) ==
                  self.current_player):  # Anti-diagonal
            return True

        return False

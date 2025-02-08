from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
from environment.tictactoe import TicTacToeEnv
from agents.ddqnagent import DDQNAgent
import os

app = Flask(__name__)

# Initialize the game environment and agents
env = TicTacToeEnv()
agent = DDQNAgent(
    env.observation_space.shape[0] * env.observation_space.shape[1],
    env.action_space.n,
)
# Load pre-trained DQN model
agent.model.load_state_dict(
    torch.load("models/DQN_TICTACTOE_0.pth", map_location=torch.device("cpu"))
)

# Reset game state
env.reset()
current_board = env.get_obs()
game_over = False


def board_to_symbols(board):
    """Convert board numbers to symbols for display: 1 -> 'X', -1 -> 'O', 0 -> ''."""
    symbols = {1: "X", -1: "O", 0: ""}
    # Convert the NumPy board to a list of lists with symbols
    return [[symbols[int(cell)] for cell in row] for row in board]


@app.route("/")
def index():
    """Render the main game page."""
    return render_template("index.html", board=board_to_symbols(current_board))


@app.route("/move", methods=["POST"])
def move():
    """Receive a move from the user and process both the human and agent moves."""
    global current_board, game_over, env

    # Get the cell index from the POSTed JSON data
    data = request.get_json()
    try:
        cell = int(data["cell"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input."}), 400

    # Validate the cell number
    if cell < 0 or cell >= 9:
        return jsonify({"error": "Cell number must be between 0 and 8."}), 400

    # Determine row and column
    row, col = divmod(cell, 3)
    if current_board[row, col] != 0:
        return jsonify({"error": "Cell already taken."}), 400

    # Process the human move
    next_obs, reward, done, info = env.step(cell)
    current_board = next_obs
    if done:
        game_over = True
        result = "Game over! (Human move ended the game)"
        return jsonify(
            {
                "board": board_to_symbols(current_board),
                "game_over": game_over,
                "result": result,
            }
        )

    # Process the DQN agent’s move
    state_np = current_board.flatten()
    action = agent.get_action(np.reshape(state_np, [1, agent.state_size]))
    next_obs, reward, done, info = env.step(action)
    current_board = next_obs

    if done:
        game_over = True
        result = "Game over! (Agent move ended the game)"
    else:
        result = ""

    return jsonify(
        {
            "board": board_to_symbols(current_board),
            "game_over": game_over,
            "result": result,
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the game state."""
    global current_board, game_over, env
    env.reset()
    current_board = env.get_obs()
    game_over = False
    return jsonify({"board": board_to_symbols(current_board), "game_over": game_over})


if __name__ == "__main__":
    # Use the port provided by the environment, default to 5000 if not available.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

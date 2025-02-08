# DQNTicTacToe

Full-Stack application to play against a self-taught Deep Q-Network agent for Tic Tac Toe game.

In addition, you can play the agent via our deployed webapp at:

[dqntictactoe](https://dqntictactoe.onrender.com/)

## Description

DQNTicTacToe is a Python project that implements Deep Q-Learning for playing Tic Tac Toe. The project uses Flask for web interface, Gymnasium for game mechanics, and PyTorch for the deep learning components.

## Installation

To install the dependencies, you can use either Poetry or pip with a custom index source.

### Using Poetry

1. Install Poetry:
   ```bash
   curl -sSL https://install-poetry.org | python -
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/JavierMatasPose/DQNTicTacToe.git
   cd DQNTicTacToe
   ```
3. Install the dependencies:
   ```bash
   poetry install
   ```

## Dependencies

The project requires the following packages:

- Flask==3.1.0
- Gymnasium==1.0.0
- NumPy==2.1.3
- PyTorch==2.5.1 (GPU version)


## Usage

Once installed, you can start the application server with:

```bash
python app.py
```

For more detailed instructions on how to use the DQN implementation, please refer to the project documentation or the source code.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests. For major changes, please open an issue first to discuss any proposed features.

import chess
import json
import numpy as np
import pickle
import random
from tensorflow import keras 

# this list specifies which element of the square array should be flagged for a given piece
# negative pieces denote black, positive white, but the code is later reversed and the negative taken
# in order to render positions strategically symmetric
SQUARE_ENCODER = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])

# helper param definitions
symb_map_white = {chess.PIECE_SYMBOLS[piece]: piece for piece in chess.PIECE_TYPES}
symb_map_black = {
    chess.PIECE_SYMBOLS[piece].upper(): -piece for piece in chess.PIECE_TYPES
}
str_int_pos_map = {**symb_map_white, **symb_map_black}

int_str_pos_map = {v: k for k, v in str_int_pos_map.items() if type(v) != list}
int_str_pos_map.update({0: " "})

# add empty space
str_int_pos_map.update({str(i): [0] * i for i in range(9)})


def flatten(l: list):
    """flatten a list w/ up to one layer of nesting"""
    return [v for i in l for v in (i if isinstance(i, list) else [i])]


def record_game():
    """play a game and return the outcome"""

    board = chess.Board()

    position_list = []

    while not board.is_game_over():
        board_fen = board.fen()
        position_list.append(board_fen)

        # choose random move - lighter way to do this??
        legal_moves = list(board.generate_legal_moves())
        chosen_move = random.sample(legal_moves, 1)[0]

        board.push(chosen_move)

    winner = board.outcome().winner
    # cleansed_winner = [winner == 0, winner is None, winner == 1]
    if winner is not None:
        return {"winner": winner, "positions": position_list}
    
    # draws aren't that interesting, we'll just drop them
    return

def fen_to_ints(fen_str: str, reverse: bool = False):
    fen_split = fen_str.split(" ")
    placement = fen_split[0]
    row_list = placement.split("/")

    board_repr = [
        flatten([
            str_int_pos_map.get(piece) for piece in row
        ]) for row in row_list
    ]

    if reverse:
        for row in board_repr:
            row.reverse()
            row *= 1

    vector_repr = [
        [val == SQUARE_ENCODER for val in row] for row in board_repr 
        # val == SQUARE_ENCODER for row in board_repr for val in row
    ]

    return vector_repr


def fill_row_empty_chars(row: str):
    for i in reversed(range(1, 9)):
        row = row.replace(" " * i, str(i))
    return row


def ints_to_fen(int_list: list):
    """reverses fen_to_ints()"""
    fen_partial = [
        "".join(e for e in [int_str_pos_map.get(x) for x in row]) for row in int_list
    ]

    fen = "/".join(x for x in [fill_row_empty_chars(row) for row in fen_partial])

    return fen


def preprocess_fen_str(fen_str: str, winner):
    """
    fen notation:
    0: placement data (str)
    1: active color (str)
    2: castling availability (str, one of '-' or any combination of K,Q,k,q)
    3: en passant square (str, ignore for now)
    4: halfmove clock (int)
    5: fullmove number (int)
    """

    fen_split = fen_str.split(" ")
    placement = fen_split[0]

    # if active color is black, reverse the board
    reverse = fen_split[1] != 'w'
    board_repr = fen_to_ints(placement, reverse=reverse)
    winner = np.flip(winner) if reverse else winner
    # if reverse:
        # winner.reverse()

    return board_repr, winner


def preprocess_raw_game_data(game_data: dict):

    # this maps raw fen position sets to outcomes
    position_winner_map = {
        position: game["winner"]
        for game in game_data.values()
        for position in game["positions"]
    }

    # this converts raw fen to normalized numeric matrix
    Xy_pairs = [
        preprocess_fen_str(position, winner)
        for position, winner in position_winner_map.items()
    ]

    return Xy_pairs


def generate_data(num_games: int = 10):
    # raw data generation
    game_data = {i: record_game() for i in range(num_games)}

    # drop draws
    game_data = {k:v for k,v in game_data.items() if v is not None}

    with open("artifacts/game_data.json", "w") as f:
        json.dump(game_data, f)

    Xy_pairs = preprocess_raw_game_data(game_data)

    with open("artifacts/prepped_game_data.pkl", "wb") as f:
        pickle.dump(Xy_pairs, f)


def train_model():

    with open("artifacts/prepped_game_data.pkl", "rb") as f:
        Xy_pairs = pickle.load(f)

    X = np.array([x[0] for x in Xy_pairs])
    y = np.array([x[1] for x in Xy_pairs])

    keras.utils.set_random_seed(0)

    model = keras.models.Sequential([
        keras.layers.Dense(100, input_shape=(8, 8, 12)),
        keras.layers.Conv2D(
            filters=20,
            kernel_size=2,
            strides=(1, 1),
            activation='relu',
        ),
        keras.layers.Dense(
            10, 
            activation='relu',
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.01
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x=X,
        y=y,
        epochs=1,
        validation_split=0.25
    )

    model.save('artifacts/model.keras')

# e.g.,
# from chaturanga import generate_data, train_model
# generate_data()
# train_model()
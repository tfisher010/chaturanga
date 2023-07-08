import chess
import json
import numpy as np
import pickle
import random
from sklearn.neural_network import MLPClassifier

# helper param definitions
symb_map_white = {chess.PIECE_SYMBOLS[piece]: piece for piece in chess.PIECE_TYPES}
symb_map_black = {
    chess.PIECE_SYMBOLS[piece].upper(): -piece for piece in chess.PIECE_TYPES
}
str_int_pos_map = symb_map_white | symb_map_black

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
    cleansed_winner = [winner == 0, winner is None, winner == 1]

    return {"winner": cleansed_winner, "positions": position_list}


def fen_to_ints(fen_str: str, flatten_output: bool = True):
    fen_split = fen_str.split(" ")
    placement = fen_split[0]
    row_list = placement.split("/")
    board_repr = np.array(
        [flatten([str_int_pos_map.get(piece) for piece in row]) for row in row_list]
    )
    if flatten_output:
        nx, ny = board_repr.shape
        board_repr = board_repr.reshape((nx * ny))
    return board_repr


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


def preprocess_fen_str(fen_str: str):
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

    board_repr = fen_to_ints(placement)

    fen_split[0] = board_repr

    # active color
    fen_split[1] = fen_split[1] == 'w'

    # castling availability
    fen_split[2] = [c in fen_split[2] for c in ["K", "Q", "k", "q"]]

    # halfmove clock
    fen_split[4] = float(fen_split[4])

    # fullmove clock
    fen_split[5] = float(fen_split[5])

    # drop en passant square, i'm too dumb
    del fen_split[3]

    return fen_split


def preprocess_raw_game_data(game_data: dict):
    position_winner_map = {
        position: game["winner"]
        for game in game_data.values()
        for position in game["positions"]
    }

    # take first two (most essential) elements of fen decomposition: piece locations and current turn
    Xy_pairs = [
        (preprocess_fen_str(position), winner)
        for position, winner in position_winner_map.items()
    ]

    return Xy_pairs


def generate_data(num_games: int = 10):
    # raw data generation
    game_data = {i: record_game() for i in range(num_games)}

    with open("artifacts/game_data.json", "w") as f:
        json.dump(game_data, f)

    Xy_pairs = preprocess_raw_game_data(game_data)

    with open("artifacts/prepped_game_data.pkl", "wb") as f:
        pickle.dump(Xy_pairs, f)


def train_model():
    with open("artifacts/prepped_game_data.pkl", "rb") as f:
        Xy_pairs = pickle.load(f)

    X = [np.concatenate([x[0][0], [x[0][1]]]) for x in Xy_pairs]
    y = [x[1] for x in Xy_pairs]
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        early_stopping=True,
        random_state=1,
        verbose=True,
    )
    clf.fit(X, y)

    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(clf, f)

# e.g.,
# from chaturanga import generate_data, train_model
# generate_data()
# train_model()
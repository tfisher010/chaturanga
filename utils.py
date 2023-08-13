import chess
import numpy as np
from constants import MODEL_FILEPATH
from datetime import datetime
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


def print_update(*update_str):
    print(str(datetime.now()), *update_str)
    # print(str(datetime.now().replace(microsecond=0)), *update_str)


def flatten(l):
    """flatten arraylike w/ up to one layer of nesting"""
    return [v for i in l for v in (i if hasattr(i, "__len__") else [i])]


def fen_to_ints(fen_str: str, reverse: bool = False):
    fen_split = fen_str.split(" ")
    placement = fen_split[0]
    row_list = placement.split("/")

    board_repr = [
        flatten([str_int_pos_map.get(piece) for piece in row]) for row in row_list
    ]

    if reverse:
        for row in board_repr:
            row.reverse()
            row *= 1

    vector_repr = [[val == SQUARE_ENCODER for val in row] for row in board_repr]

    return vector_repr


def load_model():
    model = keras.models.load_model(MODEL_FILEPATH)
    return model


def save_model(model: keras.Sequential):
    model.save(MODEL_FILEPATH)


# def fill_row_empty_chars(row: str):
#     for i in reversed(range(1, 9)):
#         row = row.replace(" " * i, str(i))
#     return row

# def ints_to_fen(int_list: list):
#     """reverses fen_to_ints()"""
#     fen_partial = [
#         "".join(e for e in [int_str_pos_map.get(x) for x in row]) for row in int_list
#     ]

#     fen = "/".join(x for x in [fill_row_empty_chars(row) for row in fen_partial])

#     return fen

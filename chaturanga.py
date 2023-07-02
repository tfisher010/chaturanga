import chess
import random 

# helper param definition
symb_map_white = {
    chess.PIECE_SYMBOLS[piece]: piece
    for piece in chess.PIECE_TYPES
}
symb_map_black = {
    chess.PIECE_SYMBOLS[piece].upper(): -piece
    for piece in chess.PIECE_TYPES
}
symb_map = symb_map_white | symb_map_black

# add empty space
symb_map.update({
    str(i): [0] * i
    for i in range(9)
})

def flatten(l: list):
    """flatten a list w/ up to one layer of nesting"""
    return [
        v for i in l for v in (
            i if isinstance(i,list) 
            else [i]
        )
    ]

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

    return {
        'winner': board.outcome().winner,
        'positions': position_list
    }

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

    fen_split = fen_str.split(' ')
    placement = fen_split[0]
    row_list = placement.split('/')
    board_repr = [
        flatten([symb_map.get(piece) for piece in row]) 
        for row in row_list
    ]
    fen_split[0] = board_repr

    # active color
    fen_split[1] = bool(fen_split[1])

    # castling availability
    fen_split[2] = [c in fen_split[2] for c in ['K', 'Q', 'k', 'q']]

    # halfmove clock
    fen_split[4] = float(fen_split[4])

    # fullmove clock
    fen_split[5] = float(fen_split[5])

    # drop en passant square, i'm too dumb
    del fen_split[3]

    return fen_split

def preprocess_raw_game_data(game_data: dict):
    position_winner_map = {
        position: game['winner']
        for game in game_data.values()
        for position in game['positions']
    }

    Xy_pairs = [
        (preprocess_fen_str(position), winner)
        for position, winner in position_winner_map.items()
    ]

    return Xy_pairs


# e.g., 
# from chaturanga import record_game, preprocess_raw_game_data
# import json 

# # raw data generation
# game_data = {
#     i: record_game() for i in range(10)
# }

# with open('game_data.json', 'w') as f:
#     json.dump(game_data, f)

# Xy_pairs = preprocess_raw_game_data(game_data)

# with open('prepped_game_data.txt', 'w') as f:
#     for line in Xy_pairs:
#         f.write(f"{line}\n")
import chess
import numpy as np
import random
from tensorflow import keras
from utils import flatten, fen_to_ints, print_update


def get_possible_new_states(f: str, return_moves=False):
    """
    given a position as fen, return one-level-out possible positions,
    with corresponding moves returned optionally
    """

    b = chess.Board(f)
    possible_states = []
    if return_moves:
        possible_moves = []
    for move in b.generate_legal_moves():
        b_temp = b.copy()
        b_temp.push(move)
        possible_states.append(b_temp.fen())
        if return_moves:
            possible_moves.append(move)
    if return_moves:
        return possible_states, possible_moves
    return possible_states


def grow_state(s):
    """given a board,"""

    return [
        flatten(
            [get_possible_new_states(possible_line) for possible_line in possible_move]
        )
        for possible_move in s
    ]


from time import time


def build_move_tree(base_state: str, depth=1):
    """
    build out a move tree, from a base state (fen), to specified depth

    return list of possible new states for every first move, along with first moves
    """

    # top-level move choices
    # print_update("Establishing base possible moves...")
    possible_new_states, first_moves = get_possible_new_states(
        base_state, return_moves=True
    )
    possible_new_states = [[state] for state in possible_new_states]

    # downstream outcome possibilities
    # print_update("Growing tree branches...")
    for _ in range(depth):
        possible_new_states = grow_state(possible_new_states)

    return possible_new_states, first_moves


def get_random_move(board: chess.Board):
    possible_moves = [move for move in board.generate_legal_moves()]
    # quick check for mates:
    for move in possible_moves:
        b_temp = board.copy()
        b_temp.push(move)
        if b_temp.is_checkmate():
            return move

    return random.choice(possible_moves)


from time import time


def get_best_move(b: chess.Board, model: keras.Sequential):
    # mate check:
    # print_update("Performing mate check...")
    for move in b.generate_legal_moves():
        b_temp = b.copy()
        b_temp.push(move)
        if b_temp.is_checkmate():
            return move, np.inf

    # print_update("Building move tree...")
    move_map, move_options = build_move_tree(b.fen(), depth=1)

    # print_update("Prepping tree data for scoring...")
    # not a tensor - possibly inhomogeneous
    model_input = [
        [
            fen_to_ints(
                submove,
                # reverse = not b.turn
            )
            for submove in move
        ]
        for move in move_map
    ]
    model_input_shape = [len(ele) for ele in model_input]

    # print_update("Scoring possible outcomes...")
    # this is tricky - scoring each set of outcome possibilities
    # (possibly of different lengths) takes too long
    # so, we flatten the outcomes, but then reassemble in shape of original list

    flattened_model_input_arr = np.array(flatten(model_input))
    scores_flat = model.predict(flattened_model_input_arr, verbose=0)

    # print_update("Compiling scores...")
    # oof
    scores = []
    score_subset = []
    score_subset_idx = 0
    score_idx = 0
    for score in scores_flat:
        score_subset_len = model_input_shape[score_subset_idx]
        score_subset.append(score)
        if len(score_subset) == score_subset_len:
            scores.append(score_subset)
            score_subset = []
            score_subset_idx += 1
            score_idx = 0
        else:
            score_idx += 1

    # scores give favorability for white, so if turn, minimax, else maximin
    # print_update("Performing minimax...")
    if b.turn:
        worst_score_per_move = [min(move_set) for move_set in scores]
        best_move_idx = worst_score_per_move.index(max(worst_score_per_move))
    else:
        worst_score_per_move = [max(move_set) for move_set in scores]
        best_move_idx = worst_score_per_move.index(min(worst_score_per_move))

    best_move_score = worst_score_per_move[best_move_idx]
    best_move = move_options[best_move_idx]

    return best_move, best_move_score

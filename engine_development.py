import chess
import json
import numpy as np
import pickle
from constants import MODEL_FILEPATH
from datetime import datetime
from model_development import train_model
from move_selection import get_best_move, get_random_move
from os import path, rename
from utils import fen_to_ints, load_model, save_model


def record_game(random_white: bool = False, random_black: bool = False):
    """play a game and return the outcome"""

    board = chess.Board()
    position_list = []

    if (not random_white) | (not random_black):
        model = load_model()

    while not board.is_game_over():
        board_fen = board.fen()
        position_list.append(board_fen)

        if (board.turn & random_white) | (not board.turn & random_black):
            chosen_move = get_random_move(board)
        else:
            chosen_move, _ = get_best_move(board, model)

        board.push(chosen_move)

    winner = board.outcome().winner
    # cleansed_winner = [winner == 0, winner is None, winner == 1]
    if winner is not None:
        return {"winner": winner, "positions": position_list}

    # draws aren't that interesting, just drop them
    return


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
    reverse = fen_split[1] != "w"
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


def generate_data(num_games: int = 10, random_play: bool = False):
    # raw data generation
    game_data = {
        i: record_game(random_white=random_play, random_black=random_play)
        for i in range(num_games)
    }

    # drop draws
    game_data = {k: v for k, v in game_data.items() if v is not None}

    with open("artifacts/game_data.json", "w") as f:
        json.dump(game_data, f)

    Xy_pairs = preprocess_raw_game_data(game_data)

    with open("artifacts/prepped_game_data.pkl", "wb") as f:
        pickle.dump(Xy_pairs, f)


def play_vs_random(n_games=10):
    results = []

    for i in range(n_games):
        game = record_game(random_white=True, random_black=False)
        if game:
            results.append(game["winner"])

    return {"wins": sum(results), "losses": len(results) - sum(results)}


def perform_learning_round(training_games=100, validation_games=100):
    random_play = not path.isfile(MODEL_FILEPATH)
    generate_data(num_games=training_games, random_play=random_play)
    model = train_model()
    save_model(model)
    return play_vs_random(validation_games)


def rename_model_file():
    fpath, ext = MODEL_FILEPATH.split(".")
    dt_str = str(datetime.now().replace(microsecond=0))
    dt_str = dt_str.replace(" ", "_")
    for ch in [":", "-"]:
        dt_str = dt_str.replace(ch, "")
    final_model_filepath = f"{fpath}_{dt_str}.{ext}"
    rename(MODEL_FILEPATH, final_model_filepath)


def train_engine(rounds=10):
    results = {}
    for i in range(rounds):
        print(f"Round {i}")
        results[i] = perform_learning_round()
    return results

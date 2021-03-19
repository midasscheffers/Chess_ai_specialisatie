import chess
import chess.engine
import random as r
import numpy


# from stockfish import Stockfish

# stfh = Stockfish("/content/stockfish_12_win_x64/stockfish_12_win_x64")


def random_board(max_depth=200):
    board = chess.Board()
    depth = r.randrange(0, max_depth)
    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = r.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break
    return board

def stockfish_score(board, depth):
    with chess.engine.SimpleEngine.popen_uci("/content/stockfish_12_win_x64/stockfish_12_win_x64") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score

# def stockfish_score(board, depth):
#     score = 0
#     stfh.set_fen_posision(board.fen)
#     return score

b = random_board()
print(b)
print(stockfish_score(b, 10))
import chess
import chess.engine
import random as r
from network import *
from get_data import *



net = NeuralNetwork([64, 100, 50, 20, 64])
net.load_from_file("tester.nw")

bord = chess.Board()


def random_board(max_depth=200):
        board = chess.Board()
        depth = 3
        if r.randint(0,1) == 1:
            depth = r.randrange(0, max_depth)
        else:
            depth = r.randrange(0, 10)
        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = r.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break
        
        return board


print(bord)
bord = random_board()
print(bord)
print(net.board_to_ai_inp(bord))
print(net.forward(net.board_to_ai_inp(bord)))
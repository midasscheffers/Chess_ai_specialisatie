import chess
import chess.engine
import random as r

piece_val_table = {"k":1.0, "q":0.7, "r":0.5, "b":0.3, "n":0.2, "p":0.1, ".":0.0, "K":-1.0, "Q":-0.7, "R":-0.5, "B":-0.3, "N":-0.2, "P":-0.1}
alfabet = ["a","b","c","d","e","f","g","h"]

class ChessData:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci("content/stockfish")

    def random_board(self, max_depth=200):
        board = chess.Board()
        depth = r.randrange(0, max_depth)
        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = r.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break
        return board
    
    def randomData(self, size, time_limit=0.1):
        data = []
        for i in range(size):
            b = self.random_board()
            limit = chess.engine.Limit(time=time_limit)

            result = self.engine.analyse(b, chess.engine.Limit(depth=10))
            score = result['score'].white().score()

            data.append([self.board_to_ai_inp(b), self.move_to_ai_out(self.engine.play(b,limit).move), score])
        return data
    

    def stockfish_score(self, board, depth):
        result = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score
    
    def move_to_ai_out(self, move):
        m = str(move)
        if m == "None":
            return [0 for i in range(64)]
        out_1 = int(alfabet.index(m[0])) * 8 + int(m[1])
        out_2 = int(alfabet.index(m[2])) * 8 + int(m[3])
        out = []
        for i in range(64):
            if i == out_1:
                out.append(1)
            elif i == out_2:
                out.append(.8)
            else:
                out.append(0)
        return out

    def board_to_ai_inp(self, board):
        b = str(board)
        inp = []
        for i in range(len(b)):
            if b[i] in piece_val_table:
                inp.append(piece_val_table[b[i]])
        return inp



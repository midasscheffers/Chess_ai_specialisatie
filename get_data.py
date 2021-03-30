import chess
import chess.engine
import random as r

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
            data.append([b, self.engine.play(b,limit).move])
        return data
    

    def stockfish_score(self, board, depth):
        result = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score
    
    def move_to_ai_out(self, move):
        pass

    def board_to_ai_inp(self, board):
        pass


dataGen = ChessData()

data = dataGen.randomData(10)
print(data)
dataGen.engine.quit()

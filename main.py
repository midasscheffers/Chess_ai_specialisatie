import chess
import chess.engine
import random as r
from network import *
from get_data import *


dataGen = ChessData()
net = NeuralNetwork([64, 20, 20, 64])

while True:
    data = dataGen.randomData(100)
    net.train(data, .1)
    testData = dataGen.randomData(1)
    print(net.get_cost(testData[0][0], testData[0][1]))


dataGen.engine.quit()
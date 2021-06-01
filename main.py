import chess
import chess.engine
import random as r
from network import *
from get_data import *



dataGen = ChessData()
net = NeuralNetwork([64, 100, 50, 20, 64])
net.load_from_file("tester.nw")

cycles = 100

for i in range(cycles):
    data = dataGen.randomData(10)
    net.train(data, .1)
    testData = dataGen.randomData(1)
    print(net.get_cost(testData[0][0], testData[0][1]))

## test file making
# net = NeuralNetwork([3, 4, 3, 2])

# cycles = 300

# for i in range(cycles):
#     net.train([[[-3,2,4], [0, 1]]], .01)
#     print(net.get_cost([-3,2,4], [0, 1]))

net.save_to_file("tester.nw")


dataGen.engine.quit()
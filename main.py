import chess
import chess.engine
import random as r
from network import *
from get_data import *


file_name = "main.nw"


dataGen = ChessData()
net = NeuralNetwork([64, 20, 60, 10, 64])

net.load_from_file(file_name)

cycles = 1000
batch_size = 1

for i in range(cycles):
    data = dataGen.randomData(batch_size)
    net.train(data, .1)
    testData = dataGen.randomData(1)
    print(i, net.get_cost(testData[0][0], testData[0][1]))
    net.save_to_file(file_name)

## test file making
# net = NeuralNetwork([3, 4, 3, 2])

# cycles = 300

# for i in range(cycles):
#     net.train([[[-3,2,4], [0, 1]]], .01)
#     print(net.get_cost([-3,2,4], [0, 1]))

net.save_to_file(file_name)


dataGen.engine.quit()
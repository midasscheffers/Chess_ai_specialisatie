from network import *

net = NeuralNetwork([3, 1, 3])
dat = [[[-2,-4,4], [1,0,0]]]

for i in range(5000):
    net.train(dat, .1)

print(net.get_cost([-2,-4,4], [1,0,0]))

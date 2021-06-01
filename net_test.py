from network import *


 ## test backprop
# net = NeuralNetwork([3, 1, 3])
# dat = [[[-2,-4,4], [1,0,0]]]

# for i in range(5000):
#     net.train(dat, .1)

# print(net.get_cost([-2,-4,4], [1,0,0]))

## test fle load

net = NeuralNetwork([2, 3, 1])
net.load_from_file("example.nw")
print(net.weights, "w", net.biases)
net.save_to_file("example_save.nw")
from network import *


 ## test backprop
# net = NeuralNetwork([3, 1, 3])
# dat = [[[-2,-4,4], [1,0,0]]]

# for i in range(1000):
#     net.train(dat, .1)

# print(net.get_cost([-2,-4,4], [1,0,0]))
# print(net.forward([-2,-4,4]))


 ## test backprop 2
net = NeuralNetwork([3, 2, 2])
dat = [[[-2,-4,4], [1,0]], [[3,-5,2], [0,1]]]

for i in range(1000):
    net.train(dat, .1)

print(net.get_cost([-2,-4,4], [1,0]), net.forward([-2,-4,4]))
print(net.get_cost([3,-5,2], [0,1]), net.forward([3,-5,2]))
# print(net.get_cost([-3,-5,4], [0,1,0]))
# print(net.forward([-2,-4,4]))
# print(net.forward([-3,-5,4]))


# test fle load

# net = NeuralNetwork([2, 3, 1])
# net.load_from_file("example_save.nw")

# net.save_to_file("example_save.nw")
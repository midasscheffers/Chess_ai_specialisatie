import math

def dot_mult_array(a1, a2):
    result = 0
    for i in range(len(a1)):
        result += a1[i] * a2[i]
    return result

class NeuralNetwork:
    def __init__(self, layers):
        self.values = []
        self.weights = []
        self.biases = []
        for i in range(len(layers)):
            l = layers[i]
            layer = []
            w_layer = []
            b_layer = []
            for j in range(l):
                # add base values
                layer.append(0)
                # add biases
                b_layer.append(0)
                w_layer_node = []
                if not i == 0:
                    for k in range(layers[i-1]):
                        # add weights
                        w_layer_node.append(1)
                w_layer.append(w_layer_node)
            self.weights.append(w_layer)
            self.biases.append(b_layer)
            self.values.append(layer)
        

    def forward(self, inp):
        if len(inp) == len(self.values[0]):
            self.values[0] = inp
        else:
            print("input should be as long as first layer")
            return None
        
        for l in range(len(self.values)):
            if l > 0:
                for n in range(len(self.values[l])):
                    self.values[l][n] = dot_mult_array(self.weights[l][n], self.values[l-1])
                    # self.values[l][n] = math.tanh(2 * self.values[l][n])
                    self.values[l][n] += self.biases[l][n]
        return self.values[-1]


net = NeuralNetwork([3, 1, 3])
print(net.values)
print(net.biases)
print(net.weights)
print(net.forward([2,4,5]))
print(net.values)
print(net.forward([2,4,4]))
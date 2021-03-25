import math


def dot_mult_array(a1, a2):
    result = 0
    for i in range(len(a1)):
        result += a1[i] * a2[i]
    return result


def activation_function(x):
    return math.tanh(x)
    

def derivitive_act_func(x):
    return 1 - ((math.tanh(x)) ** 2)


class NeuralNetwork:
    def __init__(self, layers):
        self.values = []
        self.values_no_act_func = []
        self.weights = []
        self.biases = []
        self.weights_changes = []
        self.biases_changes = []
        # setup layers
        for i in range(len(layers)):
            l = layers[i]
            layer = []
            layer_no_act = []
            w_layer = []
            b_layer = []
            wc_layer = []
            bc_layer = []
            for j in range(l):
                # add base values
                layer.append(0)
                layer_no_act.append(0)
                # add biases
                b_layer.append(0)
                bc_layer.append([])
                w_layer_node = []
                wc_layer_node = []
                if not i == 0:
                    for k in range(layers[i-1]):
                        # add weights
                        w_layer_node.append(1)
                        wc_layer_node.append([])
                w_layer.append(w_layer_node)
            self.weights.append(w_layer)
            self.biases.append(b_layer)
            self.weights_changes.append(wc_layer)
            self.biases_changes.append(bc_layer)
            self.values.append(layer)
            self.values_no_act_func.append(layer_no_act)
        

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
                    self.values[l][n] += self.biases[l][n]
                    self.values_no_act_func[l][n] = self.values[l][n]
                    self.values[l][n] = activation_function(self.values[l][n])
        return self.values[-1]

    def backprop_add_change(self, inp, expexted_out, learn_rate):
        out = self.forward(inp)
        cost = 0
        for i in range(len(out)):
            cost += (out[i] - expexted_out[i]) ** 2

        expexted_out_layer = expexted_out
        
        for l in reversed(range(len(self.values))):
            if l > 0:
                for n in range(len(self.values[l])):
                    for w in range(len(self.weights[l][n])):
                        ## calc cost
                        self.weights_changes.append(learn_rate * (self.values[l-1][w] * derivitive_act_func(self.values_no_act_func[l][n]) * 2 * (self.values[l][n]-expexted_out_layer[n])))


    def add_avr_change_to_weigths_and_biases(self):
        pass


    

net = NeuralNetwork([3, 1, 3])
print(net.values)
print(net.biases)
print(net.weights)
print(net.forward([2,4,5]))
print(net.values)
print(net.forward([2,4,4]))
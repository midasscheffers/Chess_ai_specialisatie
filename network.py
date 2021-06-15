import math
import os

piece_val_table = {"k":1.0, "q":0.7, "r":0.5, "b":0.3, "n":0.2, "p":0.1, ".":0.0, "K":-1.0, "Q":-0.7, "R":-0.5, "B":-0.3, "N":-0.2, "P":-0.1}

def dot_mult_array(a1, a2):
    result = 0
    for i in range(len(a1)):
        result += a1[i] * a2[i]
    return result


def activation_function(x):
    return math.tanh(x)
    

def derivitive_act_func(x):
    return 1 - ((math.tanh(x)) ** 2)


def avvr_list(l):
    if isinstance(l, list):
        leng = len(l)
        val = 0
        for i in range(leng):
            val += l[i]
        return val/leng
    return l


class NeuralNetwork:
    def __init__(self, layers):
        self.values = []
        self.values_no_act_func = []
        self.weights = []
        self.biases = []
        self.weights_changes = []
        self.biases_changes = []
        self.values_changes = []
        # setup layers
        for i in range(len(layers)):
            l = layers[i]
            layer = []
            vc_layer = []
            layer_no_act = []
            w_layer = []
            b_layer = []
            wc_layer = []
            bc_layer = []
            for j in range(l):
                # add base values
                layer.append(0)
                layer_no_act.append(0)
                vc_layer.append([])
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
                wc_layer.append(wc_layer_node)
            self.weights.append(w_layer)
            self.biases.append(b_layer)
            self.weights_changes.append(wc_layer)
            self.biases_changes.append(bc_layer)
            self.values.append(layer)
            self.values_changes.append(vc_layer)
            self.values_no_act_func.append(layer_no_act)
    

    def reset(self, layers):
        self.values = []
        self.values_no_act_func = []
        self.weights = []
        self.biases = []
        self.weights_changes = []
        self.biases_changes = []
        self.values_changes = []
        # setup layers
        for i in range(len(layers)):
            l = layers[i]
            layer = []
            vc_layer = []
            layer_no_act = []
            w_layer = []
            b_layer = []
            wc_layer = []
            bc_layer = []
            for j in range(l):
                # add base values
                layer.append(0)
                layer_no_act.append(0)
                vc_layer.append([])
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
                wc_layer.append(wc_layer_node)
            self.weights.append(w_layer)
            self.biases.append(b_layer)
            self.weights_changes.append(wc_layer)
            self.biases_changes.append(bc_layer)
            self.values.append(layer)
            self.values_changes.append(vc_layer)
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


    def backprop(self, inp, expexted_out, learn_rate):
        out = self.forward(inp)
        cost = 0
        for i in range(len(out)):
            cost += (out[i] - expexted_out[i]) ** 2

        self.values_changes[len(self.values)-1] = expexted_out
        
        for l in reversed(range(len(self.values))):
            if l > 0:
                for n in range(len(self.values[l])):
                    ## calc cost bias
                    self.biases_changes[l][n].append(-learn_rate * (derivitive_act_func(self.values_no_act_func[l][n]) * 2 * (self.values[l][n]-avvr_list(self.values_changes[l][n]))))
                    for w in range(len(self.weights[l][n])):
                        ## calc cost
                        self.weights_changes[l][n][w].append(-learn_rate * (self.values[l-1][w] * derivitive_act_func(self.values_no_act_func[l][n]) * 2 * (self.values[l][n]-avvr_list(self.values_changes[l][n]))))
                        ## calc cost for previous neuron
                        self.values_changes[l-1][w].append(-learn_rate * (self.weights[l][n][w] * derivitive_act_func(self.values_no_act_func[l][n]) * 2 * (self.values[l][n]-avvr_list(self.values_changes[l][n]))))

    
    def get_cost(self, inp, expexted_out):
        out = self.forward(inp)
        cost = 0
        for i in range(len(out)):
            cost += (out[i] - expexted_out[i]) ** 2
        return cost


    def add_avr_change_to_weigths_and_biases(self):
        for l in range(len(self.values)):
            if l > 0 :
                for n in range(len(self.values[l])):
                    self.biases[l][n] += avvr_list(self.biases_changes[l][n])
                    self.biases_changes[l][n] = []
                    for w in range(len(self.weights[l][n])):
                        self.weights[l][n][w] += avvr_list(self.weights_changes[l][n][w])
                        self.weights_changes[l][n][w] = []

    
    def train(self, data, learn_rate):
        for i in range(len(data)):
            self.backprop(data[i][0], data[i][1], learn_rate)
        self.add_avr_change_to_weigths_and_biases()
    

    def save_to_file(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)
        f = open(file_name, "x+")
        f.truncate(0)
        layer_str = ""
        for l in self.values:
            leng = len(l)
            layer_str += f',{leng}'
        f.write(f"{layer_str}\n:\n")
        for l in range(len(self.weights)):
            weights_str, bias_str = "", ""
            for n in range(len(self.weights[l])):
                bias_str += f",{self.biases[l][n]}"
                for w in range(len(self.weights[l][n])):
                    weights_str += f",{self.weights[l][n][w]}"
                weights_str += "%"
            f.write(f"{weights_str}&{bias_str}\n")
        f.close()


    def load_from_file(self, file_name):
        f = open(file_name, "r")
        data = f.read()
        data = data.split("\n:\n")
        data[0] = data[0].split(',')[1:]
        data[0] = [int(s) for s in data[0]]
        self.reset(data[0])
        data = data[1].split("\n")
        data = [d.split("&") for d in data]
        data = [[s.split("%") for s in d] for d in data]
        for l in range(len(data)):
            for p in range(len(data[l])):
                p_d = data[l][p]
                p_d = [i for i in p_d if not i == ""]
                for s in range(len(p_d)):
                    p_d[s] = p_d[s].split(',')
                p_d = [[i for i in c if not i == ""] for c in p_d]
                data[l][p] = p_d
        for l in range(len(self.weights)):
            if l > 0:
                for p in range(len(data[l])):
                    if p == 1:
                        for n in range(len(data[l][p][0])):
                            self.biases[l][n] = float(data[l][p][0][n])
                    elif p == 0:
                        for n in range(len(data[l][p])):
                            for w in range(len(data[l][p][n])):
                                self.weights[l][n][w] = float(data[l][p][n][w])
    
    def ai_to_move_out(self, out):
        print(out)
        alfabet = ["a", "b", "c", "d", "e", "f", "g", "h"]
        temp = ""
        if out == "None":
            return temp
        return out


    def board_to_ai_inp(self, board):
        piece_val_table = {"k":1.0, "q":0.7, "r":0.5, "b":0.3, "n":0.2, "p":0.1, ".":0.0, "K":-1.0, "Q":-0.7, "R":-0.5, "B":-0.3, "N":-0.2, "P":-0.1}
        b = str(board)
        inp = []
        for i in range(len(b)):
            if b[i] in piece_val_table:
                inp.append(piece_val_table[b[i]])
        return inp

    def get_ai_out_from_bord(self, bord_state):
        out = self.forward(self.board_to_ai_inp(bord_state))
        return self.ai_to_move_out(out)
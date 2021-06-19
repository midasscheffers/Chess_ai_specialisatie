import numpy as np
import random as r

def dot_mult_array(a1, a2):
    result = 0
    for i in range(len(a1)):
        result += a1[i] * a2[i]
    return result


class Matrix:
    def __init__(self, size_a, size_b):
        self.A = [[0 for _ in range(size_b)] for _ in range(size_a)]
        self.width = size_b
        self.height = size_a
    

    @staticmethod
    def mult(a, b):
        if isinstance(b, Matrix):
            b_trans = b.transpose()
            result = Matrix(a.height, b.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = dot_mult_array(a.A[i], b_trans.A[j])
            return result
        else:
            result = Matrix(a.height, a.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a.A[i][j] * b
            return result
    

    @staticmethod
    def add(a, b):
        if isinstance(b, Matrix):
            result = Matrix(a.height, b.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a.A[i][j] + b.A[i][j]
            return result
        else:
            result = Matrix(a.height, a.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a.A[i][j] + b
            return result
    

    @staticmethod
    def subtract(a, b):
        if isinstance(b, Matrix):
            result = Matrix(a.height, b.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a.A[i][j] - b.A[i][j]
            return result
        else:
            result = Matrix(a.height, a.width)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a.A[i][j] - b
            return result
    

    @staticmethod
    def random_matrix(a, b, max=1):
        result = Matrix(a, b)
        for i in range(result.height):
            for j in range(result.width):
                result.A[i][j] = r.random()*max
        return result
    

    # only works on one-dymensional arrays for now
    @staticmethod
    def from_array(a, dir=0):
        if dir == 0:
            result = Matrix(len(a), 1)
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a[i]
        if dir == 1:
            result = Matrix(1, len(a))
            for i in range(result.height):
                for j in range(result.width):
                    result.A[i][j] = a[j]
        return result
    

    def map(a, func):
        result = Matrix(a.height, a.width)
        for i in range(result.height):
            for j in range(result.width):
                result.A[i][j] = func(a.A[i][j])
        return result


    def transpose(self):
        if self == None:
            return
        trans = Matrix(self.width, self.height)
        for i in range(trans.height):
            for j in range(trans.width):
                trans.A[i][j] = self.A[j][i]
        return trans
    
    
    def print(self):
        cases_dict = {"tr":"┐", "br":"┘", "tl":"┌", "bl":"└", "s":"|", "ohl":"[", "ohr":"]"}
        if self.height == 1:
            out_str = cases_dict["ohl"]
            for j in range(self.width):
                out_str += f" {self.A[0][j]}"
            out_str += " " + cases_dict["ohr"]
        else:
            out_str = ""
            for i in range(self.height):
                if i == 0:
                    line = cases_dict["tl"]
                elif i == self.height-1:
                    line = cases_dict["bl"]
                else:
                    line = cases_dict["s"]
                for j in range(self.width):
                    line += f" {self.A[0][j]}"
                if i == 0:
                    line += " " + cases_dict["tr"]
                elif i == self.height-1:
                    line += " " + cases_dict["br"]
                else:
                    line += " " + cases_dict["s"]
                out_str += line + "\n"
        print(out_str)

    
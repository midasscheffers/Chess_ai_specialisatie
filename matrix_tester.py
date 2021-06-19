from matrix import *
import math


mat = Matrix(4,2)
mat.A[0][0] = 4
mat.A[1][0] = 2
mat.A[2][0] = 3
mat.A[3][0] = 2
mat.print()


# cat = Matrix(1,4)
# cat.A[0][0] = 4
# cat.A[0][1] = 7
# cat.A[0][2] = 2
# cat.A[0][3] = 2
# cat.print()


# tat = Matrix.mult(mat, cat)
# lat = Matrix.mult(tat, -1)
# print(tat.A)
# print(lat.A)
# print(Matrix.random_matrix(3,4).A)
# print(Matrix.random_matrix(10,1).A)



# # def maper(l,f):
# #     for i in range(len(l)):
# #         l[i] = f(l[i])

# def add_one(a):
#     return a + 1

# def activation_function(x):
#     return math.tanh(x)

# nat = tat.map(activation_function)
# print(nat.A)

# li = [3,4,5]
# maper(li, add_one)
# print(li)

# fa = Matrix.from_array([1,2,4,5,7], dir=1)
# print(fa.A)
# fa.print()
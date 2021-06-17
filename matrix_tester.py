from matrix import *


mat = Matrix(4,1)
mat.A[0][0] = 4
mat.A[1][0] = 2
mat.A[2][0] = 3
mat.A[3][0] = 2


cat = Matrix(1,4)
cat.A[0][0] = 4
cat.A[0][1] = 7
cat.A[0][2] = 2
cat.A[0][3] = 2


tat = Matrix.mult(mat, cat)
lat = Matrix.mult(tat, -1)
print(tat.A)
print(lat.A)
print(Matrix.random_matrix(3,4).A)
print(Matrix.random_matrix(10,1).A)



# def maper(l,f):
#     for i in range(len(l)):
#         l[i] = f(l[i])

def add_one(a):
    return a + 1

nat = tat.map(add_one)
print(nat.A)

# li = [3,4,5]
# maper(li, add_one)
# print(li)
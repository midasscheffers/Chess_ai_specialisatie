from matrix import *


mat = Matrix(2,3)
mat.A[0][0] = 4
mat.A[0][1] = 2
mat.A[0][2] = 3
mat.A[1][0] = 2
mat.A[1][1] = 4
mat.A[1][2] = 6


cat = Matrix(3,1)
cat.A[0][0] = 4
cat.A[1][0] = 7
cat.A[2][0] = 2


tat = Matrix.mult(mat, cat)
lat = Matrix.mult(tat, -1)
print(tat.A)
print(lat.A)
print(Matrix.random_matrix(3,4).A)

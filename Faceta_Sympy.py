from sympy import Matrix
from sympy import simplify 
from sympy.abc import x

# Alice's vertices with epsilon_A = 2x  following the order with the order:
#p(0|00) p(1|00) p(0|01) p(1|01) p(0|10) p(1|10) p(0|11) p(1|11)
V_A =  Matrix(     [[1, 0, 1, 0, 1-x, x, 1, 0],
                     [1, 0, 1, 0, 0, 1, x, 1-x],
                     [1, 0, 1-x, x, 1-x, x, 1, 0],
                     [1, 0, 1-x, x, 0, 1, x, 1-x],
                     [x, 1-x, 0, 1, 1-x, x, 1, 0],
                     [x, 1-x, 0, 1, 0, 1, x, 1-x],
                     [0, 1, 0, 1, 1-x, x, 1, 0],
                     [0, 1, 0, 1, 0, 1, x, 1-x],
                     [1-x, x, 1, 0, 1-x, x, 1, 0],
                     [1-x, x, 1, 0, 0, 1, x, 1-x],
                     [0, 1, x, 1-x, 1-x, x, 1, 0],
                     [0, 1, x, 1-x, 0, 1, x, 1-x],
                     [1-x, x, 1, 0, 0, 1, 0, 1],
                     [0, 1, x, 1-x, 0, 1, 0, 1],
                     [1-x, x, 1, 0, x, 1-x, 0, 1],
                     [0, 1, x, 1-x, 1, 0, 1, 0],
                     [0, 1, x, 1-x, 1, 0, 1-x, x],
                     [0, 1, x, 1-x, x, 1-x, 0, 1],
                     [1, 0, 1, 0, 1, 0, 1-x, x],
                     [1, 0, 1, 0, x, 1-x, 0, 1],
                     [1, 0, 1-x, x, 1, 0, 1, 0],
                     [1, 0, 1-x, x, 1, 0, 1-x, x],
                     [1, 0, 1-x, x, x, 1-x, 0, 1],
                     [x, 1-x, 0, 1, 1, 0, 1, 0],
                     [x, 1-x, 0, 1, 1, 0, 1-x, x],
                     [x, 1-x, 0, 1, x, 1-x, 0, 1],
                     [0, 1, 0, 1, 1, 0, 1-x, x],
                     [0, 1, 0, 1, x, 1-x, 0, 1],
                     [1, 0, 1-x, x, 0, 1, 0, 1],
                     [x, 1-x, 0, 1, 0, 1, 0, 1],
                     [1-x, x, 1, 0, 1, 0, 1, 0],
                     [1-x, x, 1, 0, 1, 0, 1-x, x],
                     [0, 1, 0, 1, 0, 1, 0, 1],
                     [1, 0, 1, 0, 0, 1, 0, 1],
                     [0, 1, 0, 1, 1, 0, 1, 0],
                     [1, 0, 1, 0, 1, 0, 1, 0]])

# Vertices of Bob's polytope for epsilon_B = 0, following the order with the order:
#p(0|00) p(1|00) p(0|01) p(1|01) p(0|10) p(1|10) p(0|11) p(1|11)
V_B = Matrix([[0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 0, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 0, 1, 1, 0],
              [1, 0, 1, 0, 1, 0, 1, 0]])

# Vertices of polytope of joint probabilities for Alice and Bob, following the order with the order:
#p(00|00) p(01|00) p(00|01) p(01|01) p(10|00) p(11|00) p(10|01) p(11|01) p(00|10) p(01|10) p(00|11) p(01|11) p(10|10) p(11|10) p(10|11) p(11|11)

V = Matrix.zeros(len(V_A[:,0])*len(V_B[:,0]), 16) 

for i in range(len(V_A[:,0])):
    for j in range(len(V_B[:,0])):
        V[len(V_B[:,0])*i + j, 0] = V_A[i, 0] * V_B[j, 0]
        V[len(V_B[:,0])*i + j, 1] = V_A[i, 0] * V_B[j, 1]
        V[len(V_B[:,0])*i + j, 2] = V_A[i, 2] * V_B[j, 2]
        V[len(V_B[:,0])*i + j, 3] = V_A[i, 2] * V_B[j, 3]
        V[len(V_B[:,0])*i + j, 4] = V_A[i, 1] * V_B[j, 0]
        V[len(V_B[:,0])*i + j, 5] = V_A[i, 1] * V_B[j, 1]
        V[len(V_B[:,0])*i + j, 6] = V_A[i, 3] * V_B[j, 2]
        V[len(V_B[:,0])*i + j, 7] = V_A[i, 3] * V_B[j, 3]
        V[len(V_B[:,0])*i + j, 8] = V_A[i, 4] * V_B[j, 4]
        V[len(V_B[:,0])*i + j, 9] = V_A[i, 4] * V_B[j, 5]
        V[len(V_B[:,0])*i + j, 10] = V_A[i, 6] * V_B[j, 6]
        V[len(V_B[:,0])*i + j, 11] = V_A[i, 6] * V_B[j, 7]
        V[len(V_B[:,0])*i + j, 12] = V_A[i, 5] * V_B[j, 4]
        V[len(V_B[:,0])*i + j, 13] = V_A[i, 5] * V_B[j, 5]
        V[len(V_B[:,0])*i + j, 14] = V_A[i, 7] * V_B[j, 6]
        V[len(V_B[:,0])*i + j, 15] = V_A[i, 7] * V_B[j, 7]

# Value of each vertice for the functional I_{epsilon_A}
Z = Matrix.zeros(len(V[:,0]), 1)
for i in range(len(V[:,0])):
    Z[i] = V[i,4] + V[i,3] + V[i,12] + V[i,10] - ((2+2*x)/(2-2*x))*(V[i,0] + V[i,7] + V[i,8] + V[i,14])
    Z[i] = simplify(Z[i])

# Extracting only the nonequal values
unique_Z = list(set(Z))
print(unique_Z)

# TODO: Colocar um check aqui mostrando que 1 eh o maior valor possivel

# Matrix containing all the vertices which achieve value equal to 1 on the functional I_{epsilon_A}
W = []
for i in range(len(Z)):
    if Z[i] == 1:
       W.append(V[i,:])

W = Matrix(W)

## TODO: Explicar que oq a gente ta interessado eh no numero de caras que sao independentes afim. Para isso da para usar o resultado que v1, ..., vn sao independentes
## se, e somente se, v'1, ..., v'n sao linearmente independentes, onde v'1 = [1, v1], ..., v'n = [1, vn]

# Matrix with a extra collumn of "1" for the extremal points that achieve the maximum of the functional I_{epsilon_A}
W_affine = Matrix.zeros(len(W[:,0]), 17)
for i in range(len(W[:,0])):
    W_affine[i,0] = 1
    for j in range(16):
        W_affine[i,j+1] = W[i,j]

# Matrix with a extra collumn of "1" for all extremal points
V_affine = Matrix.zeros(len(V[:,0]), 17)
for i in range(len(V[:,0])):
    V_affine[i,0] = 1
    for j in range(16):
        V_affine[i,j+1] = V[i,j]

# Number of affine independents extremal points that achieve the maximum of the functional I_{epsilon_A}
Dimension_face = W_affine.rank()

# Number of affine independents extremal points 
Dimension_politope = V_affine.rank()

print("The number of affine independents extremal vectors is: ", Dimension_politope)

print("The number of affine independents extremal vectors saturating the inequality is: ", Dimension_face)

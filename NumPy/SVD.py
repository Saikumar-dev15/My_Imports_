import numpy as np 
import matplotlib.pyplot as plt

A = np.array([[-1,2],[3,-2],[5,7]])
#print(A.shape)
U, d, VT = np.linalg.svd(A)       #V is already transposed
#print(U)                         # it is m*m matrics. its columns are left-singular vectors of A
#print(VT)                        # it is n*n matrics. its columns are right-singular vectors of A
#print(d)                          # it is m*n matrics. elemnets along its diagonal are singular values of A


x= np.diag(d)
#print(x)

# d must have the same dimensions as A
D = np.concatenate((np.diag(d), [[0,0]]),axis=0)
#print(D)

A_svd = np.dot(U, np.dot(D,VT))

tolerance = 1e-8
comparison = np.abs(A-A_svd) < tolerance
#print(comparison)


# SVD and eigendecomposition 
# Left Singular Vector of A = eigenvectors of AA^T
# Right Singular Vector of A = eigenvectors of A^TA
# Non-Zeros singular values of A = sqrt of eigenvalues L = sqrt eigenvalues R


#Left-singular  vectors
AAt = np.dot(A,A.T) 
lambdas1, V1 = np.linalg.eig(AAt)

sorted_indices1 = np.argsort(lambdas1)[::-1]        #sort eigenvalues and eigen vectors in descending order

lambdas1_sorted = lambdas1[sorted_indices1]

V1_sorted = V1[:, sorted_indices1]
#print(V1_sorted)

toleranceL= 1e-8
comparisonL = np.abs(abs(U) - abs(V1_sorted)) < toleranceL
#print(comparisonL)



# Right Singular vectors
AtA = np.dot(A.T,A)
lambdas2 , V2 = np.linalg.eig(AtA)

sorted_indices2 = np.argsort(lambdas2)[::-1]

lambdas2_sorted = lambdas2[sorted_indices2]

V2_sorted = V2[:, sorted_indices2]

#print(V2_sorted)

toleranceR = 1e-8
comparisonR = np.abs(abs(VT) - abs(V2_sorted)) < toleranceR
#print(comparisonR)


d == np.sqrt(lambdas2_sorted)
print(d)
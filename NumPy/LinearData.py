import numpy as np

x = np.array([[1,2],[3,4],[5,6],[7,8]])
n = x.sum()
#print(n)
m = x.sum(axis=0)  #summing over all the rows
z=x.sum(axis=1) #summing over all columns
#print(m)
#print(z)

y = np.array([[0, 1],[2, 3]])
d = np.dot(x,y)
#print(d)


x = np.linalg.norm(x)
#print(x)
yinv = np.linalg.inv(y)
#print(yinv)


I = np.array([[1, 2, 3], [4,5,6], [7, 8, 9]])

column_1 = I[:,0]
column_2 = I[:,1]
column_3 = I[:,2]

x = np.dot(column_1, column_2)
y = np.dot(column_2, column_3)
z = np.dot(column_3, column_1)

X = np.linalg.norm(x)
Y = np.linalg.norm(y)
Z = np.linalg.norm(z)

#print(X)
#print(Y)
#print(Z)

P = np.array([[-1,4], [2,-2]])
R = np.linalg.eig(P)
print(R)
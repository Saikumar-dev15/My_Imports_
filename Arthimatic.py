import numpy as np 

#scalar arithmetic

array = np.array([1,2,3])
#print(array+1)
#print(array - 2)
#print(array ** 2)

#vectorized math funcs
#print(np.sqrt(array))
#print(np.floor(array))
radii = np.array([1,2,3])
#print(np.pi * radii **2)



#Element-Wise arithmetic
array1 = np.array([1,2,3])
array2 = np.array([4,5,6])
#print(array1 + array2)
#print(array1 - array2)
#print(array1 * array2)
#print(array1 / array2)





#Comparision operators
scores = np.array([91, 55, 100, 73, 82,64])
#print(scores == 100)
#print(scores >= 60)
scores[scores < 60] =0
print(scores)
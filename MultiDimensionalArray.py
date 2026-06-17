import numpy as np 

array = np.array([[['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]])

#print(array.shape)
#print(array[0][2][1])
#print(array[0,0,1])

word = array[0,0,1] + array[0,0,2] + array[0,1,1]
print(word)

import numpy as np 

array = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]])

#array[start:end:step]

#print(array[::-2])  
#print(array[:, 0:2])     
#print(array[:, ::2])  
#print(array[:, 1::2])    
 
#print(array[0:2, 2:])                     #Row and Coloumns
#print(array[2: , 0:2])                      # [9,10] and [13,14]
print(array[2: , 2:])                        #[11,12] and [15,16]

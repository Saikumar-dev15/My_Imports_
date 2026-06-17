#Filtering = Refers to the process of selecting elements
#            from an array that much a given condition

import numpy as np
ages = np.array([[21,17,19,18,28,25,65],
                 [39,22,15,99,18,22,27]])

teenagers = ages[ages < 18]
#adults = ages[(ages>= 18) & (ages < 65)]
seniors = ages[ages >= 65]
evens = ages[ages %2 == 0]
odds = ages[ages %2 !=0]
#print(adults)

adults = np.where(ages >= 18 , ages, 0)
print(adults)
import numpy as np

#rng = np.random.default_rng()

#print(rng.integers(low=1, high=101, size=(3,2)))
#print(np.random.uniform(low =-1, high = 1, size=(3,2)))

#OR

#seed Method
np.random.seed(seed=1)
#print(np.random.uniform(low=-1, high=1, size=(3,2)))


rng = np.random.default_rng()
fruits = np.array(["apple","banana", "coconut", "pineapple"])
fruit = rng.choice(fruits, size = 3)
print(fruit)
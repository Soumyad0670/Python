# NUMPY SPECIAL OPERATIONS

import numpy as np

# numpy array is homogenous, whereas python list is heterogenous

arr1= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]]]) # declaring array 1
arr2= np.array([[[19,20,21],[22,23,24]],[[25,26,27],[28,29,30]]]) # declaring array 2
print("Shape of array 1",arr1.shape) # shape of the array
print("Shape of array 2",arr2.shape)  # shape of the array
print("Shape of concatenated array 1 and array 2",np.concatenate((arr1,arr2), axis=0).shape) # concatenate two arrays
# axis=0 → Operate along rows (apply the function to each column).

# Creating two 3D numpy arrays with compatible shapes

arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr4= np.array([[[19, 20, 21], [22, 23, 24]], [[25, 26, 27], [28, 29, 30]]])

# Printing the shape of the arrays

print("Shape of array 3",arr3.shape) # shape of the array
print("Shape of array 4",arr4.shape)  # shape of the array

# Adding the two arrays

print(arr3 + arr4)

# Concatenating the two arrays along the first axis

print("Shape of concatenated array 3 and array 4",np.concatenate((arr3, arr4), axis=1).shape)  
# axis=1 → Operate along columns (apply the function to each row).

arr5=np.array([[[1,2,3,4,5,6,7,8,4,8768,78,88]]])
print("The array is",arr5)
print("The shape of the array is",arr5.shape)

# Creating a 3x3 array of zeros

arr1 = np.zeros([3, 3]) 
print(arr1)

# Creating a 3x3 array of ones

arr2 = np.ones([3, 3])
print(arr2)

arr3=np.full([3,3],'Soumya')
print(arr3)

arr4=np.arange(1,101,4)
print(arr4)


# NUMPY RANDOM OPERATIONS

from numpy import random

arr5=random.randint(50,100,size=6)
print(arr5)

arr6=random.randint(50,100,size=(3,3))
print(arr6)

arr7=random.rand(1)
print(arr7)

arr8=random.rand(4,3,4)
print(arr8)

arr9=random.uniform(10,20,size=7)
print(arr9)

arr10=random.choice([1,4,5,3,24,54,3],size=(4,4))
print(arr10)

arr11=np.array([1,2,3,4,5,7])**4
print(arr11)
print(len(arr11))

arr12=np.sum([1,2,3,4,5,7])
print(arr12)

arr13=np.median([1,2,3,4,5,7])
print(arr13)

arr14=np.median([1,2,3,4,5,7])
print(arr14)

arr15=np.std([1,2,3,4,5,7])
print(arr15)

arr16=np.var([1,2,3,4,5,7])
print(arr16)


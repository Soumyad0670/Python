
#Matplotlib is a comprehensive part of python programming

import matplotlib.pyplot as plt
import numpy as n

x = [1, 2, 3, 4, 5, 9, 0, 87]
y = [2, 3, 5, 7, 11, 9, 7, 6]
plt.plot(x, y,color='b',marker='o',mec='k',mfc='w',ms=5,mew=1,ls='-',label='Ratings')
plt.grid()
plt.xticks(n.arange(10,50,step=9),rotation=50)
plt.yticks(n.arange(10,50,step=9),rotation=50)
plt.figure(figsize=(8,9))
plt.grid(color='c')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('line plot',fontsize=10, color='pink')
plt.legend()
# Creating x axis with range and y axis with Sine
# Function for Plotting Sine Graph
x = n.arange(0, 5*n.pi, 0.1)
y = n.sin(x)
# Plotting Sine Graph
plt.plot(x, y, color='g',marker='o',mec='k',mfc='w',ms=1,mew=1)
plt.show()

#Line Plot
x=[1,2,3,4,5]

y1=[6,7,8,9,10]
y2=[12,13,43,56,45]
y3=[14,13,21,12,45]
y4=[21,21,56,89,76]

plt.figure(figsize=(5,10))
plt.plot(x, y1, color='r', marker='o', mec='k', mfc='w', label='royal Enfield', lw=1)
plt.plot(x, y2, color='b', marker='o', mec='k', mfc='w', label='Himalyan', lw=1)
plt.plot(x, y3, color='pink', marker='o', mec='k', mfc='w', label='Harley', lw=1)
plt.plot(x, y4, color='k', marker='o', mec='k', mfc='w', label='Honda', lw=1)
plt.title('Bike details using line plot')
plt.xlabel('number of days')
plt.ylabel('Distance covered in km')
plt.xticks(n.arange(1,6,1))
plt.yticks(n.arange(0,201,20))
plt.legend()
plt.grid()
plt.show()

#Bar Graph
x1=[0.25,1.25,2.25,3.25,4.25]
x2=[0.5,1.5,2.5,3.5,4.5]
x3=[0.75,1.75,2.75,3.75,4.75]
x4=[1,2,3,4,5]

y1=[6,7,8,9,10]
y2=[12,13,43,56,45]
y3=[14,13,21,12,45]
y4=[21,21,56,89,76]

plt.figure(figsize=(5,10))
plt.bar(x1, y1, color='m',  label='royal Enfield', width=0.2)
plt.bar(x2, y2, color='b',  label='Himalyan', width=0.2)
plt.bar(x3, y3, color='pink',  label='Harley', width=0.2)
plt.bar(x4, y4, color='yellow',  label='Honda', width=0.2)
plt.title('Bike details using Bar plot')
plt.xlabel('number of days')
plt.ylabel('Distance covered in km')
plt.xticks(n.arange(1,6,1))
plt.yticks(n.arange(0,201,20))
plt.legend()
plt.show()

#Scatter plot
x=[1,2,3,4,5]

y1=[6,7,8,9,10]
y2=[12,13,43,56,45]
y3=[14,13,21,12,45]
y4=[21,21,56,89,76]

plt.figure(figsize=(5,10))
plt.scatter(x1, y1, color='m', label='royal Enfield')
plt.scatter(x2, y2, color='b', label='Himalyan')
plt.scatter(x3, y3, color='pink', label='Harley')
plt.scatter(x4, y4, color='yellow', label='Honda')
plt.title('Bike details using Scatter plot')
plt.xlabel('number of days')
plt.ylabel('Distance covered in km')
plt.xticks(n.arange(1,6,1))
plt.yticks(n.arange(0,201,20))
plt.legend()
plt.show()

#Pie chart
slices=[7,2,4,6]
labels=['iphone','laptop','watch','headphones']
cols=['c','g','b','r']
plt.pie(slices,labels=labels,colors=cols,autopct='%1.0f%%', startangle= 45, 
        shadow=True,explode=[0.011,0.013,0.051,0.032],radius=1)
plt.show()

#3D plots
fig=plt.figure(figsize=(12,5))
fig.add_subplot(projection='3d')
x=[1,2,3,4,5]

y1=[6,7,8,9,10]
y2=[12,13,43,56,45]
y3=[14,13,21,12,45]
y4=[21,21,56,89,76]

plt.plot(x, y1, color='r', marker='o', mec='k', mfc='w', label='royal Enfield', lw=1)
plt.plot(x, y2, color='b', marker='o', mec='k', mfc='w', label='Himalyan', lw=1)
plt.plot(x, y3, color='pink', marker='o', mec='k', mfc='w', label='Harley', lw=1)
plt.plot(x, y4, color='k', marker='o', mec='k', mfc='w', label='Honda', lw=1)
plt.title('Bike details using line plot')
plt.xlabel('number of days')
plt.ylabel('Distance covered in km')
plt.xticks(n.arange(1,6,1))
plt.yticks(n.arange(0,201,20))
plt.legend()
plt.grid()
plt.show()

# Histogram
hist=[3,5,2,7,8]
plt.figure(figsize=(5,10))
plt.hist(hist, bins=5, color='blue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()


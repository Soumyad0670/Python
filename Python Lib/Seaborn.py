# importing packages
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Suppress Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"
import os

# Suppress Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

# Get the list of available datasets
s1 = sns.get_dataset_names()

# # Load the 'car_crashes' dataset
s2 = sns.load_dataset('car_crashes')

# Load the 'iris' dataset
s3 = sns.load_dataset('iris')

# Print the list of available datasets
print(s1)

# Print the 'car_crashes' dataset
print(s2)

# Print the 'iris' dataset
print(s3)

#Histplot/Histogram

sns.histplot(s3['sepal_length'],color='g',bins=50)
plt.show()

print(s3['sepal_length'].mode())
print(s3['sepal_length'].median())
print(s3['sepal_length'].var())

#Kdeplot

sns.kdeplot(s3,fill='True',color='m')
plt.show()

#Distplot

print(sns.distplot(s3['sepal_length'],color='g',vertical='True'))
plt.show()

print(s3['sepal_length'].mean())
print(s3['sepal_length'].median())
print(s3['sepal_length'].var())
print(sns.distplot(s3['sepal_length'],color='g',bins=30))
plt.show()


#Box plot--->like scatter plot

sns.boxplot(s3,color='g')
plt.show()

print(s3[s3['sepal_width'] < 2.1])
print(s3[s3['sepal_width'] > 4.0])
print(s3[(s3['sepal_width'] < 2.1) | (s3['sepal_width'] > 4.0)].index)

# Drop specific rows
s3.drop(index=[15, 23, 33, 60], axis=0, inplace=True)
print(s3)

print(sns.boxplot(s3['sepal_length'],color='g'))
plt.show()

# a=sns.boxplot(s3['sepal_length'].min())
# print(a)
# b=sns.boxplot(s3['sepal_length'].max())
# print(b)

#Heatmap

numeric_columns = s3.select_dtypes(include=['float64', 'int64'])
print(numeric_columns.corr())
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Numeric Features')
plt.show()

 
 
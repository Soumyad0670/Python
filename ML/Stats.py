import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy_demo import stats

df = pandas.read_csv("c:/Users/KIIT/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/TempState/B6622E4EF1A8D811316FE50FD2975FAF/Housing.csv") 
print(df.info())
print(df.describe(percentiles=[0.2,0.4,0.6,0.8]))
print(df.shape)
print(df.select_dtypes(include=['number']).quantile(0.25))
print(df['price'].quantile(0.25))

# Sampling Techniques
# Simple random Sampling
n=10
sample_df=df.sample(n)
print('Simple random Sampling',sample_df)

# Systematic sampling
n=15
systematic_sample = df.iloc[::n].head(n)
print('Systematic sampling:', systematic_sample)

# Stratified sampling

# Get stratified sample for bedrooms
min_size = df.groupby('bedrooms').size().min()
stratified_sample = df.groupby('bedrooms', group_keys=False).apply(lambda x: x.sample(n=min_size))
print('Stratified sampling:', stratified_sample)

# Cluster sampling         

df['price_cluster'] = pandas.qcut(df['price'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
selected_clusters = np.random.choice(df['price_cluster'].unique(), size=2, replace=False)
cluster_sample = df[df['price_cluster'].isin(selected_clusters)]
print('Cluster sampling:', cluster_sample)

# Outlier detection using IQR

# n.random.seed(0)
data=np.random.normal(loc=0,size=100,scale=1)
len(data), data

Q1=np.percentile(data,25)
Q2=np.percentile(data,75)
IQR=Q2-Q1
print(IQR)

# Calculate upper and lower boundaries for outliers
lower_boundary = Q1 - 1.5 * IQR
upper_boundary = Q2 + 1.5 * IQR

print("Lower boundary for outliers:", lower_boundary)
print("Upper boundary for outliers:", upper_boundary)

# Find outliers
outliers = data[(data < lower_boundary) | (data > upper_boundary)]
print("Number of outliers:", len(outliers))
print("Outliers:", outliers)

# Create a box plot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('Box Plot for Outlier Detection')
plt.ylabel('Values')
plt.show()

# Feature selection using correlation
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
threshold = 0.7

# Find highly correlated features
high_corr_features = np.where(np.abs(correlation_matrix) > threshold)
high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
                      for x, y in zip(*high_corr_features) if x != y]

# Print highly correlated feature pairs
print("\nHighly correlated features (correlation > 0.7):")
for feat1, feat2, corr in high_corr_features:
    print(f"{feat1} - {feat2}: {corr:.2f}")

# Visualize correlation matrix for selected features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[abs(correlation_matrix) > threshold], annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Selected Features)')
plt.show()

# Calculate confidence interval for mean price

# Calculate mean and standard error
mean_price = np.mean(df['price'])
std_error = stats.sem(df['price'])

# Calculate sample size, mean, and critical value
sample_size = len(df['price'])
mean = df['price'].mean()
# Calculate critical value for 95% confidence level
critical_value = stats.t.ppf(0.975, df=sample_size-1)  # 0.975 because it's two-tailed test

print("\nSample Statistics:")
print(f"Sample Size: {sample_size}")
print(f"Mean Price: ${mean:,.2f}")
print(f"Critical Value: {critical_value:.4f}")

# Perform one-sample t-test
# H0: population mean = sample mean
# H1: population mean ≠ sample mean
hypothetical_mean = 500000  # null hypothesis value
t_stat, p_value = stats.ttest_1samp(df['price'], hypothetical_mean)

print("\nOne-Sample T-Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print("Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis")

# Perform two-sample t-test
# Compare prices of houses with 2 bedrooms vs 3 bedrooms
two_bed = df[df['bedrooms'] == 2]['price']
three_bed = df[df['bedrooms'] == 3]['price']
t_stat2, p_value2 = stats.ttest_ind(two_bed, three_bed)

print("\nTwo-Sample T-Test Results:")
print(f"T-statistic: {t_stat2:.4f}")
print(f"P-value: {p_value2:.4f}")
print("Reject null hypothesis" if p_value2 < 0.05 else "Fail to reject null hypothesis")

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='price', data=df[df['bedrooms'].isin([2, 3])])
plt.title('Price Comparison: 2 vs 3 Bedrooms')
plt.show()




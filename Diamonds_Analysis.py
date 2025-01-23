# Part 1 Getting started
#load libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#load csv data
data = pd.read_csv("C:\\Users\\Nachu\\OneDrive - University of Pittsburgh\\ECON_2824\\Homework\\diamonds (cleaned).csv")

#print header, summary stat of data
print(data.head())
print(data.info())
print(data.describe())

# Part 2 Data Manipulation
#convert str variables to number and mutate new variable volume
data[['Length','Width','Height']] = data[['Length','Width','Height']].apply(pd.to_numeric)
data['Volume'] = data['Length']*data['Width']*data['Height']

#drop Fluorescence column
data = data.drop(columns=['Fluorescence'])

#mutate Logprice column
data['LogPrice'] = np.log(data['Price'])

#mutate Expensive column - 1 if price > meanprice and store as int
data['Expensive'] = (data['Price'] > data['Price'].mean()).astype(int)

#define clarity mapping and map clarity column to numbers
clarity_mapping = {
    'IF': 1,
    'VVS1': 2,
    'VVS2': 3,
    'VS1': 4,
    'VS2': 5,
    'SI1': 6,
    'SI2': 7,
    'I1': 8
}

data['Clarity'] = data['Clarity'].map(clarity_mapping)

#mutate new column - standardised carat weight
data['StdCaratWeight'] = (data['Carat Weight'] -
                          data['Carat Weight'].mean()) /data['Carat Weight'].std()

#Part 3 Data Visualisation
#1. Generate a bar chart showing the distribution of the Shape feature. 
#Which diamond shape are the most common? Why do you think that is?
shape_count = data['Shape'].value_counts()
plt.figure(figsize=(8, 8))
shape_count.plot(kind='bar', color='purple', edgecolor='lightgrey')

for index, value in enumerate(shape_count):
    plt.text(index, value + 2, str(value), ha='center', fontsize=12)

plt.title('Common Diamond Shapes', fontsize=16) 
plt.xlabel('Diamond Shape', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.show()

#Round is the most common diamond shape.

#Analysis 

#Find out mean round diamond price 
round_mean_price = data[data['Shape'] == 'Round']['Price'].mean()

#Cut analysis for round  diamonds
round_cut_counts = data[data['Shape'] == 'Round']['Cut'].value_counts()
round_cut_counts['Total'] = round_cut_counts.sum() 
print(round_cut_counts)

#Filter diamond shapes not rated for cut by shape
nan_count = data[(data['Shape'] != 'Round') & 
                 (data['Shape'].notna()) &
                 (data['Cut']  .isna()) ].shape[0]
print(nan_count)  


#Why is round the most common diamond shape? 

# The mean price of round diamonds is $2,859.91 compared to the population mean of $3,529.39.
# The lower price indicates round diamonds are not as rare as other shapes,
# and is more commonly found. 
# Among diamond shapes other than round, 4298 shapes are not rated for cut at all.
# For round diamonds all 888 are rated, 606/888 are rated excellent, 146/888 are ideal 
# and the rest are rated very good. 
#This shows that round diamonds are a better cut compared to other shapes
# and are commonly prefered.   

#2. Create a scatter plot of Carat Weight versus Price. What does this plot tell you?
# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Carat Weight'], data['Price'], color='skyblue', alpha=1, edgecolor='blue')

# Add titles and labels
plt.title('Carat vs Price', fontsize=16)
plt.xlabel('Carat', fontsize=14)
plt.ylabel('Price', fontsize=14)

#Add custom ticks
plt.xticks(ticks=np.arange(0, max(data['Carat Weight']), step=1))

# Add grid
plt.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.show()

# The scatter plot shows that there are other determinants of price other than carat, 
# and carat and price do not have a linear relationship. However, generally, 
# higher carat means higher price, with other factors (like cut, clarity) being constant.
# The plot also shows several outliers, 
# where price could be highly influenced by a factor other than carat weight

#3.Plot a line graph of Depth % versus Table %. What’s the relationship between the two?

#plot line graph
plt.figure(figsize=(15, 6))
plt.plot(data.index, data['Depth %'], label='Depth %', color='pink', linestyle='-', alpha=1)
plt.plot(data.index, data['Table %'], label='Table %', color='lightblue', linestyle='-', alpha=1)

# Add titles and labels
plt.title('Line Graph: Depth % vs Table %', fontsize=16)
plt.xlabel('Index (Row Number)', fontsize=14)
plt.ylabel('%', fontsize=14)

# Add a legend to differentiate lines
plt.legend(fontsize=12)

# Customize the grid
plt.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.show()

# The line graph shows that Depth% vs Table% is similar across observations, with a few outliers. 

#4. Produce a box plot for the LogPrice feature and describe what you see in the output.
plt.figure(figsize=(8, 6))
plt.boxplot(data['LogPrice'], vert=True, patch_artist=True, boxprops=dict(facecolor="red"))
# Add titles and labels
plt.title('Box Plot of LogPrice', fontsize=16)
plt.ylabel('LogPrice', fontsize=14)

# Display the plot
plt.tight_layout()
plt.show()

# The median of logprice is around 8.2. The interquartile range 
# is between 7.2 and 8.4, indicating that most values lie in this range.
# There are some outliers concentrated around the 10.25-10.5 mark.
# The total range is between 7 to 10.6

# Part 4 - 1. Columns with missing values and % 
# Calculate missing values and percentages
missing_data = data.isna().sum()
missing_percentage = (missing_data / data.shape[0]) * 100

# Combine counts and percentages in a DataFrame
missing_data_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing Values': missing_data.values,
    'Percentage': missing_percentage.values
})

print(missing_data_df)

# This step is important because it helps identify any biases that would affect 
# analysis because of missing data patterns. 

#2. Drop rows where Price is missing, if any.
#Price is not missing for any observation
 
#3. Input missing values in Depth % with the column mean.
data['Depth %'] = data['Depth %'].fillna(data['Depth %'].mean())  

#Part 5 - Data Analysis
#1. Calculate and display the correlation matrix for the numerical features.
# Calculate the correlation matrix
numeric_df = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()    
print(correlation_matrix)

#2. Price Summary Statistics
price_summary = data['Price'].describe()
print(price_summary)

#3. Regression of LogPrice on StdCaratWeight, Clarity, and Color
model1 = smf.ols(formula = 'LogPrice ~ StdCaratWeight + Clarity + Color', data=data).fit()

# Print the summary of the regression model
print(model1.summary())

# No, the effect of clarity and color on log price is not accurately reflected.
# Clarity is coded as dummy variables from 1 to 8, and the effect of each color (E to H)
# is reflected, with D reflected in the intercept. Clarity shows a negative effect
# on the logprice which is not accurate - because each clarity variable is coded 1 to 8
# (equal spacing between each clarity observation) which may have lead to the inaccuracy.
# The correct approach would be to treat them as categorical variables.

#4. Present regression in a table
regression_summary1 = pd.DataFrame({
    'Coefficient': model1.params,
    'Standard Error': model1.bse,
    'P-Value': model1.pvalues
})

# Display the formatted table
print(regression_summary1)

# Revised regression presented in a table

# Load data  with Clarity and Color are categorical variables
data['Clarity'] = data['Clarity'].astype('category')
data['Color'] = data['Color'].astype('category')

# Define the formula, automatically generating dummies
formula = 'LogPrice ~ StdCaratWeight + C(Clarity) + C(Color)'

# Fit the regression model
model2 = smf.ols(formula=formula, data=data).fit()

# Display the summary of the regression
print(model2.summary())

regression_summary2 = pd.DataFrame({
    'Coefficient': model2.params,
    'Standard Error': model2.bse,
    'P-Value': model2.pvalues
})

# Display the formatted table
print(regression_summary2)




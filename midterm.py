import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

data = pd.read_csv('./Dataset/avocado.csv')

# Get the basic info of the data
print(data.info())
print(data.head())
print(data.shape)
print("------------------------------------------")

# Basic data cleaning and validation
data = data[data['region'] != 'TotalUS']  # TotalUS is not a region
data = data.drop(["Unnamed: 0", "Date"], axis=1)
print("Check null data: ")
print(data.isnull().sum())

# Outlier Detection and Removal
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Skewness Detection
skewness_before = data[numeric_columns].skew()
print("\nSkewness of each numeric column before the removal of outliers: ")
print(skewness_before)


def outliers_count(df):
    outlier_count_dict = {}

    for column in df:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_thresh = q1 - (1.5 * iqr)
        upper_thresh = q3 + (1.5 * iqr)

        outlier_count = ((df[column] < lower_thresh) | (df[column] > upper_thresh)).sum()
        outlier_count_dict[column] = outlier_count

    return outlier_count_dict


print("\nNumber of outliers in each numeric column: ")
outlier_counts = outliers_count(data[numeric_columns])

df_before = pd.DataFrame(list(outlier_counts.items()), columns=["Column", "No. of outliers"])
df_before.reset_index(drop=True, inplace=True)
print(df_before)


def remove_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_thresh = q1 - (1.5 * iqr)
    upper_thresh = q3 + (1.5 * iqr)
    df_out = df[~((df < lower_thresh) | (df > upper_thresh)).any(axis=1)]
    return df_out


data[numeric_columns] = remove_outliers(data[numeric_columns])

skewness_after = data[numeric_columns].skew()
print("\nSkewness of each numeric column after the removal of outliers: ")
print(skewness_after)
print("------------------------------------------")

# Perform Yeo-Johnson Transformation to investigate the reduction of skewness
pt = PowerTransformer(method='yeo-johnson')
data2 = data.copy()
data2[numeric_columns] = pt.fit_transform(data2[numeric_columns])

skewness_after_transformation = data2[numeric_columns].skew()
print("\nSkewness of each numeric column after the Yeo-Johnson transformation: ")
print(skewness_after_transformation)

'''
********
Performing Yeo-Johnson Transformation can make the data more closely follow a normal distribution. However the 
transformation changes the values in the data so untransformed data would be used for analysis.
********
'''

print(
    "\nPerforming Yeo-Johnson Transformation can make the data more closely follow a normal distribution. However the "
    "transformation changes the values in the data so untransformed data would be used for analysis.")
print("------------------------------------------")
# Analysis
# Q1
print("1)   Find out the average number of Avocados with PLU 4046 sold in each region (20 points)")
average_avocado_4046 = data.groupby('region')['4046'].mean()
print(average_avocado_4046)
print("------------------------------------------")

# Q2
print("2)   Find out the top ten regions organized by total volume arranged highest to lowest (20 points)")
top_ten_regions_by_volume = data.groupby('region')['Total Volume'].sum().sort_values(ascending=False).head(10)
print(top_ten_regions_by_volume)
print("------------------------------------------")

# Q3
print("3)   An average millennial has a rent of $2000. In general they spend 40% of their rent on food and 20% of "
      "that amount is spent on breakfast. Which region is the best area to live for millennials if the millennial "
      "like to have avocado toast breakfast every 1 time out of three (30 points) Assume you are having one avocado "
      "in the breakfast.")

monthly_food_budget = 2000 * 0.4  # 800
monthly_breakfast_budget = monthly_food_budget * 0.2  # 160
avocado_budget_for_each = monthly_breakfast_budget / 30  # 5.33

data['affordable'] = data['AveragePrice'] <= avocado_budget_for_each
affordable_regions = data[data['affordable']]
print("The average price of avocado with in each region:")
print(affordable_regions.groupby('region')['AveragePrice'].mean())
best_region_for_millennials = affordable_regions.groupby('region')['AveragePrice'].mean().idxmin()
print("\nBest region for millennials (with cheapest avocados):", best_region_for_millennials, "with an average price "
                                                                                              "of ",
      min(affordable_regions.groupby('region')['AveragePrice'].mean()))
print("------------------------------------------")

# Q4
'''
Assuming the sales data for each day of the week is evenly distributed throughout the week, the sales for each day 
would be estimated.
'''
print("4)	If you were an avocado seller and your income depended on selling highest amount of avocados which region "
      "would you take your avocado truck to based on the day. So which region would the truck go to on Monday, "
      "Wednesday and Friday? (30 points)")

# Calculate the daily sales assuming sales are evenly distributed throughout the week
data['Daily Volume'] = data['Total Volume'] / 7

# Estimate the sales for Monday, Wednesday, and Friday
data['MWF Volume'] = data['Daily Volume'] * 3

total_volume_mwf = data.groupby('region')['Daily Volume'].sum()

print("\nThe daily sales of avocados in each region: ")
print(total_volume_mwf)
best_region = total_volume_mwf.idxmax()
print("\nThe best region to sell avocados on Monday, Wednesday, and Friday is", best_region,
      "with an daily volume of",
      max(data.groupby('region')['Daily Volume'].sum()))
print("------------------------------------------")

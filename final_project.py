# -*- coding: utf-8 -*-
"""
Final Project — Connecticut Housing Sales Analysis (2022)
Author: Zehra Toprak

Goal: Predict residential home sale prices in Connecticut using
      machine learning models trained on real 2022 sales data.

Data Source: data.ct.gov — Connecticut Real Estate Sales 2022
"""


# SECTION 1 — IMPORTS

# importing libraries for data handling, visualization, and machine learning
# bringing in pandas so I can work with the data as a table (DataFrame)
# numpy is for doing math stuff on arrays
# matplotlib is what I'm using to make all the charts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # turning off warnings so the output stays clean and readable

# importing tools from sklearn for preprocessing, modeling, and evaluation
# LabelEncoder turns text columns into numbers so the model can use them
# train_test_split divides the data into a training set and a test set
# LinearRegression is the basic straight-line model
# PolynomialFeatures adds extra columns (like squared values) to make the model more flexible
# tree is the decision tree model
# mean_squared_error and r2_score are how I measure how well the models did
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score



# SECTION 2 — LOAD DATA

# loading the CT housing dataset from data.ct.gov
df = pd.read_csv("ct_housing_2022.csv")

# print dataset size to confirm it loaded correctly and so I can see how big the dataset is
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")



# SECTION 3 — EXPLORE DATA

# showing the first 5 rows of the table so I can see what the data looks like
print("\n--- First 5 rows ---")
print(df.head())

# checking data types and how many values are in each column
print("\n--- Column Info ---")
df.info()

# showing basic stats like min, max, mean, and standard deviation for every number column
print("\n--- Summary Statistics ---")
print(df.describe())

# calculating what percentage of each column is missing
# only printing the ones that actually have missing values
print("\n--- Missing Values ---")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0])

# seeing what kinds of properties are in the dataset
print("\n--- Property Types ---")
print(df['Property Type'].value_counts())

# count residential sub types
print("\n--- Residential Types ---")
print(df['Residential Type'].value_counts())

# check how many unique towns are in the dataset
print(f"\nNumber of unique towns: {df['Town'].nunique()}")



# SECTION 4 — CLEAN DATA

# dropping columns that are mostly empty or won't help predict price
cols_to_drop = ['Non Use Code', 'Assessor Remarks', 'OPM remarks', 'Location']
df = df.drop(columns=cols_to_drop)
print("\nDropped low-quality columns. Remaining:", df.columns.tolist())

# these columns came in as strings with $ and commas, so converting them to numbers
df['Sale Amount'] = pd.to_numeric(
    df['Sale Amount'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
)
df['Assessed Value'] = pd.to_numeric(
    df['Assessed Value'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
)

# confirm data types are now numeric
print("\nSale Amount dtype:", df['Sale Amount'].dtype)
print("Assessed Value dtype:", df['Assessed Value'].dtype)

# keeping only residential properties (remove commercial/industrial)
df = df[df['Property Type'] == 'Residential'].copy()
df = df.drop(columns=['Property Type'])
print(f"\nFiltered to residential only: {len(df):,} rows")

# removing rows where the sale price is below $10k or above $2 million
# those are extreme values that would throw off the model
df = df[(df['Sale Amount'] >= 10_000) & (df['Sale Amount'] <= 2_000_000)].copy()
print(f"After removing outliers: {len(df):,} rows")

# filling any missing residential type with 'Unknown' instead of dropping rows
df['Residential Type'] = df['Residential Type'].fillna('Unknown')

# sampling 4000 rows so the model runs faster
# random_state=42 makes sure I get the same sample every time
df = df.sample(n=4000, random_state=42).reset_index(drop=True)
print(f"\nFinal dataset shape: {df.shape}")
print(df.head())

# save cleaned dataset to file
df.to_csv("ct_housing_2022_clean.csv", index=False)



# SECTION 5 — EXPLORATORY CHARTS

# Chart 1 — distribution of house prices
# what does the price distribution look like overall?
# I wanted to see what the most common price range was before building any models
# this draws a bar chart where each bar represents a price range
# and the height shows how many homes sold in that range


fig, ax = plt.subplots(figsize=(10, 5))
_, bins, patches = ax.hist(df['Sale Amount'], bins=50, color='r', edgecolor='black', linewidth=0.5)

# calculating the center of each bin and normalizing it to a 0-1 range
# then using that value to pick a color from the RdYlBu colormap for each bar
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)

cm = plt.get_cmap('RdYlBu_r')
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

ax.set_title('Most Connecticut Homes Sold Between 200K and 400K in 2022', fontsize=14)
ax.set_xlabel('Sale Amount ($)')
ax.set_ylabel('Number of Properties')
# formatting the x-axis with dollar signs so it's easier to read
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
fig.savefig('chart1_histogram.png', dpi=150, bbox_inches='tight')
plt.show()




# Chart 2: Box Plot — average price by residential type
# this makes a box plot for each home type so I can compare their price distributions

# I wanted to see if the type of home (condo vs single family etc.) actually
# affects the price, or if they're all about the same

res_types = df['Residential Type'].unique()
# grouping the sale amounts by residential type into a list of arrays
data_by_type = [df[df['Residential Type'] == t]['Sale Amount'].values for t in res_types]


# sorting the home types from lowest to highest median price
# so the chart reads left to right from cheapest to most expensive
medians = [np.median(d) for d in data_by_type]
sorted_idx = np.argsort(medians)
res_types = [res_types[i] for i in sorted_idx]
data_by_type = [data_by_type[i] for i in sorted_idx]

n = len(res_types)

colors = [(0.40, 0.28, 0.27, 1.0)] * n   # setting all boxes to the same brown color

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(data_by_type, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='grey',
                                markersize=4, linestyle='none', alpha=0.5),
                whiskerprops=dict(color='grey', linewidth=1.5),
                capprops=dict(color='grey', linewidth=1.5))

# looping through each box and applying the brown color
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticklabels(res_types, rotation=20, ha='right')
ax.set_title('The Price Gap Between Home Residential Types in Connecticut (2022)', fontsize=14)
ax.set_xlabel('Residential Type')
ax.set_ylabel('Sale Amount ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chart2_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()


# Chart 3: Horizontal Bar - top 10 towns with highest average prices
#  which towns have the most expensive homes?
# grouping by town, calculating the average sale price, sorting it,
# and grabbing the top 10

top_towns = df.groupby('Town')['Sale Amount'].mean().sort_values(ascending=True).tail(10)
n = len(top_towns)

# highlighting the top 3 towns in navy blue and the cheapest one in red
# so the viewer's eye goes straight to the most interesting data points
colors = ['#cde1ec'] * n              # light blue for middle bars
colors[9] = (0.02, 0.12, 0.27, 1.0)  # navy blue — Westport (top)
colors[8] = (0.02, 0.12, 0.27, 1.0)  # navy blue — New Canaan
colors[7] = (0.02, 0.12, 0.27, 1.0)  # navy blue — Weston
colors[0] = (0.69, 0.02, 0.02, 1.0)  # red — Salisbury (bottom)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_towns.index, top_towns.values, color=colors, edgecolor='white')
ax.set_title('The 10 Most Expensive Towns to Buy a Home in Connecticut (2022)', fontsize=14)
ax.set_xlabel('Average Sale Amount ($)')
ax.set_ylabel('Town')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
# adding a dollar label at the end of each bar
for bar, val in zip(bars, top_towns.values):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}', va='center', fontsize=8)
    
    ax.set_xlim(0, 1400000)
    
plt.tight_layout()
plt.savefig('chart3_expensive_towns.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTop 10 most expensive towns:")
print(top_towns.sort_values(ascending=False).apply(lambda x: f"${x:,.0f}"))


# Chart 4: Bar Chart — top 10 towns by number of sales
# counting how many times each town appears in the dataset and taking the top 10
top_sales_towns = df['Town'].value_counts().head(10)

bar_colors = ['#2066a8', '#8ec1da', '#cde1ec', '#ededed', '#f6d6c2', '#d47264', '#ae282c']
n = len(top_sales_towns)
bar_colors_selected = [bar_colors[int(i * (len(bar_colors) - 1) / (n - 1))] for i in range(n)]

plt.figure(figsize=(11, 5))
bars = plt.bar(top_sales_towns.index, top_sales_towns.values, color=bar_colors_selected, edgecolor='white')

# Exact numbers on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1.5,
        f'{int(height):,}',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333'
    )

plt.title('The 10 Towns Where Homes Sold the Most in Connecticut (2022)', fontsize=14)
plt.xlabel('Town')
plt.ylabel('Number of Sales')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart4_most_sales.png', dpi=150, bbox_inches='tight')
plt.show()


# Chart 5: Line Chart: average sale price by month
# how did prices trend across 2022?
# converting the date column to datetime format so I can pull out the month number
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
df['Month'] = df['Date Recorded'].dt.month

# calculating the average sale price for each month (1 through 12)
monthly_avg = df.groupby('Month')['Sale Amount'].mean()
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# creating one color per month 
marker_colors = [plt.cm.RdYlBu(i / 11) for i in range(12)]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly_avg.index, monthly_avg.values, color='steelblue',
        linewidth=2.5, zorder=1)

# plotting each month's dot separately so I can give each one its own color
for x, y, c in zip(monthly_avg.index, monthly_avg.values, marker_colors):
    ax.scatter(x, y, color=c, s=80, zorder=2, edgecolors='white', linewidths=1)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_title("The Monthly Picture of Connecticut's Housing Market in 2022", fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Average Sale Amount ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig('chart5_monthly.png', dpi=150, bbox_inches='tight')
plt.show()


# Chart 6: Pie Chart — share of sales by residential type
# which type of home sells the most?
# counting sales per type and drawing each one as a slice of the pie
type_counts = df['Residential Type'].value_counts()

pie_colors = [
    '#113047',  
    '#739ab9',  
    '#fbf0d8',  
    '#6F4D38',  
    '#b02a29',  
]
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%', # showing the percentage on each slice
    colors=pie_colors[:len(type_counts)],
    startangle=140,
    pctdistance=1.2, # pushing the percentage labels outside the pie
    labeldistance=1.4,  # pushing the type name labels even further out
    wedgeprops=dict(edgecolor='white', linewidth=2)
)

for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
    autotext.set_color('black')

ax.set_title("What Type of Home Do Connecticut Buyers Actually Choose the Most?", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('chart6_residential_type.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSales by type:")
print(type_counts)


# SECTION 6 — PREPARE DATA FOR MACHINE LEARNING

# create a copy of the dataset for modeling
df_model = df.copy()

# encode categorical variables (Town and Residential Type) into numbers
# the Town and Residential Type columns are text, but models only take numbers
# LabelEncoder replaces each unique string with a unique integer
le_town = LabelEncoder()
le_res  = LabelEncoder()

df_model['Town']             = le_town.fit_transform(df_model['Town'])
df_model['Residential Type'] = le_res.fit_transform(df_model['Residential Type'])

# X = features, y = sale price (what I'm trying to predict)
# X holds the three input features the model will learn from
# y holds the sale price — the value the model is trying to predict
X = df_model[['Assessed Value', 'Town', 'Residential Type']].values
y = df_model['Sale Amount'].values

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# splitting X and y into training and test sets
# split data into training (70%) and testing (30%) 
# random_state=42 keeps the split the same every time I run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

print(f"\nTraining rows: {X_train.shape[0]}")
print(f"Testing rows:  {X_test.shape[0]}")





# SECTION 7 — TRAIN AND EVALUATE MODELS

# --- Model 1: Linear Regression ---
# fitting a straight line through the training data to predict sale price
lr = LinearRegression()
lr.fit(X_train, y_train)

# evaluate model performance
# r2_score measures how well predictions match actual values (1.0 = perfect)
# mean_squared_error is the average of all the squared prediction errors
lr_train_r2  = r2_score(y_train, lr.predict(X_train))
lr_train_mse = mean_squared_error(y_train, lr.predict(X_train))
lr_test_r2   = r2_score(y_test, lr.predict(X_test))
lr_test_mse  = mean_squared_error(y_test, lr.predict(X_test))

print("\n=== Linear Regression ===")
print(f"  Train R²:  {lr_train_r2:.4f}")
print(f"  Test  R²:  {lr_test_r2:.4f}")
print(f"  Test  MSE: {lr_test_mse:,.0f}")


# --- Model 2: Polynomial Regression  ---
# PolynomialFeatures creates new columns by squaring each feature and
# multiplying pairs of features together, giving the model more to work with
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

# then fitting a regular linear regression on top of those new polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# evaluate polynomial model
poly_train_r2  = r2_score(y_train, poly_model.predict(X_train_poly))
poly_train_mse = mean_squared_error(y_train, poly_model.predict(X_train_poly))
poly_test_r2   = r2_score(y_test, poly_model.predict(X_test_poly))
poly_test_mse  = mean_squared_error(y_test, poly_model.predict(X_test_poly))

print("\n=== Polynomial Regression (degree=2) ===")
print(f"  Train R²:  {poly_train_r2:.4f}")
print(f"  Test  R²:  {poly_test_r2:.4f}")
print(f"  Test  MSE: {poly_test_mse:,.0f}")


# --- Model 3: Decision Tree Regressor ---
# this model learns by splitting the data into smaller and smaller groups
# based on feature values until it reaches a prediction
# max_depth=5 means the tree can only split 5 times — keeps it from overfitting
dtr = tree.DecisionTreeRegressor(max_depth=5, random_state=42)
dtr.fit(X_train, y_train)

# evaluate decision tree model
dtr_train_r2  = r2_score(y_train, dtr.predict(X_train))
dtr_train_mse = mean_squared_error(y_train, dtr.predict(X_train))
dtr_test_r2   = r2_score(y_test, dtr.predict(X_test))
dtr_test_mse  = mean_squared_error(y_test, dtr.predict(X_test))

print("\n=== Decision Tree Regressor (max_depth=5) ===")
print(f"  Train R²:  {dtr_train_r2:.4f}")
print(f"  Test  R²:  {dtr_test_r2:.4f}")
print(f"  Test  MSE: {dtr_test_mse:,.0f}")



# SECTION 8 — COMPARE ALL MODELS
# compare all models using R² and MSE
# putting all three models' results into one DataFrame so they're easy to compare
results = pd.DataFrame({
    'Model':     ['Linear Regression', 'Polynomial Regression', 'Decision Tree'],
    'Train R²':  [lr_train_r2,  poly_train_r2,  dtr_train_r2],
    'Test  R²':  [lr_test_r2,   poly_test_r2,   dtr_test_r2],
    'Test  MSE': [lr_test_mse,  poly_test_mse,  dtr_test_mse]
})

print("\n--- Model Comparison ---")
print(results.to_string(index=False))


# drawing a pie chart where each slice shows one model's test R²
# so we can see which model explained the most variance in price
pie_colors = ['#750608', '#F2C4CD', '#051F45']  

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    results['Test  R²'],
    labels=results['Model'],
    autopct='%1.1f%%',
    colors=pie_colors,
    startangle=140,
    pctdistance=1.2,
    labeldistance=1.4,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)

for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
    autotext.set_color('#555555')

ax.set_title("Which Model Predicted Connecticut Home Prices the Best?", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('chart7_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# SECTION 9 — PREDICT A SAMPLE HOUSE

# testing the best model on a made up house to see if it gives a realistic price
sample_town     = le_town.transform(['Hartford'])[0]
sample_res_type = le_res.transform(['Single Family'])[0]

# putting the three feature values into a list so the model can take it as input
sample_house = [[150000, sample_town, sample_res_type]]

# running the decision tree on the sample house to get a predicted sale price
predicted_price = dtr.predict(sample_house)[0]

print("\n--- Sample Prediction ---")
print("  Town:             Hartford")
print("  Type:             Single Family")
print("  Assessed Value:   $150,000")

print(f"  Predicted Price:  ${predicted_price:,.2f}")

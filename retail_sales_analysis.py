
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Oasis Infobyte - Data Analytics")
print("Task 1: EDA on Retail Sales Data")

# Load the dataset
df = pd.read_csv("retail_sales_dataset.csv")

# Display first five rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# Duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract month and quarter
df["Month"] = df["Date"].dt.to_period("M").astype(str)
df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

# 1. Monthly Sales Trend
monthly_sales = df.groupby("Month")["Total Amount"].sum()

plt.figure(figsize=(10, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.show()

# 2. Sales by Product Category
category_sales = df.groupby("Product Category")["Total Amount"].sum()

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("category_sales.png")
plt.show()

# 3. Sales by Gender
gender_sales = df.groupby("Gender")["Total Amount"].sum()

plt.figure(figsize=(6, 5))
gender_sales.plot(kind="bar")
plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("gender_sales.png")
plt.show()

# 4. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nCharts created successfully!")
print("\nSales by Product Category:")
print(category_sales.sort_values(ascending=False))

print("\nQuarterly Sales:")
print(df.groupby("Quarter")["Total Amount"].sum())

# Additional Business Analysis

print("\n===== BUSINESS INSIGHTS =====")

# Total revenue
total_revenue = df["Total Amount"].sum()
print("Total Revenue:", total_revenue)

# Total transactions
print("Total Transactions:", df["Transaction ID"].nunique())

# Average transaction amount
print("Average Transaction Amount:", round(df["Total Amount"].mean(), 2))

# Top selling category by revenue
top_category = df.groupby("Product Category")["Total Amount"].sum().idxmax()
print("Top Category by Revenue:", top_category)

# Sales by age group
df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 45, 100],
    labels=["18-25", "26-35", "36-45", "46+"]
)

age_sales = df.groupby("Age Group", observed=False)["Total Amount"].sum()
print("\nSales by Age Group:")
print(age_sales)

# Save age group chart
plt.figure(figsize=(8, 5))
age_sales.plot(kind="bar")
plt.title("Sales by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("age_group_sales.png")
plt.show()

print("\nAdditional analysis completed!")
# Quarterly Sales Trend

df["Date"] = pd.to_datetime(df["Date"])

df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)

quarterly_sales = df.groupby("Quarter")["Total Amount"].sum()

print("\nQuarterly Sales:")
print(quarterly_sales)

plt.figure(figsize=(10, 5))
quarterly_sales.plot(kind="line", marker="o")

plt.title("Quarterly Sales Trend")
plt.xlabel("Quarter")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("quarterly_sales_trend.png")
plt.show()

print("Quarterly Sales Trend chart saved successfully!")

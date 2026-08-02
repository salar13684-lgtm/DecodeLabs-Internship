import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Global Visual & Display Settings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# PART 1: LOAD DATASET (FIXED WINDOWS PATH)

print("=" * 70)
print("PART 1: LOADING DATASET")
print("=" * 70)

# Prefixing with r'' creates a raw string, resolving the backslash escape error
FILE_PATH = (r"C:\Users\HP\Desktop\Decodelabs Internship\Week 2 project\Dataset for Data Analytics P2.xlsx")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found at: {FILE_PATH}\nPlease verify your file path.")

df = pd.read_excel(FILE_PATH)
print("Dataset successfully loaded.")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n--- First 5 Rows ---")
display(df.head())

print("\n--- Last 5 Rows ---")
display(df.tail())

# PART 2: DATA AUDIT

print("\n" + "=" * 70)
print("PART 2: DATA AUDIT")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\nDataset Info:")
df.info()

print("\nNumerical Columns Summary:")
display(df.describe().T)

print("\nCategorical Columns Summary:")
display(df.describe(include='object').T)

# PART 3: DATA CLEANING

print("\n" + "=" * 70)
print("PART 3: DATA CLEANING")
print("=" * 70)

print("\nUnique Values Per Column")

display(df.nunique())

# Missing values check & handling
missing_val = df.isnull().sum()
missing_pct = (missing_val / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_val, 'Percentage (%)': missing_pct})
print("\nMissing Values Summary:")
display(missing_df[missing_df['Missing Count'] > 0])

# Impute missing values for CouponCode
if 'CouponCode' in df.columns:
    df['CouponCode'].fillna('NO_COUPON', inplace=True)

# Duplicate check & drop
dup_count = df.duplicated().sum()
print(f"\nDuplicate Rows Found: {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)

# Whitespace cleaning for object columns
obj_cols = df.select_dtypes(include=['object']).columns
for col in obj_cols:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Datetime parsing
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print("Data cleaning completed successfully.")

# PART 4: DESCRIPTIVE STATISTICS

print("\n" + "=" * 70)
print("PART 4: DESCRIPTIVE STATISTICS")
print("=" * 70)

num_cols = df.select_dtypes(include=[np.number]).columns

def compute_statistics(dataframe, columns):
    stats = []
    for col in columns:
        data = dataframe[col].dropna()
        stats.append({
            'Feature': col,
            'Count': data.count(),
            'Mean': round(data.mean(), 2),
            'Median': round(data.median(), 2),
            'Mode': data.mode()[0],
            'Min': data.min(),
            'Max': data.max(),
            'Range': data.max() - data.min(),
            'Variance': round(data.var(), 2),
            'Std Dev': round(data.std(), 2),
            'Q1': round(data.quantile(0.25), 2),
            'Q3': round(data.quantile(0.75), 2),
            'IQR': round(data.quantile(0.75) - data.quantile(0.25), 2),
            'Skewness': round(data.skew(), 2),
            'Kurtosis': round(data.kurt(), 2)
        })
    return pd.DataFrame(stats).set_index('Feature')

desc_stats = compute_statistics(df, num_cols)
display(desc_stats)

# PART 5: UNIVARIATE ANALYSIS

print("\n" + "=" * 70)
print("PART 5: UNIVARIATE ANALYSIS PLOTS")
print("=" * 70)

for col in num_cols:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Histogram: {col}')
    
    sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot: {col}')
    
    sns.violinplot(x=df[col], ax=axes[2], color='salmon')
    axes[2].set_title(f'Violin Plot: {col}')
    
    plt.tight_layout()
    plt.show()

# PART 6: CATEGORICAL ANALYSIS

print("\n" + "=" * 70)
print("PART 6: CATEGORICAL ANALYSIS")
print("=" * 70)

cat_cols = ['Product', 'PaymentMethod', 'OrderStatus', 'ReferralSource', 'CouponCode']

for col in cat_cols:
    if col in df.columns:
        val_counts = df[col].value_counts()
        val_pct = (df[col].value_counts(normalize=True) * 100).round(2)
        cat_summary = pd.DataFrame({'Count': val_counts, 'Percentage (%)': val_pct})
        
        print(f"\nCategorical Breakdown for '{col}':")
        display(cat_summary)
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=val_counts.values, y=val_counts.index, palette='viridis')
        plt.title(f'Distribution: {col}')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.show()

# PART 7: NUMERICAL ANALYSIS

print("\n" + "=" * 70)
print("PART 7: NUMERICAL ANALYSIS")
print("=" * 70)

for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
    if col in df.columns:
        print(f"\n--- {col} Metrics ---")
        print(f"Highest Value: {df[col].max()}")
        print(f"Lowest Value:  {df[col].min()}")
        print(f"Average Value: {df[col].mean():.2f}")
        print(f"Median Value:  {df[col].median():.2f}")


# PART 8: OUTLIER DETECTION

print("\n" + "=" * 70)
print("PART 8: OUTLIER DETECTION (IQR METHOD)")
print("=" * 70)

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers detected in '{col}': {len(outliers)}")
    if not outliers.empty:
        display(outliers[['OrderID', 'Product', col]].head())


# PART 9: CORRELATION ANALYSIS

print("\n" + "=" * 70)
print("PART 9: CORRELATION ANALYSIS")
print("=" * 70)

corr_matrix = df[num_cols].corr()
display(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

print("\nPairplot of Numerical Variables")

sns.pairplot(
    df[num_cols],
    corner=True
)

plt.show()

# PART 10: TIME SERIES ANALYSIS

print("\n" + "=" * 70)
print("PART 10: TIME SERIES ANALYSIS")
print("=" * 70)

if 'Date' in df.columns:
    monthly_sales = df.set_index('Date').resample('ME')['TotalPrice'].sum()
    monthly_orders = df.set_index('Date').resample('ME')['OrderID'].count()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(monthly_sales.index, monthly_sales.values, color='tab:blue', marker='o', label='Revenue ($)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Revenue ($)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(monthly_orders.index, monthly_orders.values, color='tab:red', marker='s', linestyle='--', label='Orders')
    ax2.set_ylabel('Total Order Volume', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Monthly Sales Revenue & Order Volume Trends')
    plt.tight_layout()
    plt.show()

# PART 11: BUSINESS QUESTIONS

print("\n" + "=" * 70)
print("PART 11: BUSINESS QUESTIONS ANSWERED")
print("=" * 70)

print(f"1. Highest Revenue Product:           {df.groupby('Product')['TotalPrice'].sum().idxmax()}")
print(f"2. Most Popular Payment Method:      {df['PaymentMethod'].value_counts().idxmax()}")
print(f"3. Top Referral Channel:             {df['ReferralSource'].value_counts().idxmax()}")
print(f"4. Average Order Value (AOV):        ${df['TotalPrice'].mean():.2f}")
print(f"5. Highest Single Order Amount:      ${df['TotalPrice'].max():.2f}")
print(f"6. Lowest Single Order Amount:       ${df['TotalPrice'].min():.2f}")
print(f"7. Most Frequently Ordered Product:  {df['Product'].value_counts().idxmax()}")
print(f"8. Most Common Order Status:         {df['OrderStatus'].value_counts().idxmax()}")

# PART 12: VISUALIZATIONS

print("\n" + "=" * 70)
print("PART 12: ADDITIONAL VISUALIZATIONS")
print("=" * 70)

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='UnitPrice', y='TotalPrice', hue='Quantity', palette='viridis', style='Quantity', s=70)
plt.title("Unit Price vs Total Price (Colored by Quantity Purchased)")
plt.xlabel("Unit Price ($)")
plt.ylabel("Total Price ($)")
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(6, 6))
payment_counts = df['PaymentMethod'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title("Payment Method Share")
plt.tight_layout()
plt.show()


# PART 13: DYNAMIC BUSINESS INSIGHTS

print("\n" + "=" * 70)
print("PART 13: EXECUTIVE BUSINESS INSIGHTS")
print("=" * 70)

total_orders = len(df)
average_order_value = df["TotalPrice"].mean()
highest_order = df["TotalPrice"].max()
lowest_order = df["TotalPrice"].min()

top_revenue_product = (
    df.groupby("Product")["TotalPrice"]
      .sum()
      .idxmax()
)

top_revenue = (
    df.groupby("Product")["TotalPrice"]
      .sum()
      .max()
)

most_ordered_product = (
    df["Product"]
      .value_counts()
      .idxmax()
)

most_ordered_count = (
    df["Product"]
      .value_counts()
      .max()
)

top_payment = (
    df["PaymentMethod"]
      .value_counts()
      .idxmax()
)

top_payment_pct = (
    df["PaymentMethod"]
      .value_counts(normalize=True)
      .max()*100
)

top_referral = (
    df["ReferralSource"]
      .value_counts()
      .idxmax()
)

top_referral_count = (
    df["ReferralSource"]
      .value_counts()
      .max()
)

top_coupon = (
    df["CouponCode"]
      .value_counts()
      .idxmax()
)

top_coupon_count = (
    df["CouponCode"]
      .value_counts()
      .max()
)

common_status = (
    df["OrderStatus"]
      .value_counts()
      .idxmax()
)

common_status_count = (
    df["OrderStatus"]
      .value_counts()
      .max()
)

avg_quantity = df["Quantity"].mean()

outlier_count = 0

for col in num_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr

    outlier_count += (
        ((df[col] < lower) | (df[col] > upper))
        .sum()
    )

print(f"1. Total Orders Analysed           : {total_orders}")

print(f"2. Average Order Value             : ${average_order_value:.2f}")

print(f"3. Highest Revenue Product         : {top_revenue_product} (${top_revenue:,.2f})")

print(f"4. Most Frequently Purchased Item  : {most_ordered_product} ({most_ordered_count} orders)")

print(f"5. Most Popular Payment Method     : {top_payment} ({top_payment_pct:.2f}% share)")

print(f"6. Best Referral Source            : {top_referral} ({top_referral_count} customers)")

print(f"7. Most Used Coupon                : {top_coupon} ({top_coupon_count} uses)")

print(f"8. Most Common Order Status        : {common_status} ({common_status_count} orders)")

print(f"9. Average Quantity Purchased      : {avg_quantity:.2f}")

print(f"10. Highest Order Value            : ${highest_order:.2f}")

print(f"11. Lowest Order Value             : ${lowest_order:.2f}")

print(f"12. Total Outliers Detected        : {outlier_count}")

if average_order_value > df["TotalPrice"].median():
    print("13. Average order value is higher than the median, indicating the presence of high-value orders.")

if df["CouponCode"].eq("NO_COUPON").sum() > 0:
    no_coupon_pct = (
        df["CouponCode"]
        .eq("NO_COUPON")
        .mean()*100
    )
    print(f"14. {no_coupon_pct:.2f}% of orders were completed without using any coupon.")

if "Date" in df.columns:

    monthly_sales = (
        df
        .set_index("Date")
        .resample("ME")["TotalPrice"]
        .sum()
    )

    best_month = monthly_sales.idxmax()

    print(
        f"15. Highest monthly revenue was generated in {best_month.strftime('%B %Y')}."
    )
# PART 14: EXPORT CLEANED DATA

print("\n" + "=" * 70)
print("PART 14: EXPORTING CLEANED DATA")
print("=" * 70)

# Export cleaned file to current working directory
OUTPUT_FILE = os.path.join(os.path.dirname(FILE_PATH), 'Cleaned_Data.xlsx')
df.to_excel(OUTPUT_FILE, index=False)
print(f"Cleaned dataset successfully saved to: '{OUTPUT_FILE}'")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
# Replace 'your_file.csv' with your actual file path
df = pd.read_csv('your_file.csv')

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\n")

# Display first few rows
print("First 5 rows:")
print(df.head())
print("\n")

# Data types and missing values
print("="*50)
print("DATA TYPES AND MISSING VALUES")
print("="*50)
print(df.info())
print("\n")

# Missing values summary
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Missing values per column:")
    print(missing[missing > 0].sort_values(ascending=False))
else:
    print("No missing values found!")
print("\n")

# Target variable distribution
print("="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)
print("Target distribution:")
print(df['Target'].value_counts())
print("\nTarget percentage:")
print(df['Target'].value_counts(normalize=True) * 100)
print("\n")

# Visualize target distribution
plt.figure(figsize=(10, 6))
df['Target'].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Distribution of Target Variable', fontsize=16, fontweight='bold')
plt.xlabel('Target Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: target_distribution.png\n")

# Basic statistics for numerical columns
print("="*50)
print("NUMERICAL FEATURES STATISTICS")
print("="*50)
print(df.describe())
print("\n")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove 'Target' from the lists if it's there
if 'Target' in numerical_cols:
    numerical_cols.remove('Target')
if 'Target' in categorical_cols:
    categorical_cols.remove('Target')

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:10]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print("\n")

# Age at enrollment analysis
if 'Age at enrollment' in df.columns:
    print("="*50)
    print("AGE ANALYSIS")
    print("="*50)
    print(f"Age statistics:")
    print(df['Age at enrollment'].describe())
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df['Age at enrollment'].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Age Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Age at Enrollment')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column='Age at enrollment', by='Target', figsize=(8, 6))
    plt.title('Age by Target Category', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Target')
    plt.ylabel('Age at Enrollment')
    
    plt.tight_layout()
    plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: age_analysis.png\n")

# Gender analysis
if 'Gender' in df.columns:
    print("="*50)
    print("GENDER ANALYSIS")
    print("="*50)
    print("Gender distribution:")
    print(df['Gender'].value_counts())
    
    # Cross-tabulation of Gender and Target
    gender_target = pd.crosstab(df['Gender'], df['Target'], normalize='index') * 100
    print("\nDropout rate by Gender (%):")
    print(gender_target)
    print("\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df['Gender'].value_counts().plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Count')
    
    gender_target.plot(kind='bar', ax=axes[1], stacked=False)
    axes[1].set_title('Target Distribution by Gender (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(title='Target')
    
    plt.tight_layout()
    plt.savefig('gender_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: gender_analysis.png\n")

# Scholarship holder analysis
if 'Scholarship holder' in df.columns:
    print("="*50)
    print("SCHOLARSHIP ANALYSIS")
    print("="*50)
    scholarship_target = pd.crosstab(df['Scholarship holder'], df['Target'], normalize='index') * 100
    print("Dropout rate by Scholarship status (%):")
    print(scholarship_target)
    print("\n")

# Tuition fees analysis
if 'Tuition fees up to date' in df.columns:
    print("="*50)
    print("TUITION FEES ANALYSIS")
    print("="*50)
    tuition_target = pd.crosstab(df['Tuition fees up to date'], df['Target'], normalize='index') * 100
    print("Dropout rate by Tuition payment status (%):")
    print(tuition_target)
    print("\n")

# Debtor analysis
if 'Debtor' in df.columns:
    print("="*50)
    print("DEBTOR ANALYSIS")
    print("="*50)
    debtor_target = pd.crosstab(df['Debtor'], df['Target'], normalize='index') * 100
    print("Dropout rate by Debtor status (%):")
    print(debtor_target)
    print("\n")

# Curricular units analysis (first semester example)
curricular_cols = [col for col in df.columns if 'Curricular units' in col]
if curricular_cols:
    print("="*50)
    print("ACADEMIC PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Found {len(curricular_cols)} curricular unit columns")
    
    # Focus on 1st semester grades if available
    first_sem_grade_col = [col for col in curricular_cols if '1st sem' in col and 'grade' in col.lower()]
    if first_sem_grade_col:
        col_name = first_sem_grade_col[0]
        print(f"\nAnalyzing: {col_name}")
        print(df.groupby('Target')[col_name].describe())
        
        plt.figure(figsize=(10, 6))
        df.boxplot(column=col_name, by='Target')
        plt.title(f'{col_name} by Target', fontsize=14, fontweight='bold')
        plt.suptitle('')
        plt.xlabel('Target')
        plt.ylabel('Grade')
        plt.tight_layout()
        plt.savefig('academic_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: academic_performance.png\n")

# Correlation analysis for numerical features
print("="*50)
print("CORRELATION ANALYSIS")
print("="*50)

# Select a subset of important numerical columns for correlation
important_numerical = [col for col in numerical_cols if any(keyword in col.lower() 
                      for keyword in ['age', 'grade', 'approved', 'credited', 'gdp', 'inflation', 'unemployment'])][:15]

if len(important_numerical) > 0:
    correlation_data = df[important_numerical].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Key Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: correlation_matrix.png\n")

print("="*50)
print("EDA COMPLETE!")
print("="*50)
print("\nGenerated files:")
print("1. target_distribution.png")
print("2. age_analysis.png")
print("3. gender_analysis.png")
print("4. academic_performance.png")
print("5. correlation_matrix.png")
print("\nNext steps:")
print("- Review the visualizations and statistics")
print("- Identify important features for your model")
print("- Consider feature engineering (e.g., creating performance trends)")
print("- Handle class imbalance if needed")
print("- Prepare for model building")
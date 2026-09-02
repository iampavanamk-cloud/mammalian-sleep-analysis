import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Dataset
df = pd.read_csv('dataset_2191_sleep.csv')

print("--- RAW DATA LOADED ---")
print(df.head())

# 2. Data Cleaning: Replace '?' with NaN and convert columns to numbers
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('?', ''), errors='coerce')

# Check for missing values after cleaning
print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())

# 3. Calculate Key Statistics
print("\n--- KEY METRICS ---")
print(f"Average Body Weight: {df['body_weight'].mean():.2f}")
print(f"Lifespan -> Avg: {df['max_life_span'].mean():.1f} | Min: {df['max_life_span'].min()} | Max: {df['max_life_span'].max()}")
print(f"Total Sleep -> Avg: {df['total_sleep'].mean():.1f} hrs | Min: {df['total_sleep'].min()} hrs | Max: {df['total_sleep'].max()} hrs")

# 4. Create Visualizations
import os
os.makedirs('Outputs', exist_ok=True)

# Chart 1: Scatter Plot (Body Weight vs Total Sleep)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='body_weight', y='total_sleep', hue='danger_index', palette='viridis')
plt.title('Body Weight vs. Total Sleep (Colored by Danger Index)')
plt.xlabel('Body Weight')
plt.ylabel('Total Sleep (Hours)')
plt.xscale('log') 
plt.savefig('Outputs/body_weight_vs_sleep.png', bbox_inches='tight')
plt.close()

print("\nSuccess! Chart saved to Outputs/body_weight_vs_sleep.png")
# Chart 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Mammalian Sleep Factors')
plt.savefig('Outputs/correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\nSuccess! Both charts saved to the Outputs/ folder.")
# Chart 3: Top 10 Sleeping Animals Bar Chart
plt.figure(figsize=(10, 6))
animal_col = df.columns[0]
top_sleepers = df.nlargest(10, 'total_sleep')
sns.barplot(data=top_sleepers, x='total_sleep', y=animal_col, palette='magma')
plt.title('Top 10 Animal Sleep Champions')
plt.xlabel('Total Sleep (Hours)')
plt.ylabel('Animal Species')
plt.savefig('Outputs/top_sleepers.png', bbox_inches='tight')
plt.close()

print("\nSuccess! All 3 charts saved to the Outputs/ folder.")

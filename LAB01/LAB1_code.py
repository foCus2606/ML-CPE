import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#Part1
df = pd.read_csv(r"D:\ML\dog_adoption_master.csv")

print("Shape :", df.shape)
print("Columns :", df.columns.tolist())
print()

print(df.head())
print()

print(df.dtypes)
print()

print(df.describe())
print()

print("Missing Values")
print(df.isnull().sum())
print()

print("Duplicate :", df.duplicated().sum())
print()

print("Class Distribution")
print(df["home_type"].value_counts())
print()
print(df["medical_needs"].value_counts())
print()

# # Part2
plt.figure(figsize=(8,5))
plt.hist(df["adoption_counseling"])
plt.title('Histogram')
plt.xlabel('adoption counseling')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12,10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

#Part3
df_clean = df.copy()
# print('\nMissing Values :')
df_clean['days_to_return'] = df_clean['days_to_return'].fillna(0)
df_clean = df_clean.drop_duplicates()
if 'adoption_id' in df_clean.columns:
    df_clean = df_clean.drop(columns=['adoption_id'])
print("--- Compare Mean and Median Before/After Cleaning ---")
compare_df = pd.DataFrame({
    'Original Mean': df[['age_years', 'weight_kg', 'days_in_shelter']].mean(),
    'Original Median': df[['age_years', 'weight_kg', 'days_in_shelter']].median(),
    'Cleaned Mean': df_clean[['age_years', 'weight_kg', 'days_in_shelter']].mean(),
    'Cleaned Median': df_clean[['age_years', 'weight_kg', 'days_in_shelter']].median()
})
print(compare_df)
print()

#Part4
df_encoded = df_clean.copy()
le = LabelEncoder()
df_encoded['size_encoded'] = le.fit_transform(df_encoded['size'])

categorical_cols = ['breed_group', 'intake_type', 'medical_needs', 'home_type', 'return_reason']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
print("--- Final Shape after Feature Engineering ---")
print(df_encoded.shape)
print(df_encoded.head())

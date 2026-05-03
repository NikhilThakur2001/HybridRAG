# Explore the extracted parquet file to understand its structure and content.

import pandas as pd

df = pd.read_parquet("cfpb_telecom.parquet")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nProduct distribution:\n{df['Product'].value_counts()}")

print(f"\n--- Issue column (top 40) ---")
print(df["Issue"].value_counts().head(40))

print(f"\n--- Sub-issue column (top 30) ---")
print(df["Sub-issue"].value_counts().head(30))

print(f"\n--- Issue + Sub-issue combo (top 20) ---")
df["issue_full"] = df["Issue"].fillna("") + " | " + df["Sub-issue"].fillna("")
print(df["issue_full"].value_counts().head(20))

print(f"\n--- Null counts ---")
print(df[["Issue", "Sub-issue", "Consumer complaint narrative"]].isnull().sum())

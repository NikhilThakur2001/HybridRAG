#Explore the extracted parquet file to understand its structure and content.

import pandas as pd

df = pd.read_parquet("cfpb_telecom.parquet")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nProduct distribution:\n{df['Product'].value_counts()}")
print(f"\nSample narrative:\n{df['Consumer complaint narrative'].iloc[0]}")

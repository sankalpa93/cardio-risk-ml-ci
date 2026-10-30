"""Step 1: create the fixed reference data.

Classroom simulation: the cleaned dataset is split once.
  85% -> historical data (what the model is built on)  -> reference data
  15% -> production pool (pretend this data arrives later from real users)
"""
from common import (FEATURES, HISTORICAL, POOL, REFERENCE, load_clean)

df = load_clean().sample(frac=1.0, random_state=42).reset_index(drop=True)

cut = int(len(df) * 0.85)
historical = df.iloc[:cut]
pool = df.iloc[cut:]

historical.to_csv(HISTORICAL, index=False)
historical[FEATURES].to_csv(REFERENCE, index=False)
pool.to_csv(POOL, index=False)

print("Reference dataset created.")
print("Historical / reference rows :", len(historical))
print("Production pool rows        :", len(pool))
print("File:", REFERENCE)

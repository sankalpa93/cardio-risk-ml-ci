"""Step 2: create a batch of 'recent production data'.

stable -> a random sample of the production pool (same kind of patients)
drift  -> real patients from the pool who are older AND have higher blood
          pressure (a different population than the reference)
Real labels (cardio) come from the dataset, so nothing is invented.
"""
import argparse

import pandas as pd

from common import CURRENT, CURRENT_LABELED, FEATURES, POOL, TARGET

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["stable", "drift"], default="stable")
parser.add_argument("--rows", type=int, default=1000)
args = parser.parse_args()

pool = pd.read_csv(POOL)

if args.mode == "drift":
    subset = pool[(pool["age_years"] >= 58) & (pool["ap_hi"] >= 140)]
    print("Pool rows matching the drift condition:", len(subset))
else:
    subset = pool

n = min(args.rows, len(subset))
batch = subset.sample(n=n, random_state=42).reset_index(drop=True)

batch[FEATURES].to_csv(CURRENT, index=False)                     # no label
batch[FEATURES + [TARGET]].to_csv(CURRENT_LABELED, index=False)  # with label

print("Mode:", args.mode)
print("Rows:", len(batch))
print("File:", CURRENT)

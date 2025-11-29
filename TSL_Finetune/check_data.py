import os
import numpy as np

DATA_FOLDER = 'landmarks_json_tsl'

print(f"Checking {DATA_FOLDER} for NaN values...")
bad_files = []

for root, dirs, files in os.walk(DATA_FOLDER):
    for filename in files:
        if filename.endswith('.npy'):
            path = os.path.join(root, filename)
            try:
                data = np.load(path)
                if np.isnan(data).any():
                    print(f"❌ Found NaN in: {filename}")
                    bad_files.append(path)
                if np.isinf(data).any():
                    print(f"❌ Found Infinity in: {filename}")
                    bad_files.append(path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if bad_files:
    print(f"\n⚠️ Found {len(bad_files)} bad files. You must delete or fix them.")
else:
    print("\n✅ Data looks clean (No NaNs found).")
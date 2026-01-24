import pandas as pd
import numpy as np
import os

csv_file = './training_data/qtz_train_003s-1599.csv'
output_dir = './data/brilliant_blue'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

print(f"CSV shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Signal length: {df.shape[1] - 1}")

for idx, row in df.iterrows():
    sample_name = row.iloc[0]
    intensities = row.iloc[1:].values
    
    output_file = os.path.join(output_dir, f'{sample_name}.txt')
    
    np.savetxt(output_file, intensities, fmt='%.2f')
    
    if idx % 100 == 0:
        print(f"Processed {idx}/{df.shape[0]} samples...")

print(f"\nConversion complete!")
print(f"Created {df.shape[0]} txt files in {output_dir}/")
print(f"Each file contains {df.shape[1] - 1} intensity values")

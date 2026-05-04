
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

csv_in  = "C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\CAN-MIRGU.csv"
csv_out = "C:\\Users\\whf80\\Desktop\\Car-Dataset\\CAN-MIRGU-main\\enhanced_CAN-MIRGU.csv"
win_size_dt = 50    # Δt 
win_size_id = 100   # ID 

# Load data 
df = pd.read_csv(csv_in)

df['delta_time'] = df['Timestamp'].diff()
df['frequency'] = 1.0 / df['delta_time']

def split_bytes(hex_str):
    if not isinstance(hex_str, str) or len(hex_str) % 2 != 0:
        return ['00'] * 8
    return [hex_str[i:i+2] for i in range(0, min(len(hex_str), 16), 2)] + ['00'] * (8 - len(hex_str) // 2)

bytes_df = df['DATA'].apply(split_bytes).apply(pd.Series)
bytes_df.columns = [f'byte_{i}' for i in range(8)]

for col in bytes_df.columns:
    bytes_df[col] = bytes_df[col].map(lambda x: int(x, 16))

df['data_mean'] = bytes_df.mean(axis=1)
df['data_std']  = bytes_df.std(axis=1)
df['data_max']  = bytes_df.max(axis=1)
df['data_min']  = bytes_df.min(axis=1)

# Entropy 
df['entropy'] = bytes_df.apply(
    lambda row: shannon_entropy(np.bincount(row, minlength=256) / 8.0),
    axis=1
)

df['is_all_zero'] = bytes_df.apply(lambda row: int((row == 0).all()), axis=1)

# Hamming weight
df['hamming_weight'] = bytes_df.apply(
    lambda row: sum(bin(b).count('1') for b in row),
    axis=1
)

# Arbitration_ID 
# df['id_total_count'] = df['Arbitration_ID'].map(df['Arbitration_ID'].value_counts())
id_group = df.groupby('Arbitration_ID')['delta_time']
df['id_mean_period'] = id_group.transform('mean')
df['id_std_period']  = id_group.transform('std')

# Sliding window Δt 
df['rolling_dt_mean'] = df['delta_time'].rolling(win_size_dt, min_periods=1).mean()
df['rolling_dt_std']  = df['delta_time'].rolling(win_size_dt, min_periods=1).std()
# df['rolling_dt_skew'] = df['delta_time'].rolling(win_size_dt, min_periods=1).skew()
# df['rolling_dt_kurt'] = df['delta_time'].rolling(win_size_dt, min_periods=1).kurt()

# Arbitration_ID entropy 
df['id_code'] = pd.Categorical(df['Arbitration_ID']).codes
n_cat = int(df['id_code'].max()) + 1
df['rolling_id_entropy'] = (
    df['id_code']
      .rolling(win_size_id, min_periods=1)
      .apply(lambda arr: shannon_entropy(np.bincount(arr.astype(int), minlength=n_cat) / len(arr)), raw=False)
)

df['prev_id'] = df['Arbitration_ID'].shift()
df['id_switch'] = (df['Arbitration_ID'] != df['prev_id']).astype(int)
df.drop(columns=['prev_id'], inplace=True)

payload_stats = bytes_df.join(df['Arbitration_ID']).groupby('Arbitration_ID').agg({
    **{f'byte_{i}': ['mean','std'] for i in range(8)}
})
payload_stats.columns = [f'{col[0]}_{col[1]}' for col in payload_stats.columns]
df = df.join(payload_stats, on='Arbitration_ID')

# df['id_cumcount'] = df.groupby('Arbitration_ID').cumcount()

df.drop(columns=['id_code'], inplace=True)

df.to_csv(csv_out, index=False)
csv_out

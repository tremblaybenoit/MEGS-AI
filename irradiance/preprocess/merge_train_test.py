import pandas as pd

df1 = pd.read_csv('/mnt/converted_data/matches/matches_256.csv', parse_dates=['eve_dates'])
df2 = pd.read_csv('/mnt/converted_data_1hr/matches/matches_256_limb.csv', parse_dates=['eve_dates'])
df2['aia_stack'] = df2['aia_stack'].str.replace('/mnt/converted_data', '/mnt/converted_data_1hr')
combined = pd.concat([df1, df2], ignore_index=True)

combined.drop_duplicates(subset=['eve_indices'])
combined.to_csv('/mnt/converted_data_1hr/matches/merged_256_limb.csv', index=False)
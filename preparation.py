import pandas as pd
import numpy as np


print("Loading data...")

wd = pd.read_feather('data/Dados_Jan1980_mar2020_interpolado.feather')

print("Processing...")

# Sort by location instead of year
wdc = wd.sort_values(by=['lat', 'lon'])

# Remove the first 2 months from 1980
wdc = wdc[(wdc.year != 1980) | (wdc.month > 2)]
wdc = wdc[(wdc.year != 2020) | (wdc.month <= 2)]

# Reindex to 0..n, n = number of rows
wdc.reset_index(drop=True, inplace=True)

# Average every 3 elements
wdc = wdc.groupby(wdc.index // 3).mean()

# Select data from each season and the next
# Autumn (outono) = month 4 ( (3+4+5)/3 )
# Winter (inverno) = month 7 ( (6+7+8)/3 )
# Spring (primavera) = month 10 ( (9+10+11)/3 )
# Summer (verão) = month 5 ( (12+1+2)/3 )

outono = wdc[(wdc.month == 4.0)].copy()
outono['winter'] = wdc[(wdc.month == 7.0)].loc[:, 'prgpcp'].values

inverno = wdc[(wdc.month == 7.0)].copy()
inverno['spring'] = wdc[(wdc.month == 10.0)].loc[:, 'prgpcp'].values

primavera = wdc[(wdc.month == 10.0)].copy()
primavera['summer'] = wdc[(wdc.month == 5.0)].loc[:, 'prgpcp'].values

# Because data starts in autumn, summer would need to include autumn from
# the next year, not the current. "roll" fixes this by rolling data from autumn
# to above.
verao = wdc[(wdc.month == 5.0)].copy()
verao['autumn'] = np.roll(wdc[(wdc.month == 4.0)].loc[:, 'prgpcp'].values, -1)

print("Writing seasonal data...")

# Write the seasonal sheets
outono.reset_index().to_feather('data/autumn_data.feather')
inverno.reset_index().to_feather('data/winter_data.feather')
primavera.reset_index().to_feather('data/spring_data.feather')
verao.reset_index().to_feather('data/summer_data.feather')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

# File paths
eq_csv = '/Users/abhudaysingh/Documents/KLIV/KLIV_Intern/Graphs/equation_analysis_comprehensive.csv'
metrics_csv = '/Users/abhudaysingh/Downloads/ICASSPmetric-Sheet7_New.csv'

# Load equation analysis
equation_analysis = pd.read_csv(eq_csv)
equation_analysis.columns = equation_analysis.columns.str.strip()

# Load and fix metrics (header starts at second row for real data)
metrics = pd.read_csv(metrics_csv)
metrics_fixed = metrics.iloc[1:].copy()
metrics_fixed.columns = metrics.iloc[0].values
metrics_fixed = metrics_fixed.reset_index(drop=True)

# Filter: FLOPs vs Latency and linear
eq_filt = equation_analysis[
    (equation_analysis['X_Metric'] == 'FLOPs') &
    (equation_analysis['Y_Metric'] == 'Latency') &
    (equation_analysis['Scale'] == 'linear') &
    (equation_analysis['Line_Type'].str.contains('Network Line'))
]

results = []
for idx, row in eq_filt.iterrows():
    # Clean network name
    lt = row['Line_Type']
    name = (lt.replace(' Network Line','')
            .replace(' w/o skip','')
            .replace(' w/o Skip Connection','')
            .replace('-BO','')
            .replace('+ OCR','')
            .strip())
    gid = row['Graph_ID']
    theta = row['Slope']

    # Find params from metrics by robust matching
    match = metrics_fixed[
        metrics_fixed['Network'].str.lower()
          .str.replace('w/o skip connection','')
          .str.replace('w/o skip','')
          .str.replace('-bo','')
          .str.replace('+ ocr','')
          .str.strip()
          .str.contains(name.lower())
    ]
    params = None
    if not match.empty:
        try:
            params = float(match.iloc[0]['# Params (M)'])
        except:
            params = None
    results.append({'Graph_ID': gid, 'Network': name, 'Param': params, 'Theta': theta})

# Create DataFrame
df = pd.DataFrame(results)
df = df[df['Param'].notna()]

# ---------- Original scatter plot with line connecting all points ----------
plt.figure(figsize=(12,8))
unique_networks = df['Network'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_networks)))

# Scatter by network
for i, network in enumerate(unique_networks):
    network_data = df[df['Network'] == network]
    plt.scatter(network_data['Param'], network_data['Theta'], 
                color=colors[i], label=network, s=60, alpha=0.7)

# Sort for global line and connect ALL points (independent of color/network)
df_sorted = df.sort_values(by='Param')
plt.plot(df_sorted['Param'], df_sorted['Theta'], 
         color='black', alpha=0.3, linewidth=2, marker='o', label='All Networks (Connected)')

plt.xlabel('Parameters (Millions)')
plt.ylabel('Theta (Slope)')
plt.title('Parameter vs Theta by Network (FLOPs vs Latency, Linear)')
plt.legend(title='Network', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- Histogram of Theta with fitted Gamma distribution ----------
theta_vals = df['Theta'].astype(float)
plt.figure(figsize=(8,5))
plt.hist(theta_vals, bins=10, density=True, alpha=0.6, label='Theta Histogram')

# Fit and plot gamma
shape, loc, scale = gamma.fit(theta_vals)
x = np.linspace(theta_vals.min(), theta_vals.max(), 100)
plt.plot(x, gamma.pdf(x, shape, loc, scale), 'r-', lw=2, label='Gamma PDF Fit')

plt.xlabel('Theta (Slope)')
plt.ylabel('Density')
plt.title('Histogram of Theta with Fitted Gamma Distribution')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Gamma fit: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")

"""
This script:
1. Loads the ICASSP metric CSV.
2. Automatically categorises networks into Small / Medium / Large
   using the 33rd / 66th percentiles of Theoretical_FLOPS.
3. Generates 18 graphs:
      • 3 metric comparisons
      • each in Linear and Log scale
      • for each of the 3 categories
   Every graph contains:
      – polynomial trendline + black dashed extrapolation  
      – latching of data points to the extended curve  
      – target markers with drop lines to axes
"""

# ---------- Imports ----------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ---------- Parameters ----------
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'

# Metric-pair configs: (x, y, target-values)
CONFIGS = [
    ('Theoretical_FLOPS', 'Actual_Latency',  [25, 10, 1, 0.1]),
    ('Theoretical_FLOPS', 'Activation_Size', [1000, 500, 100, 10]),
    ('Activation_Size',   'Actual_Latency',  [25, 10, 1, 0.1])
]

# ---------- Load / Clean ----------
df = pd.read_csv(CSV_FILE_PATH, header=1)
df.rename(columns={
    '# Params (M)': 'Params',
    'Input Size': 'Input_Channels',
    'Theoretical GFLops': 'Theoretical_FLOPS',
    'PTFLOPS': 'Actual_FLOPS',
    'F.Pass Activation Size (MB)': 'Activation_Size',
    'Practical Latency (ms)': 'Actual_Latency'
}, inplace=True)

# Retain rows with complete key metrics
keep_cols = [
    'Theoretical_FLOPS', 'Activation_Size', 'Actual_Latency',
    'Params', 'Network'
]
df = df[keep_cols].dropna()

# ---------- Colour map ----------
networks = sorted(df['Network'].unique())
network_colour = {
    n: px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)]
    for i, n in enumerate(networks)
}

# ---------- Size categorisation ----------
q33, q66 = df['Theoretical_FLOPS'].quantile([0.33, 0.66])

def size_bucket(f):
    if f <= q33:      return 'Small'
    if f <= q66:      return 'Medium'
    return 'Large'

df['Size_Category'] = df['Theoretical_FLOPS'].apply(size_bucket)

print('Network–size thresholds')
print(f'  Small   ≤ {q33:.2f}')
print(f'  Medium  ≤ {q66:.2f}')
print(f'  Large   > {q66:.2f}\n')

# ---------- Utility ----------
def closest_point(xp, yp, x_curve, y_curve):
    d = np.hypot(x_curve - xp, y_curve - yp)
    i = d.argmin()
    return x_curve[i], y_curve[i]

# ---------- Core plotting ----------
def make_plot(dsub, x_col, y_col, targets, scale='linear'):
    """Return a plotly Figure for the supplied dataframe subset."""
    eps  = 1e-9
    log  = (scale == 'log')
    Xraw = np.log10(dsub[x_col]+eps) if log else dsub[x_col]
    Yraw = np.log10(dsub[y_col]+eps) if log else dsub[y_col]

    # Polynomial fit
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(Xraw.values.reshape(-1, 1), Yraw.values)

    # Extended curve domain
    xmin, xmax = Xraw.min(), Xraw.max()
    span       = xmax - xmin
    x_ext      = np.linspace(xmin - 1.5*span, xmax + 1.5*span, 500)
    y_ext      = model.predict(x_ext.reshape(-1, 1))

    fig = go.Figure()

    # Scatter points
    for net, grp in dsub.groupby('Network'):
        xs = np.log10(grp[x_col]+eps) if log else grp[x_col]
        ys = np.log10(grp[y_col]+eps) if log else grp[y_col]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers', name=net,
            marker={'color': network_colour[net], 'size': 10, 'opacity': 0.8}
        ))
        # Latching
        for xp, yp in zip(xs, ys):
            xc, yc = closest_point(xp, yp, x_ext, y_ext)
            fig.add_shape(type='line', x0=xp, y0=yp, x1=xc, y1=yc,
                          line={'color': 'grey', 'dash': 'dot', 'width': 1})
            fig.add_trace(go.Scatter(
                x=[xc], y=[yc], mode='markers',
                marker={'color': 'red', 'symbol': 'x', 'size': 6},
                showlegend=False
            ))

    # Trend curves
    in_dom = np.linspace(xmin, xmax, 200)
    fig.add_trace(go.Scatter(
        x=in_dom, y=model.predict(in_dom.reshape(-1, 1)),
        mode='lines', name='Trendline',
        line={'color': 'rgba(150,150,255,0.9)', 'width': 4}
    ))
    fig.add_trace(go.Scatter(
        x=x_ext, y=y_ext, mode='lines', name='Extended',
        line={'color': 'black', 'dash': 'dash', 'width': 2}
    ))

    # Targets + droplines
    tgt_vals = [np.log10(t+eps) if log else t for t in targets]
    for t_raw, t_val in zip(targets, tgt_vals):
        idx = np.abs(y_ext - t_val).argmin()
        xt, yt = x_ext[idx], y_ext[idx]
        fig.add_trace(go.Scatter(
            x=[xt], y=[yt], mode='markers',
            marker={'symbol': 'diamond', 'color': 'black', 'size': 10},
            name=f'{t_raw} target'
        ))
        fig.add_shape(type='line', x0=xt, y0=min(y_ext), x1=xt, y1=yt,
                      line={'color': 'black', 'dash': 'dot'})
        fig.add_shape(type='line', x0=min(x_ext), y0=yt, x1=xt, y1=yt,
                      line={'color': 'black', 'dash': 'dot'})

    # Axes limits (tight to extended curve)
    yr_pad = 0.05*(y_ext.max() - y_ext.min())
    fig.update_layout(
        title=f'{y_col} vs {x_col} – {scale.capitalize()} scale',
        xaxis={'title': x_col, 'range': [x_ext.min(), x_ext.max()]},
        yaxis={'title': y_col, 'range': [y_ext.min()-yr_pad, y_ext.max()+yr_pad]},
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02},
        margin={'l': 80, 'r': 350, 't': 120, 'b': 80}
    )
    return fig

# ---------- Generate 18 graphs ----------
categories = ['Small', 'Medium', 'Large']
tot = 0
for cat in categories:
    df_cat = df[df['Size_Category'] == cat]
    for x_col, y_col, tgt in CONFIGS:
        for scale in ('linear', 'log'):
            fig = make_plot(df_cat, x_col, y_col, tgt, scale)
            if not fig.data:      # skip empty
                continue
            fig.update_layout(title_text=f'{cat} Networks – ' + fig.layout.title.text)
            fig.show()
            tot += 1

print(f'\nTotal graphs shown: {tot}')

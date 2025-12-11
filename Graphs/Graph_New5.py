import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# === Load file ===
csv_path = '/Users/abhudaysingh/Documents/KLIV/KLIV_Intern/Graphs/network_benchmark_summary.csv'
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
df['Input_Size'] = df['Input_Size'].str.replace(' ', '')
df['FLOPs_G'] = pd.to_numeric(df['FLOPs_G'], errors='coerce')
df['Latency_ms'] = pd.to_numeric(df['Latency_ms'], errors='coerce')

networks = sorted(df['Network'].unique())
palette = px.colors.qualitative.T10 + px.colors.qualitative.Alphabet
color_map = {n: palette[i % len(palette)] for i, n in enumerate(networks)}

input_sizes = sorted(df['Input_Size'].dropna().unique(),
                     key=lambda s: [int(x) for x in s.replace('x', ' ').split() if x.isdigit()])

for input_size in input_sizes:
    df_in = df[df['Input_Size'] == input_size]

    # Ensure all networks appear (even if no latency for some)
    all_nets = sorted(df['Network'].unique())
    fig = go.Figure()

    # For plotting range
    xs = df_in['FLOPs_G']
    ys = df_in['Latency_ms'][df_in['Latency_ms'].notnull()]
    xpad = 0.22*(np.log10(xs.max()) - np.log10(xs.min()) if len(xs)>1 else 1)
    ypad = 0.22*(np.log10(ys.max()) - np.log10(ys.min()) if len(ys)>1 else 1)
    xrange = [10**(np.log10(xs.min())-xpad), 10**(np.log10(xs.max())+xpad)]
    yrange = [10**(np.log10(ys.min())-ypad), 10**(np.log10(ys.max())+ypad)]

    plotted = set()

    # Proper points: those with both FLOPs and Latency
    for net in all_nets:
        net_data = df_in[df_in['Network'] == net]
        has_valid = not net_data['Latency_ms'].isnull().all()
        # Plot actual point if latency exists, greyed-out if only FLOPs exists
        if has_valid:
            points = net_data[net_data['Latency_ms'].notnull()]
            fig.add_trace(go.Scatter(
                x=points['FLOPs_G'], y=points['Latency_ms'],
                name=net,
                mode="markers+text",
                marker=dict(
                    color=color_map[net], size=16, line=dict(width=2, color='black')
                ),
                text=[net]*len(points),
                textposition="top center",
                legendgroup=net,
                showlegend=True
            ))
            plotted.add(net)
        # Plot ghost if missing latency (but known FLOPs)
        if not has_valid and not net in plotted:
            ghost = net_data.iloc[0]
            fig.add_trace(go.Scatter(
                x=[ghost['FLOPs_G']], y=[yrange[0]],  # "phantom" at bottom
                name=net + " (no latency)",
                mode="markers+text",
                marker=dict(
                    color="rgba(150,150,150,0.4)", size=16, line=dict(width=2, color='grey'),
                    symbol="circle",
                ),
                text=[net],
                textposition="top center",
                legendgroup=net,
                showlegend=True
            ))

    # Trendline only if ≥3 points with both values
    finite = df_in[df_in['Latency_ms'].notnull()]
    if len(finite) >= 3:
        X = np.log10(finite['FLOPs_G'].values.reshape(-1,1))
        y = np.log10(finite['Latency_ms'].values)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        x_fit = np.linspace(X.min()-0.15, X.max()+0.15, 120)
        y_fit = model.predict(poly.transform(x_fit.reshape(-1,1)))
        fig.add_trace(go.Scatter(
            x=10**x_fit, y=10**y_fit,
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="Poly Trendline",
            showlegend=True
        ))

    fig.update_layout(
        title=f'FLOPs vs Practical Latency<br><sup>Input Size {input_size}</sup>',
        xaxis=dict(
            title='Theoretical FLOPs (G, log scale)',
            type='log', range=[np.log10(xrange[0]), np.log10(xrange[1])],
            tickfont=dict(size=13)),
        yaxis=dict(
            title='Practical Latency (ms, log scale)',
            type='log', range=[np.log10(yrange[0]), np.log10(yrange[1])],
            tickfont=dict(size=13)),
        legend=dict(title='Network', x=1.02, y=1),
        template="plotly_white",
        margin=dict(l=75, r=230, t=90, b=80),
        font=dict(size=15)
    )

    fig.show()

print("All input sizes visualized with every network in legend and trendline, matching presentation style.")

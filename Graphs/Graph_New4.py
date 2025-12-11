import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# ---------- Configuration ----------
# Make sure this path is correct for your system
CSV_PATH = '/Users/abhudaysingh/Downloads/ICASSPmetric-Sheet7_New.csv'

# -------- Data Loading & Cleaning --------
def load_and_clean_data(path):
    print("--- Starting Data Loading and Cleaning ---")
    
    # Read the CSV with the correct header: row 2 (index 1) contains the actual column names
    try:
        df = pd.read_csv(path, header=1)
        print(f"Initial DataFrame shape: {df.shape}")
        print(f"Initial columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        return pd.DataFrame()

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    print(f"Columns after stripping whitespace: {df.columns.tolist()}")

    # Explicitly replace 'zsh: killed' with NaN across the entire DataFrame
    df = df.replace('zsh: killed', np.nan)
    print("Replaced 'zsh: killed' with NaN across DataFrame.")

    # Check if 'Network' column exists after stripping
    if 'Network' not in df.columns:
        print("❌ ERROR: 'Network' column not found after stripping whitespace. Please check the CSV structure.")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Forward-fill Network names
    df['Network'] = df['Network'].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(f"Unique networks after ffill: {df['Network'].nunique()} - {df['Network'].dropna().unique().tolist()}")

    # Coalesce Activation Size and Latency
    act_unquantised = pd.to_numeric(df.get('F.Pass Activation Size (MB)', np.nan), errors='coerce')
    act_quantised = pd.to_numeric(df.get('F.Pass Activation Size (MB).1', np.nan), errors='coerce')
    df['Act_Size'] = act_unquantised.fillna(act_quantised)

    lat_unquantised = pd.to_numeric(df.get('Practical Latency (ms)', np.nan), errors='coerce')
    lat_quantised = pd.to_numeric(df.get('Practical Latency', np.nan), errors='coerce')
    df['Latency'] = lat_unquantised.fillna(lat_quantised)

    # Convert Params & FLOPs
    df['Params'] = pd.to_numeric(df.get('# Params (M)', np.nan), errors='coerce')
    df['FLOPs']  = pd.to_numeric(df.get('Theoretical GFLops', np.nan), errors='coerce')

    print(f"\nDataFrame state before final filtering (first 5 rows and relevant cols):")
    print(df[['Network', 'Params', 'FLOPs', 'Act_Size', 'Latency']].head())
    print(f"Number of NaN in 'Params' before drop: {df['Params'].isna().sum()}")


    # Select and keep only the necessary columns (prevents issues with extraneous columns)
    needed_cols = ['Network', 'Params', 'FLOPs', 'Act_Size', 'Latency']
    df = df[[col for col in needed_cols if col in df.columns]]

    # --- CRITICAL FILTERING STEP ---
    # Drop rows where 'Network' or 'Params' are missing/invalid.
    # This is the point where networks might be lost if their 'Params' became NaN.
    before_drop = len(df)
    
    # Check if 'Network' and 'Params' columns exist before dropping.
    if 'Network' not in df.columns or 'Params' not in df.columns:
        print("❌ ERROR: Essential 'Network' or 'Params' columns are missing after intermediate processing. Cannot proceed.")
        return pd.DataFrame()

    df_filtered = df.dropna(subset=['Network', 'Params'])
    df_filtered = df_filtered[df_filtered['Params'] > 0] # Remove rows where Params is 0 or less

    dropped_count = before_drop - len(df_filtered)
    print(f"\nDropped {dropped_count} rows due to missing or invalid 'Network'/'Params' values (including 'zsh: killed' in Params).")
    
    unique_networks_after_filter = df_filtered['Network'].unique()
    print(f"Final count of networks after filtering: {len(unique_networks_after_filter)} - {sorted(list(unique_networks_after_filter))}")

    if df_filtered.empty:
        print("❌ Final DataFrame is empty after cleaning.")
        return pd.DataFrame()

    print("--- Data Loading and Cleaning Complete ---")
    return df_filtered.reset_index(drop=True)

# -------- Categorization --------
def categorize(df):
    def cat_bucket(p):
        if p <= 10: return "Small"
        if p <= 50: return "Medium"
        return "Large"
    df = df.copy()
    df['Category'] = df['Params'].apply(cat_bucket)
    return df

# -------- Plot 18-Graph Analysis --------
def plot_18_graphs(df):
    print("--- Starting Plot Generation ---")
    categories = ["Small", "Medium", "Large"]
    metric_pairs = [("FLOPs","Latency"), ("FLOPs","Act_Size"), ("Act_Size","Latency")]
    scales = ["linear","log"]

    fig = make_subplots(
        rows=3, cols=6,
        subplot_titles=[f"{cat}: {x} vs {y} ({s})"
                        for cat in categories for (x,y) in metric_pairs for s in scales],
        horizontal_spacing=0.03, vertical_spacing=0.07
    )

    all_networks = sorted(df['Network'].unique())
    print(f"Networks identified for legend: {all_networks}")

    # Using more colors to ensure distinct colors for all networks if possible
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.T10 + px.colors.qualitative.Alphabet
    network_colors = {net: palette[i % len(palette)] for i,net in enumerate(all_networks)}

    # Add dummy legend traces for ALL networks identified in the data
    for net_name in all_networks:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=network_colors[net_name]), name=net_name, legendgroup=net_name
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='black', dash='dash'),
        name='Global Fit', legendgroup='Global'
    ))

    # Iterate through categories, metric pairs, and scales to create all 18 subplots
    for i, cat in enumerate(categories, start=1):
        category_df = df[df['Category'] == cat]
        print(f"Processing category: {cat} with {len(category_df)} data points.")
        for j, (xcol, ycol) in enumerate(metric_pairs):
            for k, scale in enumerate(scales):
                row_idx, col_idx = i, j*2 + k + 1
                
                # Check if required columns exist for the current subplot
                if xcol not in category_df.columns or ycol not in category_df.columns:
                    print(f"  Plot ({cat}, {xcol} vs {ycol}, {scale}): Missing '{xcol}' or '{ycol}' column. Skipping.")
                    fig.add_annotation(x=0.5, y=0.5, text=f"Missing '{xcol}' or '{ycol}' data",
                                       xref=f"x{row_idx}{col_idx} domain", yref=f"y{row_idx}{col_idx} domain",
                                       showarrow=False, font=dict(color="red", size=10))
                    fig.update_xaxes(title_text=xcol, row=row_idx, col=col_idx)
                    fig.update_yaxes(title_text=ycol, row=row_idx, col=col_idx)
                    continue

                # Filter data for the current subplot, dropping NaNs for the specific metrics
                plot_data = category_df.dropna(subset=[xcol, ycol])

                # Apply log transformation if necessary
                if scale == 'log':
                    # Filter out non-positive values for log scale
                    plot_data = plot_data[(plot_data[xcol] > 0) & (plot_data[ycol] > 0)]
                    # If no data remains after log filtering, handle it explicitly
                    if plot_data.empty:
                        x_vals, y_vals = pd.Series(), pd.Series() # Empty series
                    else:
                        x_vals, y_vals = np.log10(plot_data[xcol]), np.log10(plot_data[ycol])
                    xlabel, ylabel = f"log₁₀({xcol})", f"log₁₀({ycol})"
                else:
                    x_vals, y_vals = plot_data[xcol], plot_data[ycol]
                    xlabel, ylabel = xcol, ycol

                # Global Linear Regression Fit
                r2_value = np.nan
                if len(plot_data) > 1:
                    try:
                        lr_global = LinearRegression().fit(x_vals.values.reshape(-1,1), y_vals.values)
                        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_fit = lr_global.predict(x_range.reshape(-1,1))
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_fit, mode='lines',
                            line=dict(color='black', dash='dash'),
                            showlegend=False
                        ), row=row_idx, col=col_idx)
                        r2_value = r2_score(y_vals, lr_global.predict(x_vals.values.reshape(-1,1)))
                    except ValueError: # Catch cases where min/max might fail on single point or other issues
                        pass

                # Per-network scatter points and individual fits
                if not plot_data.empty:
                    for net, group in plot_data.groupby('Network'):
                        group_x = np.log10(group[xcol]) if scale == 'log' else group[xcol]
                        group_y = np.log10(group[ycol]) if scale == 'log' else group[ycol]
                        
                        fig.add_trace(go.Scatter(
                            x=group_x, y=group_y, mode='markers',
                            marker=dict(color=network_colors[net], size=6),
                            showlegend=False
                        ), row=row_idx, col=col_idx)

                        if len(group) > 1:
                            try:
                                lr_network = LinearRegression().fit(group_x.values.reshape(-1,1), group_y.values)
                                x_range_network = np.linspace(group_x.min(), group_x.max(), 2)
                                y_fit_network = lr_network.predict(x_range_network.reshape(-1,1))
                                fig.add_trace(go.Scatter(
                                    x=x_range_network, y=y_fit_network, mode='lines',
                                    line=dict(color=network_colors[net]),
                                    showlegend=False
                                ), row=row_idx, col=col_idx)
                            except ValueError:
                                pass # Not enough points for a line fit for this specific network group

                # Add annotation for R2 and number of data points
                annotation_text = f"n={len(plot_data)}"
                if not np.isnan(r2_value):
                    annotation_text = f"R²={r2_value:.2f}  " + annotation_text

                if len(plot_data) == 0:
                    annotation_text = "No data points"
                
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref=f"x{row_idx}{col_idx} domain", yref=f"y{row_idx}{col_idx} domain",
                    text=annotation_text, showarrow=False,
                    bgcolor="rgba(255,255,255,0.7)", font=dict(size=12)
                )
                fig.update_xaxes(title_text=xlabel, row=row_idx, col=col_idx)
                fig.update_yaxes(title_text=ylabel, row=row_idx, col=col_idx)
    
    print("--- Plot Generation Complete ---")
    # Update overall layout
    fig.update_layout(
        height=1600, width=2400,
        title_text="<b>18-Graph Analysis: FLOPs, Activation Size & Latency</b>",
        title_x=0.5, template="plotly_white",
        legend=dict(
            orientation="v", yanchor="top", y=1,
            xanchor="left", x=1.02,
            title="<b>Networks</b>"
        ),
        margin=dict(l=50, r=200, t=100, b=50),
        hovermode="x unified"
    )
    return fig

# -------- Main Execution --------
def main():
    df_cleaned = load_and_clean_data(CSV_PATH)
    if df_cleaned.empty:
        print("Analysis terminated: No valid data could be loaded or processed.")
        return
    
    df_categorized = categorize(df_cleaned)
    print("Generating 18-graph visualization...")
    
    final_figure = plot_18_graphs(df_categorized)
    if final_figure:
        final_figure.show()
        print("Analysis complete and charts rendered.")
    else:
        print("Failed to generate the plot.")

if __name__ == "__main__":
    main()

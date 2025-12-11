import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

# ---------- Configuration ----------
CSV_PATH = '/Users/abhudaysingh/Downloads/ICASSPmetric-Sheet7_New.csv'

# Global list to store all equation results for CSV export
equation_results = []

# -------- Enhanced Equation Analysis Function --------
def analyze_line_equation(x_vals, y_vals, line_type, graph_info, category, x_metric, y_metric, scale):
    """
    Extracts equation, theta, and parameters for a given line and stores for CSV export
    """
    if len(x_vals) < 2:
        print(f"⚠️  Insufficient data for {graph_info} - {line_type}")
        return None
    
    try:
        # Fit linear regression (as used in original code)
        lr = LinearRegression().fit(x_vals.reshape(-1, 1), y_vals)
        
        # Extract parameters
        slope = lr.coef_[0]
        intercept = lr.intercept_
        theta = lr.coef_  # Coefficient vector
        parameters = [intercept, slope]  # Intercept + coefficients
        
        # Calculate R²
        r2 = r2_score(y_vals, lr.predict(x_vals.reshape(-1, 1)))
        
        # Build equation
        equation = f"y = {intercept:.6g} + {slope:.6g}*x"
        
        # Store results for CSV export
        result = {
            'Graph_ID': f"{category}_{x_metric}_vs_{y_metric}_{scale}",
            'Category': category,
            'X_Metric': x_metric,
            'Y_Metric': y_metric,
            'Scale': scale,
            'Line_Type': line_type,
            'Equation': equation,
            'Intercept': intercept,
            'Slope': slope,
            'Theta_Vector': str(theta.tolist()),
            'Parameters_List': str(parameters),
            'R_Squared': r2,
            'Data_Points': len(x_vals),
            'X_Min': float(x_vals.min()),
            'X_Max': float(x_vals.max()),
            'Y_Min': float(y_vals.min()),
            'Y_Max': float(y_vals.max())
        }
        
        equation_results.append(result)
        
        # Print analysis (as before)
        print(f"\n{'='*80}")
        print(f"📊 GRAPH: {graph_info}")
        print(f"📈 LINE TYPE: {line_type}")
        print(f"{'='*80}")
        print(f"🔢 EQUATION: {equation}")
        print(f"🎯 THETA (coefficients): {theta.tolist()}")
        print(f"📋 PARAMETERS (intercept + coefficients): {parameters}")
        print(f"📈 R-squared: {r2:.6f}")
        print(f"📊 Data points: {len(x_vals)}")
        print(f"{'='*80}")
        
        return {'equation': equation, 'theta': theta, 'parameters': parameters, 'r2': r2}
    
    except Exception as e:
        print(f"❌ Error analyzing {graph_info} - {line_type}: {e}")
        return None

# -------- Data Loading & Cleaning (UNCHANGED) --------
def load_and_clean_data(path):
    print("--- Starting Data Loading and Cleaning ---")
    
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

    df.columns = df.columns.str.strip()
    print(f"Columns after stripping whitespace: {df.columns.tolist()}")

    df = df.replace('zsh: killed', np.nan)
    print("Replaced 'zsh: killed' with NaN across DataFrame.")

    if 'Network' not in df.columns:
        print("❌ ERROR: 'Network' column not found after stripping whitespace.")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    df['Network'] = df['Network'].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(f"Unique networks after ffill: {df['Network'].nunique()} - {df['Network'].dropna().unique().tolist()}")

    act_unquantised = pd.to_numeric(df.get('F.Pass Activation Size (MB)', np.nan), errors='coerce')
    act_quantised = pd.to_numeric(df.get('F.Pass Activation Size (MB).1', np.nan), errors='coerce')
    df['Act_Size'] = act_unquantised.fillna(act_quantised)

    lat_unquantised = pd.to_numeric(df.get('Practical Latency (ms)', np.nan), errors='coerce')
    lat_quantised = pd.to_numeric(df.get('Practical Latency', np.nan), errors='coerce')
    df['Latency'] = lat_unquantised.fillna(lat_quantised)

    df['Params'] = pd.to_numeric(df.get('# Params (M)', np.nan), errors='coerce')
    df['FLOPs']  = pd.to_numeric(df.get('Theoretical GFLops', np.nan), errors='coerce')

    print(f"\nDataFrame state before final filtering (first 5 rows and relevant cols):")
    print(df[['Network', 'Params', 'FLOPs', 'Act_Size', 'Latency']].head())
    print(f"Number of NaN in 'Params' before drop: {df['Params'].isna().sum()}")

    needed_cols = ['Network', 'Params', 'FLOPs', 'Act_Size', 'Latency']
    df = df[[col for col in needed_cols if col in df.columns]]

    before_drop = len(df)
    
    if 'Network' not in df.columns or 'Params' not in df.columns:
        print("❌ ERROR: Essential columns missing.")
        return pd.DataFrame()

    df_filtered = df.dropna(subset=['Network', 'Params'])
    df_filtered = df_filtered[df_filtered['Params'] > 0]

    dropped_count = before_drop - len(df_filtered)
    print(f"\nDropped {dropped_count} rows due to missing/invalid 'Network'/'Params' values.")
    
    unique_networks_after_filter = df_filtered['Network'].unique()
    print(f"Final count of networks: {len(unique_networks_after_filter)} - {sorted(list(unique_networks_after_filter))}")

    if df_filtered.empty:
        print("❌ Final DataFrame is empty after cleaning.")
        return pd.DataFrame()

    print("--- Data Loading and Cleaning Complete ---")
    return df_filtered.reset_index(drop=True)

# -------- Categorization (UNCHANGED) --------
def categorize(df):
    def cat_bucket(p):
        if p <= 10: return "Small"
        if p <= 50: return "Medium"
        return "Large"
    df = df.copy()
    df['Category'] = df['Params'].apply(cat_bucket)
    return df

# -------- Export Results to CSV --------
def export_results_to_csv():
    """
    Export all equation analysis results to easy-to-read CSV files
    """
    if not equation_results:
        print("⚠️ No equation results to export!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(equation_results)
    
    # Export comprehensive results
    comprehensive_file = 'equation_analysis_comprehensive.csv'
    results_df.to_csv(comprehensive_file, index=False)
    print(f"📄 Comprehensive results exported to: {comprehensive_file}")
    
    # Export summary results (easier to read)
    summary_columns = ['Graph_ID', 'Category', 'X_Metric', 'Y_Metric', 'Scale', 'Line_Type', 
                      'Equation', 'R_Squared', 'Data_Points']
    summary_df = results_df[summary_columns]
    
    summary_file = 'equation_analysis_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"📄 Summary results exported to: {summary_file}")
    
    # Export by category for easier analysis
    for category in results_df['Category'].unique():
        cat_df = results_df[results_df['Category'] == category]
        cat_file = f'equation_analysis_{category.lower()}.csv'
        cat_df.to_csv(cat_file, index=False)
        print(f"📄 {category} results exported to: {cat_file}")
    
    # Export by line type
    for line_type in results_df['Line_Type'].unique():
        if 'Global' in line_type:
            type_df = results_df[results_df['Line_Type'].str.contains('Global')]
            type_file = 'equation_analysis_global_trendlines.csv'
            type_df.to_csv(type_file, index=False)
            print(f"📄 Global trendlines exported to: {type_file}")
        else:
            type_df = results_df[~results_df['Line_Type'].str.contains('Global')]
            type_file = 'equation_analysis_network_lines.csv'
            type_df.to_csv(type_file, index=False)
            print(f"📄 Network-specific lines exported to: {type_file}")
    
    print(f"\n✅ TOTAL EQUATIONS ANALYZED: {len(results_df)}")
    print(f"📊 Breakdown by Category:")
    print(results_df['Category'].value_counts())
    print(f"📈 Breakdown by Line Type:")
    print(results_df['Line_Type'].value_counts())

# -------- Plot 18-Graph Analysis WITH EQUATION EXTRACTION --------
def plot_18_graphs(df):
    global equation_results
    equation_results = []  # Reset results
    
    print("--- Starting Plot Generation with Equation Analysis ---")
    print("\n🚀 EXTRACTING EQUATIONS FROM ALL 18 GRAPHS")
    print("="*100)
    
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

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.T10 + px.colors.qualitative.Alphabet
    network_colors = {net: palette[i % len(palette)] for i,net in enumerate(all_networks)}

    # Add dummy legend traces (UNCHANGED)
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

    # Iterate through all 18 subplots WITH equation analysis
    for i, cat in enumerate(categories, start=1):
        category_df = df[df['Category'] == cat]
        print(f"Processing category: {cat} with {len(category_df)} data points.")
        
        for j, (xcol, ycol) in enumerate(metric_pairs):
            for k, scale in enumerate(scales):
                row_idx, col_idx = i, j*2 + k + 1
                graph_title = f"{cat}: {xcol} vs {ycol} ({scale})"
                
                if xcol not in category_df.columns or ycol not in category_df.columns:
                    print(f"  Plot ({graph_title}): Missing column. Skipping.")
                    fig.add_annotation(x=0.5, y=0.5, text=f"Missing '{xcol}' or '{ycol}' data",
                                       xref=f"x{row_idx}{col_idx} domain", yref=f"y{row_idx}{col_idx} domain",
                                       showarrow=False, font=dict(color="red", size=10))
                    fig.update_xaxes(title_text=xcol, row=row_idx, col=col_idx)
                    fig.update_yaxes(title_text=ycol, row=row_idx, col=col_idx)
                    continue

                plot_data = category_df.dropna(subset=[xcol, ycol])

                if scale == 'log':
                    plot_data = plot_data[(plot_data[xcol] > 0) & (plot_data[ycol] > 0)]
                    if plot_data.empty:
                        x_vals, y_vals = pd.Series(), pd.Series()
                    else:
                        x_vals, y_vals = np.log10(plot_data[xcol]), np.log10(plot_data[ycol])
                    xlabel, ylabel = f"log₁₀({xcol})", f"log₁₀({ycol})"
                else:
                    x_vals, y_vals = plot_data[xcol], plot_data[ycol]
                    xlabel, ylabel = xcol, ycol

                # *** ANALYZE GLOBAL FIT LINE ***
                r2_value = np.nan
                if len(plot_data) > 1:
                    try:
                        lr_global = LinearRegression().fit(x_vals.values.reshape(-1,1), y_vals.values)
                        x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_fit = lr_global.predict(x_range.reshape(-1,1))
                        
                        # ADD TRENDLINE TO PLOT (UNCHANGED)
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_fit, mode='lines',
                            line=dict(color='black', dash='dash'),
                            showlegend=False
                        ), row=row_idx, col=col_idx)
                        
                        r2_value = r2_score(y_vals, lr_global.predict(x_vals.values.reshape(-1,1)))
                        
                        # *** ANALYZE THE GLOBAL TRENDLINE EQUATION ***
                        analyze_line_equation(
                            x_vals.values, y_vals.values, 
                            "Global Trendline", 
                            graph_title,
                            cat, xcol, ycol, scale
                        )
                        
                    except ValueError:
                        pass

                # *** ANALYZE PER-NETWORK LINES ***
                if not plot_data.empty:
                    for net, group in plot_data.groupby('Network'):
                        group_x = np.log10(group[xcol]) if scale == 'log' else group[xcol]
                        group_y = np.log10(group[ycol]) if scale == 'log' else group[ycol]
                        
                        # ADD SCATTER POINTS (UNCHANGED)
                        fig.add_trace(go.Scatter(
                            x=group_x, y=group_y, mode='markers',
                            marker=dict(color=network_colors[net], size=6),
                            showlegend=False
                        ), row=row_idx, col=col_idx)

                        # INDIVIDUAL NETWORK TRENDLINES
                        if len(group) > 1:
                            try:
                                lr_network = LinearRegression().fit(group_x.values.reshape(-1,1), group_y.values)
                                x_range_network = np.linspace(group_x.min(), group_x.max(), 2)
                                y_fit_network = lr_network.predict(x_range_network.reshape(-1,1))
                                
                                # ADD NETWORK TRENDLINE TO PLOT (UNCHANGED)
                                fig.add_trace(go.Scatter(
                                    x=x_range_network, y=y_fit_network, mode='lines',
                                    line=dict(color=network_colors[net]),
                                    showlegend=False
                                ), row=row_idx, col=col_idx)
                                
                                # *** ANALYZE NETWORK-SPECIFIC TRENDLINE EQUATION ***
                                analyze_line_equation(
                                    group_x.values, group_y.values, 
                                    f"{net} Network Line", 
                                    graph_title,
                                    cat, xcol, ycol, scale
                                )
                                
                            except ValueError:
                                pass

                # ADD R² ANNOTATION (UNCHANGED)
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
    
    # FINAL LAYOUT (UNCHANGED)
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
        
        # Export all results to CSV files
        print("\n" + "="*100)
        print("📄 EXPORTING EQUATION ANALYSIS RESULTS TO CSV FILES")
        print("="*100)
        export_results_to_csv()
        
        print("\n🎯 ANALYSIS COMPLETE!")
        print("📊 All line equations, theta values, and parameters:")
        print("   - Printed above in console")
        print("   - Exported to multiple CSV files for easy analysis")
        print("📈 Interactive plots displayed in browser (unchanged).")
    else:
        print("Failed to generate the plot.")

if __name__ == "__main__":
    main()

"""
Enhanced Large-Scale Latency Prediction Model Validation Script
- Creates much larger, highly visible plots arranged in a neat grid
- Displays equation formulas on each plot for easy reference
- Significantly increased dimensions for better readability
- Comprehensive error analysis with enhanced visualization
- Fixed subplot annotation references
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ---------- Configuration ----------
CSV_FILE_PATH = '/Users/abhudaysingh/Downloads/ICASSP metrics - Sheet7.csv'

# ---------- Data Loading Functions ----------
def load_and_clean_data():
    """Load CSV and clean data for analysis"""
    df = pd.read_csv(CSV_FILE_PATH, header=1)
    
    df.rename(columns={
        '# Params (M)': 'Params',
        'Input Size': 'Input_Channels',
        'Theoretical GFLops': 'Theoretical_FLOPS',
        'PTFLOPS': 'Actual_FLOPS',
        'F.Pass Activation Size (MB)': 'Activation_Size',
        'Practical Latency (ms)': 'Actual_Latency'
    }, inplace=True)
    
    keep_cols = ['Theoretical_FLOPS', 'Activation_Size', 'Actual_Latency', 'Params', 'Network']
    df = df[keep_cols].dropna()
    
    # Remove invalid entries including "zsh: killed"
    def is_valid_number(x):
        try:
            return isinstance(x, (int, float)) and x > 0
        except:
            return False
    
    df = df[df['Actual_Latency'].apply(is_valid_number)]
    df = df[df['Theoretical_FLOPS'].apply(is_valid_number)]
    df = df[df['Activation_Size'].apply(is_valid_number)]
    df = df[df['Params'].apply(is_valid_number)]
    
    return df

def categorize_models(df):
    """Categorize models by parameter count"""
    def param_bucket(params):
        if params < 1:      return 'Tiny'
        elif params <= 10:  return 'Small'
        elif params <= 50:  return 'Medium'
        else:               return 'Large'
    
    df['Param_Category'] = df['Params'].apply(param_bucket)
    return df

def apply_equations(df):
    """Apply all 8 latency prediction equations"""
    predictions = {}
    
    predictions['Eq1_Single_FLOPs'] = 11.31 * df['Theoretical_FLOPS'] - 514
    predictions['Eq2_Single_Activation'] = 2.28 * df['Activation_Size'] + 2532
    predictions['Eq3_Best_Global'] = (8.50 * df['Theoretical_FLOPS'] + 
                                     1.66 * df['Activation_Size'] - 2138)
    predictions['Eq4_Three_Factor'] = (8.42 * df['Theoretical_FLOPS'] + 
                                      1.65 * df['Activation_Size'] + 
                                      0.03 * df['Params'] - 2152)
    predictions['Eq5_Power_Law'] = (24.24 * 
                                   np.power(df['Theoretical_FLOPS'], 0.704) *
                                   np.power(df['Activation_Size'], 0.191) *
                                   np.power(df['Params'], -0.184))
    
    # Step-wise models
    tiny_mask = df['Params'] < 1
    predictions['Eq6_Tiny'] = np.where(tiny_mask,
                                      19.38 * df['Theoretical_FLOPS'] + 
                                      0.12 * df['Activation_Size'] + 1803,
                                      np.nan)
    
    small_mask = (df['Params'] >= 1) & (df['Params'] <= 10)
    predictions['Eq7_Small'] = np.where(small_mask,
                                       -4.78 * df['Theoretical_FLOPS'] + 
                                       2.38 * df['Activation_Size'] - 129,
                                       np.nan)
    
    medium_mask = (df['Params'] > 10) & (df['Params'] <= 50)
    predictions['Eq8_Medium'] = np.where(medium_mask,
                                        9.65 * df['Theoretical_FLOPS'] + 
                                        1.61 * df['Activation_Size'] - 8096,
                                        np.nan)
    
    return predictions

def calculate_metrics(actual, predicted):
    """Calculate performance metrics for predictions"""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'Count': 0}
    
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    actual_nonzero = actual_clean != 0
    
    r2 = r2_score(actual_clean, predicted_clean)
    mae = mean_absolute_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    mape = np.mean(np.abs((actual_clean[actual_nonzero] - predicted_clean[actual_nonzero]) / 
                         actual_clean[actual_nonzero])) * 100 if actual_nonzero.sum() > 0 else np.nan
    
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Count': mask.sum()}

# ---------- ENHANCED EXTRA-LARGE GRID VISUALIZATION WITH EQUATIONS ----------
def create_comprehensive_grid_plot(df, predictions):
    """Create an extra-large, comprehensive grid plot with all 8 equations and formula displays"""
    
    # Equation names, keys, and formulas for display
    equation_data = [
        ('Eq1_Single_FLOPs', 'Eq1: Single FLOPs', 'L = 11.31×FLOPs - 514'),
        ('Eq2_Single_Activation', 'Eq2: Single Activation', 'L = 2.28×Act_MB + 2532'),
        ('Eq3_Best_Global', 'Eq3: Best Global Linear', 'L = 8.50×FLOPs + 1.66×Act_MB - 2138'),
        ('Eq4_Three_Factor', 'Eq4: Three-Factor Linear', 'L = 8.42×FLOPs + 1.65×Act_MB + 0.03×Params - 2152'),
        ('Eq5_Power_Law', 'Eq5: Power-Law', 'L = 24.24×FLOPs^0.704×Act_MB^0.191×Params^-0.184'),
        ('Eq6_Tiny', 'Eq6: Tiny Models', 'L = 19.38×FLOPs + 0.12×Act_MB + 1803'),
        ('Eq7_Small', 'Eq7: Small Models', 'L = -4.78×FLOPs + 2.38×Act_MB - 129'),
        ('Eq8_Medium', 'Eq8: Medium Models', 'L = 9.65×FLOPs + 1.61×Act_MB - 8096')
    ]
    
    # Create 2x4 subplot grid with increased spacing
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[eq_name for _, eq_name, _ in equation_data],
        horizontal_spacing=0.12,  # Increased spacing for equation display
        vertical_spacing=0.18     # Increased spacing for equation display
    )
    
    # Color mapping for networks
    networks = sorted(df['Network'].unique())
    colors = px.colors.qualitative.T10
    network_colors = {net: colors[i % len(colors)] for i, net in enumerate(networks)}
    
    for i, (eq_key, eq_name, eq_formula) in enumerate(equation_data):
        row = i // 4 + 1
        col = i % 4 + 1
        
        # Get valid data points
        actual = df['Actual_Latency'].values
        predicted = predictions[eq_key]
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        
        if mask.sum() == 0:
            continue
            
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        df_clean = df[mask]
        
        # Add scatter points colored by network with larger markers
        for network in df_clean['Network'].unique():
            net_mask = df_clean['Network'] == network
            if net_mask.sum() == 0:
                continue
                
            fig.add_trace(
                go.Scatter(
                    x=actual_clean[net_mask],
                    y=predicted_clean[net_mask],
                    mode='markers',
                    name=network,
                    marker=dict(
                        color=network_colors[network],
                        size=12,  # Increased marker size
                        opacity=0.8,
                        line=dict(width=2, color='white')  # Added border
                    ),
                    showlegend=(i == 0)  # Only show legend for first plot
                ),
                row=row, col=col
            )
        
        # Perfect prediction line with thicker line
        min_val = min(actual_clean.min(), predicted_clean.min())
        max_val = max(actual_clean.max(), predicted_clean.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=4),  # Thicker line
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        # Calculate and display metrics
        metrics = calculate_metrics(actual, predicted)
        
        # Fixed subplot reference format for annotations
        subplot_number = (row - 1) * 4 + col
        
        if subplot_number == 1:
            xref_domain = 'x domain'
            yref_domain = 'y domain'
        else:
            xref_domain = f'x{subplot_number} domain'
            yref_domain = f'y{subplot_number} domain'
        
        # Add equation formula at the top
        fig.add_annotation(
            x=0.5, y=0.95,
            xref=xref_domain, 
            yref=yref_domain,
            text=f"<b>{eq_formula}</b>",
            showarrow=False,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='blue',
            borderwidth=2,
            font=dict(size=12, color='blue'),
            align='center'
        )
        
        # Add performance metrics at the bottom
        fig.add_annotation(
            x=0.02, y=0.05,
            xref=xref_domain, 
            yref=yref_domain,
            text=f"<b>R²={metrics['R2']:.3f} | MAPE={metrics['MAPE']:.1f}% | n={metrics['Count']}</b>",
            showarrow=False,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=2,
            font=dict(size=11, color='black'),
            align='left'
        )
    
    # Update layout for much larger visibility
    fig.update_layout(
        height=1500,  # Even larger height to accommodate equation displays
        width=2400,   # Even larger width to accommodate equation displays
        title_text="<b>Latency Prediction Model Validation: Actual vs Predicted (with Equation Formulas)</b>",
        title_x=0.5,
        title_font_size=24,  # Larger title font
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14)  # Larger legend font
        )
    )
    
    # Update all subplot axes with larger fonts
    for i in range(8):
        row = i // 4 + 1
        col = i % 4 + 1
        fig.update_xaxes(
            title_text="Actual Latency (ms)", 
            title_font=dict(size=16),  # Larger axis title
            tickfont=dict(size=14),    # Larger tick font
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="Predicted Latency (ms)", 
            title_font=dict(size=16),  # Larger axis title
            tickfont=dict(size=14),    # Larger tick font
            row=row, col=col
        )
    
    return fig

# ---------- EXTRA-LARGE INDIVIDUAL PLOTS WITH EQUATIONS ----------
def create_individual_large_plots(df, predictions):
    """Create extra-large individual plots for each equation with prominent equation display"""
    
    equation_info = [
        ('Eq1_Single_FLOPs', 'Equation 1: Single FLOPs', 'L = 11.31 × FLOPs - 514'),
        ('Eq2_Single_Activation', 'Equation 2: Single Activation', 'L = 2.28 × Act_MB + 2532'),
        ('Eq3_Best_Global', 'Equation 3: Best Global Linear', 'L = 8.50 × FLOPs + 1.66 × Act_MB - 2138'),
        ('Eq4_Three_Factor', 'Equation 4: Three-Factor Linear', 'L = 8.42 × FLOPs + 1.65 × Act_MB + 0.03 × Params - 2152'),
        ('Eq5_Power_Law', 'Equation 5: Power-Law', 'L = 24.24 × FLOPs^0.704 × Act_MB^0.191 × Params^-0.184'),
        ('Eq6_Tiny', 'Equation 6: Tiny Models (<1M params)', 'L = 19.38 × FLOPs + 0.12 × Act_MB + 1803'),
        ('Eq7_Small', 'Equation 7: Small Models (1-10M params)', 'L = -4.78 × FLOPs + 2.38 × Act_MB - 129'),
        ('Eq8_Medium', 'Equation 8: Medium Models (10-50M params)', 'L = 9.65 × FLOPs + 1.61 × Act_MB - 8096')
    ]
    
    networks = sorted(df['Network'].unique())
    colors = px.colors.qualitative.T10
    network_colors = {net: colors[i % len(colors)] for i, net in enumerate(networks)}
    
    figures = []
    
    for eq_key, eq_name, eq_formula in equation_info:
        actual = df['Actual_Latency'].values
        predicted = predictions[eq_key]
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        
        if mask.sum() == 0:
            continue
            
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        df_clean = df[mask]
        
        fig = go.Figure()
        
        # Add scatter points by network with larger markers
        for network in df_clean['Network'].unique():
            net_mask = df_clean['Network'] == network
            if net_mask.sum() == 0:
                continue
                
            fig.add_trace(go.Scatter(
                x=actual_clean[net_mask],
                y=predicted_clean[net_mask],
                mode='markers',
                name=network,
                marker=dict(
                    color=network_colors[network],
                    size=14,  # Larger markers
                    opacity=0.8,
                    line=dict(width=2, color='white')  # Added border
                )
            ))
        
        # Perfect prediction line with thicker line
        min_val = min(actual_clean.min(), predicted_clean.min())
        max_val = max(actual_clean.max(), predicted_clean.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=5)  # Much thicker line
        ))
        
        # Calculate metrics
        metrics = calculate_metrics(actual, predicted)
        
        # Update layout for extra-large, clear visualization with equation prominently displayed
        fig.update_layout(
            title=dict(
                text=f"<b>{eq_name}</b><br>"
                     f"<span style='color:blue; font-size:18px;'><b>{eq_formula}</b></span><br>"
                     f"<span style='font-size:14px;'>R² = {metrics['R2']:.3f} | MAPE = {metrics['MAPE']:.1f}% | n = {metrics['Count']}</span>",
                x=0.5,
                font=dict(size=20)  # Larger title font
            ),
            xaxis=dict(
                title='Actual Latency (ms)',
                title_font=dict(size=18),  # Larger axis title
                tickfont=dict(size=16)     # Larger tick font
            ),
            yaxis=dict(
                title='Predicted Latency (ms)',
                title_font=dict(size=18),  # Larger axis title
                tickfont=dict(size=16)     # Larger tick font
            ),
            width=1200,  # Much larger width
            height=900,  # Much larger height
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=14)  # Larger legend font
            ),
            margin=dict(l=100, r=250, t=180, b=100)  # Increased top margin for equation display
        )
        
        figures.append(fig)
    
    return figures

# ---------- Error Analysis Table ----------
def create_error_analysis_table(df, predictions):
    """Create detailed error analysis table with equation formulas"""
    results = []
    
    equation_data = [
        ('Eq1_Single_FLOPs', 'Single FLOPs', 'L = 11.31×FLOPs - 514'),
        ('Eq2_Single_Activation', 'Single Activation', 'L = 2.28×Act_MB + 2532'),
        ('Eq3_Best_Global', 'Best Global Linear', 'L = 8.50×FLOPs + 1.66×Act_MB - 2138'),
        ('Eq4_Three_Factor', 'Three-Factor Linear', 'L = 8.42×FLOPs + 1.65×Act_MB + 0.03×Params - 2152'),
        ('Eq5_Power_Law', 'Power-Law', 'L = 24.24×FLOPs^0.704×Act_MB^0.191×Params^-0.184'),
        ('Eq6_Tiny', 'Tiny Models (<1M)', 'L = 19.38×FLOPs + 0.12×Act_MB + 1803'),
        ('Eq7_Small', 'Small Models (1-10M)', 'L = -4.78×FLOPs + 2.38×Act_MB - 129'),
        ('Eq8_Medium', 'Medium Models (10-50M)', 'L = 9.65×FLOPs + 1.61×Act_MB - 8096')
    ]
    
    for eq_key, eq_name, eq_formula in equation_data:
        metrics = calculate_metrics(df['Actual_Latency'].values, predictions[eq_key])
        
        # Find a representative example
        mask = ~np.isnan(predictions[eq_key])
        if mask.sum() > 0:
            example_idx = mask.argmax()  # First valid prediction
            example_actual = df['Actual_Latency'].iloc[example_idx]
            example_predicted = predictions[eq_key][example_idx]
            example_error = abs((example_predicted - example_actual) / example_actual) * 100
            example_network = df['Network'].iloc[example_idx]
        else:
            example_actual = example_predicted = example_error = np.nan
            example_network = "No valid data"
        
        results.append({
            'Equation': eq_name,
            'Formula': eq_formula,
            'R²': f"{metrics['R2']:.3f}" if not np.isnan(metrics['R2']) else "N/A",
            'MAPE (%)': f"{metrics['MAPE']:.1f}" if not np.isnan(metrics['MAPE']) else "N/A",
            'RMSE': f"{metrics['RMSE']:.1f}" if not np.isnan(metrics['RMSE']) else "N/A",
            'Sample Count': metrics['Count'],
            'Example Network': example_network,
            'Example Actual': f"{example_actual:.1f}" if not np.isnan(example_actual) else "N/A",
            'Example Predicted': f"{example_predicted:.1f}" if not np.isnan(example_predicted) else "N/A",
            'Example Error (%)': f"{example_error:.1f}" if not np.isnan(example_error) else "N/A"
        })
    
    return pd.DataFrame(results)

# ---------- Main Execution ----------
def main():
    print("🔄 Loading and processing data...")
    
    df = load_and_clean_data()
    df = categorize_models(df)
    predictions = apply_equations(df)
    
    print(f"✅ Loaded {len(df)} valid data points")
    print(f"📊 Networks: {', '.join(df['Network'].unique())}")
    print(f"📈 Parameter categories: {df['Param_Category'].value_counts().to_dict()}")
    
    # Option 1: Create extra-large comprehensive grid plot with equations
    print("\n🎯 Creating extra-large comprehensive grid visualization with equation formulas...")
    grid_fig = create_comprehensive_grid_plot(df, predictions)
    grid_fig.show()
    
    # Option 2: Create extra-large individual plots with equations
    print("\n📈 Creating extra-large individual plots with equation formulas...")
    individual_figs = create_individual_large_plots(df, predictions)
    
    for i, fig in enumerate(individual_figs):
        print(f"   Showing plot {i+1}/{len(individual_figs)}")
        fig.show()
    
    # Create and display error analysis table with equations
    print("\n📋 Error Analysis Summary with Equation Formulas:")
    error_table = create_error_analysis_table(df, predictions)
    print(error_table.to_string(index=False))
    
    # Export detailed results
    detailed_results = df.copy()
    for eq_key in predictions:
        detailed_results[f'Predicted_{eq_key}'] = predictions[eq_key]
        detailed_results[f'Error_Abs_{eq_key}'] = abs(predictions[eq_key] - df['Actual_Latency'])
        detailed_results[f'Error_Pct_{eq_key}'] = abs((predictions[eq_key] - df['Actual_Latency']) / df['Actual_Latency']) * 100
    
    detailed_results.to_csv('latency_prediction_validation_detailed_with_equations.csv', index=False)
    print("\n💾 Detailed results exported to 'latency_prediction_validation_detailed_with_equations.csv'")
    
    print("\n✅ All visualizations with equation formulas completed successfully!")
    return df, predictions, grid_fig, individual_figs

if __name__ == "__main__":
    df, predictions, grid_fig, individual_figs = main()

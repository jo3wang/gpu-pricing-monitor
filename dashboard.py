#!/usr/bin/env python3
"""
GPU Pricing Dashboard - Simplified Clean Design
Daily spot and on-demand pricing trends across AWS, Azure, and GCP (US regions only).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Page configuration with dark theme
st.set_page_config(
    page_title="GPU Pricing Monitor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and card styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-change {
        font-size: 16px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 8px;
    }
    .change-up {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    .change-down {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    h1 {
        color: #f1f5f9 !important;
        font-size: 36px !important;
        font-weight: 300 !important;
        margin-bottom: 30px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_pricing_data(data_dir: Path) -> pd.DataFrame:
    """Load pricing data from ALL US regions across all clouds.

    Data is aggregated using MEDIAN prices per accelerator-hour, with outlier removal
    via IQR filtering. Each row represents the median of all instance prices for that
    accelerator model, cloud, region, and date.

    Aggregation: MEDIAN (not mean) - more robust to outliers
    """
    summary_path = data_dir / 'gpu_pricing_summary.csv'

    if not summary_path.exists():
        st.error(f"⚠️ Data file not found: {summary_path}")
        st.info("Run data collection first: `python3 collect_daily_prices.py`")
        return pd.DataFrame()

    df = pd.read_csv(summary_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

    # Filter to US regions only (across all clouds)
    # AWS: us-east-1, us-west-1, us-west-2
    # Azure: eastus, westus, westus2, westus3, centralus, northcentralus, southcentralus
    # GCP: us-central1, us-west1, us-west2, us-west3, us-west4, us-east1, us-east4, us-east5, us-south1
    # Also include "ALL" for global aggregations
    df = df[df['region'].str.contains('us', case=False, na=False) | (df['region'] == 'ALL')]

    return df


def calculate_price_change(df: pd.DataFrame, cloud: str, model: str) -> tuple:
    """Calculate price change percentage for a cloud/model."""
    cloud_data = df[(df['cloud'] == cloud) & (df['accelerator_model'] == model) & (df['region'] == 'ALL')]

    if cloud_data.empty or len(cloud_data) < 2:
        return None, None

    cloud_data = cloud_data.sort_values('date')
    latest = cloud_data.iloc[-1]['median_price_per_accel_hour_ondemand']
    previous = cloud_data.iloc[-2]['median_price_per_accel_hour_ondemand']

    if pd.isna(latest) or pd.isna(previous) or previous == 0:
        return None, None

    change_pct = ((latest - previous) / previous) * 100
    return latest, change_pct


def create_pricing_card(cloud_name: str, price: float, change_pct: float, icon: str):
    """Create a pricing metric card."""
    if price is None:
        price_display = "N/A"
        change_display = ""
    else:
        # Convert to thousands for display
        if price > 1000:
            price_display = f"{price/1000:.1f}K"
        else:
            price_display = f"{price:.0f}"

        if change_pct is not None:
            change_class = "change-up" if change_pct >= 0 else "change-down"
            arrow = "↑" if change_pct >= 0 else "↓"
            change_display = f'<div class="metric-change {change_class}">{abs(change_pct):.1f}% {arrow}</div>'
        else:
            change_display = ""

    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {cloud_name}</div>
        <div class="metric-value">{price_display}</div>
        {change_display}
    </div>
    """
    return card_html


def main():
    """Main dashboard application."""

    # Header
    today = datetime.now()
    st.title(f"GPU pricings today {today.strftime('%b %d')}")

    # Load data
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    df = load_pricing_data(data_dir)

    if df.empty:
        st.stop()

    # Sidebar for model selection
    st.sidebar.header("⚙️ Settings")

    # GPU model selection
    available_models = sorted(df[df['accelerator_type'] == 'GPU']['accelerator_model'].unique())
    selected_model = st.sidebar.selectbox(
        "GPU Model",
        available_models,
        index=available_models.index('H100') if 'H100' in available_models else 0
    )

    # Pricing type selection
    pricing_type = st.sidebar.radio(
        "Pricing Type",
        ["Spot Pricing", "On-Demand Pricing"],
        index=0
    )

    # Date range
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    # Top pricing cards
    col1, col2, col3 = st.columns(3)

    # Calculate current prices and changes for each cloud
    aws_price, aws_change = calculate_price_change(df, 'aws', selected_model)
    azure_price, azure_change = calculate_price_change(df, 'azure', selected_model)
    gcp_price, gcp_change = calculate_price_change(df, 'gcp', selected_model)

    with col1:
        st.markdown(create_pricing_card("AWS", aws_price, aws_change, "☁️"), unsafe_allow_html=True)

    with col2:
        st.markdown(create_pricing_card("Azure", azure_price, azure_change, "👤"), unsafe_allow_html=True)

    with col3:
        st.markdown(create_pricing_card("GCP", gcp_price, gcp_change, "➕"), unsafe_allow_html=True)

    # Main chart section
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter data for chart
    df_chart = df[
        (df['accelerator_model'] == selected_model) &
        (df['region'] == 'ALL')
    ].copy()

    if df_chart.empty:
        st.warning(f"No data available for {selected_model}")
        st.stop()

    # Determine which price column to use
    if pricing_type == "Spot Pricing":
        price_col = 'median_price_per_accel_hour_spot'
        title_prefix = f"{selected_model} spot pricing"
    else:
        price_col = 'median_price_per_accel_hour_ondemand'
        title_prefix = f"{selected_model} on-demand pricing"

    # Remove rows with missing pricing data
    df_chart = df_chart[df_chart[price_col].notna()]

    # Create the main chart
    fig = go.Figure()

    # Color scheme matching the screenshot
    colors = {
        'aws': '#a855f7',      # Purple/magenta
        'azure': '#06b6d4',    # Cyan/blue
        'gcp': '#22c55e'       # Green
    }

    # Add traces for each cloud
    for cloud in df_chart['cloud'].unique():
        cloud_data = df_chart[df_chart['cloud'] == cloud].sort_values('date')

        if not cloud_data.empty:
            # Calculate current price for display
            current_price = cloud_data.iloc[-1][price_col]

            # Calculate change percentage
            if len(cloud_data) >= 2:
                prev_price = cloud_data.iloc[-2][price_col]
                if prev_price > 0:
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                else:
                    change_pct = 0
            else:
                change_pct = 0

            fig.add_trace(go.Scatter(
                x=cloud_data['date'],
                y=cloud_data[price_col],
                mode='lines+markers',
                name=cloud.upper(),
                line=dict(
                    color=colors.get(cloud, '#94a3b8'),
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                marker=dict(
                    size=8,
                    color=colors.get(cloud, '#94a3b8'),
                    line=dict(color='#0e1117', width=2)
                ),
                hovertemplate=(
                    f'<b>{cloud.upper()}</b><br>' +
                    '$%{y:.2f}<br>' +
                    '%{x|%b %d, %Y}<br>' +
                    '<extra></extra>'
                ),
                showlegend=True
            ))

    # Get latest price for title
    latest_data = df_chart.sort_values('date').iloc[-1]
    latest_price = latest_data[price_col]
    if latest_price > 1000:
        price_display = f"${latest_price/1000:.1f}K"
    else:
        price_display = f"${latest_price:.0f}"

    # Calculate overall change
    all_data = df_chart.sort_values('date')
    if len(all_data) >= 2:
        first_price = all_data.iloc[0][price_col]
        last_price = all_data.iloc[-1][price_col]
        overall_change = ((last_price - first_price) / first_price) * 100
    else:
        overall_change = 0

    arrow = "↑" if overall_change >= 0 else "↓"

    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text=f"{title_prefix}<br><sup style='font-size: 32px; font-weight: 700;'>{price_display}</sup> "
                 f"<sup style='font-size: 16px; color: {'#22c55e' if overall_change >= 0 else '#ef4444'};'>"
                 f"{abs(overall_change):.1f}% {arrow}</sup>",
            x=0,
            font=dict(size=20, color='#f1f5f9')
        ),
        xaxis=dict(
            title="",
            gridcolor='#1e293b',
            showgrid=True,
            color='#94a3b8',
            tickformat='%b %d'
        ),
        yaxis=dict(
            title="Price per GPU-hour ($)",
            gridcolor='#1e293b',
            showgrid=True,
            color='#94a3b8',
            tickformat='$,.0f'
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#f1f5f9'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='#334155',
            borderwidth=1
        ),
        height=600,
        margin=dict(t=100, b=60, l=60, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data summary table
    with st.expander("📊 View Data Table"):
        display_df = df_chart[['date', 'cloud', 'accelerator_model', price_col]].sort_values(['cloud', 'date'])
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df[price_col] = display_df[price_col].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            display_df.rename(columns={
                'date': 'Date',
                'cloud': 'Cloud',
                'accelerator_model': 'Model',
                price_col: 'Price/Hour'
            }),
            hide_index=True,
            use_container_width=True
        )


if __name__ == "__main__":
    main()

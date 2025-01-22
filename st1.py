import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, binary_dilation
from shapely.geometry import Point
import os

# Page configuration
st.set_page_config(
    page_title="Mineral Concentration Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual appeal
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox { margin-bottom: 1rem; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Core Analysis Functions
def analyze_element_correlations(df, target_element, top_percent=30):
    """Analyze correlations between target element and all other elements."""
    exclude_cols = ['OBJECTID', 'SAMPLENO', 'LONGITUDE', 'LATITUDE',
                   'geometry', 'TOPOSHEET', 'REGION']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df = df[feature_cols]
    correlations = df.corr()[target_element]
    correlations = correlations.drop(target_element)
    correlations_abs = correlations.abs()
    
    threshold = np.percentile(correlations_abs, 100 - top_percent)
    top_correlations = correlations[correlations_abs >= threshold]
    top_correlations = top_correlations.reindex(
        correlations_abs[correlations_abs >= threshold].sort_values(ascending=False).index
    )
    
    top_element_names = correlations_abs[correlations_abs >= threshold].index.tolist()
    return top_element_names

def prepare_data(df, target='CU', test_size=0.2, random_state=42):
    """Prepare data for model training."""
    exclude_cols = ['OBJECTID', 'SAMPLENO', 'LONGITUDE', 'LATITUDE',
                   'geometry', 'TOPOSHEET', 'REGION']
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target]]
    
    X = df[feature_cols]
    y = df[target]
    lon = df['LONGITUDE']
    lat = df['LATITUDE']
    
    X_train, X_test, y_train, y_test, lon_train, lon_test, lat_train, lat_test = train_test_split(
        X, y, lon, lat, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, lon_test, lat_test, scaler, feature_cols

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    rf_model = RandomForestRegressor(
        n_estimators=3000,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_range=300):
    """Evaluate model performance."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, train_pred)
    
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)
    
    train_mse_pct = (train_mse / (target_range**2)) * 100
    train_rmse_pct = (train_rmse / target_range) * 100
    test_mse_pct = (test_mse / (target_range**2)) * 100
    test_rmse_pct = (test_rmse / target_range) * 100
    
    return {
        'train': {
            'mse': train_mse, 'mse_pct': train_mse_pct,
            'rmse': train_rmse, 'rmse_pct': train_rmse_pct,
            'r2': train_r2
        },
        'test': {
            'mse': test_mse, 'mse_pct': test_mse_pct,
            'rmse': test_rmse, 'rmse_pct': test_rmse_pct,
            'r2': test_r2
        }
    }

def create_contour_plot_plotly(lon_test, lat_test, actual_values, predicted_values, 
                             title, region, target, mineralization_df):
    """Create contour plot for actual vs predicted values."""
    grid_x, grid_y = np.mgrid[
        lon_test.min():lon_test.max():100j,
        lat_test.min():lat_test.max():100j
    ]
    
    # Fix column names for mineralization data
    if not mineralization_df.empty:
        # Standardize column names
        mineralization_df = mineralization_df.rename(columns={
            'LONGITUDE': 'longitude',
            'LATITUDE': 'latitude',
            'longitude': 'longitude',  # Keep if already correct
            'latitude': 'latitude'     # Keep if already correct
        })
    
    mask = (
        (mineralization_df['longitude'] >= lon_test.min()) &
        (mineralization_df['longitude'] <= lon_test.max()) &
        (mineralization_df['latitude'] >= lat_test.min()) &
        (mineralization_df['latitude'] <= lat_test.max())
    ) if not mineralization_df.empty else pd.Series([])
    
    filtered_mineralization = mineralization_df[mask].copy() if not mineralization_df.empty else pd.DataFrame()
    
    actual_grid = griddata(
        (lon_test, lat_test),
        actual_values,
        (grid_x, grid_y),
        method='cubic'
    )
    
    predicted_grid = griddata(
        (lon_test, lat_test),
        predicted_values,
        (grid_x, grid_y),
        method='cubic'
    )
    
    valid_mask = ~np.isnan(actual_grid)
    valid_mask = binary_dilation(valid_mask, iterations=3)
    
    actual_grid[~valid_mask] = np.nan
    predicted_grid[~valid_mask] = np.nan
    
    z_min = 0
    z_max = max(np.nanmax(actual_grid), np.nanmax(predicted_grid))
    interval_size = z_max / 3
    colorbar_intervals = [0, interval_size, 2 * interval_size, z_max]
    
    concentration_colorscale = [
        [0, 'green'], [0.33, 'yellowgreen'],
        [0.33, 'yellow'], [0.66, 'orange'],
        [0.66, 'red'], [1, 'darkred']
    ]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Actual {target} Concentration",
            f"Predicted {target} Concentration ({title})"
        ],
        horizontal_spacing=0.15
    )
    
    # Common settings for concentration plots
    conc_settings = dict(
        colorscale=concentration_colorscale,
        zmin=z_min,
        zmax=z_max,
        contours=dict(
            coloring='heatmap',
            showlines=True,
            start=z_min,
            end=z_max,
            size=(z_max - z_min) / 30,
            showlabels=True,
            labelfont=dict(size=8)
        ),
        ncontours=10,
        connectgaps=False
    )

    # Add actual values contour plot
    fig.add_trace(go.Contour(
        z=actual_grid,
        x=np.linspace(lon_test.min(), lon_test.max(), 100),
        y=np.linspace(lat_test.min(), lat_test.max(), 100),
        name=f'Actual {target} Concentration',
        colorbar=dict(
            title=target,
            tickvals=colorbar_intervals,
            ticktext=[f"{val:.1f}" for val in colorbar_intervals],
            tickmode='array',
            x=0.46,
            y=0.5,
            len=0.9,
            thickness=25,
            tickfont=dict(size=12)
        ),
        **conc_settings
    ), row=1, col=1)

    # Add predicted values contour plot
    fig.add_trace(go.Contour(
        z=predicted_grid,
        x=np.linspace(lon_test.min(), lon_test.max(), 100),
        y=np.linspace(lat_test.min(), lat_test.max(), 100),
        name=f'Predicted {target} Concentration',
        colorbar=dict(
            title=target,
            tickvals=colorbar_intervals,
            ticktext=[f"{val:.1f}" for val in colorbar_intervals],
            tickmode='array',
            x=1.05,
            y=0.5,
            len=0.9,
            thickness=25,
            tickfont=dict(size=12)
        ),
        **conc_settings
    ), row=1, col=2)

    # Add scatter points and mineralization points
    for col in [1, 2]:
        fig.add_trace(go.Scatter(
            x=lon_test,
            y=lat_test,
            mode='markers',
            marker=dict(color='black', size=4),
            name='Test Points',
            showlegend=False
        ), row=1, col=col)

        if len(filtered_mineralization) > 0:
            fig.add_trace(go.Scatter(
                x=filtered_mineralization['longitude'],
                y=filtered_mineralization['latitude'],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='blue',
                    line=dict(color='white', width=1)
                ),
                text=filtered_mineralization['ShortName'],
                textposition="top center",
                name='Mineralization Points',
                textfont=dict(size=10),
                showlegend=False
            ), row=1, col=col)

    # Update layout
    fig.update_layout(
        title_text=f"Concentration Analysis for {target} in district/toposheet {region}",
        title_x=0.5,
        height=600,
        width=1200,
        margin=dict(r=100, l=100),
        showlegend=False
    )

    # Update axes
    for j in [1, 2]:
        fig.update_xaxes(
            title_text="Longitude",
            scaleanchor=f'y{j}',
            scaleratio=1,
            row=1,
            col=j
        )
        fig.update_yaxes(
            title_text="Latitude",
            row=1,
            col=j
        )

    return fig, grid_x, grid_y, actual_grid, predicted_grid

# UI Functions (Already in app.py)
def sidebar_controls():
    """Create sidebar controls for the application."""
    st.sidebar.title("🎯 Control Panel")
    
    # Theme selection
    theme = st.sidebar.selectbox(
        "Color Theme",
        ["Default", "Viridis", "Magma", "Plasma"],
        help="Select color scheme for plots"
    )
    
    # Export settings
    st.sidebar.subheader("Export Settings")
    export_format = st.sidebar.radio(
        "Export Format",
        ["CSV", "Shapefile", "Both"]
    )
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    show_advanced = st.sidebar.checkbox("Show Advanced Options")
    
    if show_advanced:
        interpolation_method = st.sidebar.selectbox(
            "Interpolation Method",
            ["cubic", "linear", "nearest"],
            help="Method used for spatial interpolation"
        )
        contour_levels = st.sidebar.slider(
            "Contour Levels",
            5, 50, 20,
            help="Number of contour levels in plots"
        )
    else:
        interpolation_method = "cubic"
        contour_levels = 20
    
    return {
        "theme": theme,
        "export_format": export_format,
        "interpolation_method": interpolation_method,
        "contour_levels": contour_levels
    }

def save_grid_as_shapefile(grid_x, grid_y, grid_z, data_folder, target, region, data_type, crs="EPSG:4326"):
    """Save the interpolated grid as a shapefile."""
    try:
        # Flatten the grid arrays
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        flat_z = grid_z.flatten()

        # Remove NaN values
        valid_indices = ~np.isnan(flat_z)
        flat_x = flat_x[valid_indices]
        flat_y = flat_y[valid_indices]
        flat_z = flat_z[valid_indices]

        # Create a GeoDataFrame
        geometries = [Point(x, y) for x, y in zip(flat_x, flat_y)]
        gdf = gpd.GeoDataFrame({
            "Value": flat_z,
            "Longitude": flat_x,
            "Latitude": flat_y
        }, geometry=geometries)

        # Set the CRS
        gdf.set_crs(crs, inplace=True)

        # Create directory if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)

        # Save to shapefile
        output_path = f"{data_folder}/{target}_{region}_{data_type}.shp"
        gdf.to_file(output_path, driver="ESRI Shapefile")
        
        return output_path
    except Exception as e:
        raise Exception(f"Error saving shapefile: {str(e)}")

def main():
    """Main application function."""
    # Get sidebar settings
    settings = sidebar_controls()
    st.session_state.settings = settings
    
    # Application title
    st.title("🌍 AI/ML in Mineral Exploration")
    st.markdown("""
    This application helps analyze and predict mineral concentrations using machine learning.
    Upload your data, train models, and visualize results with interactive maps.
    """)
    
    # Main sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Analysis",
        "🔍 Exploratory Analysis",
        "🤖 Model Training",
        "🗺️ Visualization"
    ])
    
    with tab1:
        st.header("Data Analysis")
        data_section()
    
    with tab2:
        st.header("Exploratory Analysis")
        if 'df' in st.session_state:
            add_exploratory_analysis(
                st.session_state.df,
                st.session_state.get('target_element', 'CU')
            )
        else:
            st.warning("⚠️ Please upload data first")
    
    with tab3:
        st.header("Model Training & Evaluation")
        model_section()
    
    with tab4:
        st.header("Spatial Visualization")
        visualization_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Critical Mineral Trackers Geo Solution Centre<br>
               Tech Research Park, IIT, Hyderabad.<br>
               director@cmtgsc.ai, Phone +91 8500975852</p>
        </div>
    """, unsafe_allow_html=True)

def data_section():
    """Handle data upload and initial analysis."""
    st.subheader("Upload Your Data")
    
    # File upload
    data_file = st.file_uploader(
        "Upload your dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file containing your mineral concentration data"
    )
    
    mineralization_file = st.file_uploader(
        "Upload mineralization points (CSV)",
        type=['csv'],
        help="Upload a CSV file containing mineralization points"
    )
    
    if data_file is not None:
        df = pd.read_csv(data_file)
        st.session_state.df = df
        st.success("✅ Data uploaded successfully!")
        
        # Data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            st.text(f"Shape of data: {df.shape}")
        
        # Target element selection
        target_element = st.selectbox(
            "Select target element for analysis",
            options=[col for col in df.columns if col not in ['OBJECTID', 'SAMPLENO', 'LONGITUDE', 'LATITUDE', 'geometry', 'TOPOSHEET', 'REGION']],
            key="target_element"
        )
        
        if mineralization_file is not None:
            mineralization_df = pd.read_csv(mineralization_file)
            # Standardize column names
            mineralization_df = mineralization_df.rename(columns={
                'LONGITUDE': 'longitude',
                'LATITUDE': 'latitude'
            })
            st.session_state.mineralization_df = mineralization_df
            st.success("✅ Mineralization data uploaded successfully!")
            
            # Show preview of mineralization data
            with st.expander("Preview Mineralization Data"):
                st.dataframe(mineralization_df.head())
                st.text(f"Number of mineralization points: {len(mineralization_df)}")
        
        if st.button("Analyze Correlations"):
            with st.spinner("Analyzing correlations..."):
                correlated_elements = analyze_element_correlations(df, target_element)
                st.write("Top correlated elements:", correlated_elements)
                
                # Store in session state for use in other sections
                st.session_state.correlated_elements = correlated_elements
                st.session_state.target = target_element

def model_section():
    """Handle model training and evaluation."""
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data in the Data Analysis tab first")
        return
        
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model parameters
        st.write("Random Forest Parameters")
        n_estimators = st.slider("Number of trees", 100, 5000, 3000, 100)
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        
    with col2:
        # Feature selection
        st.write("Feature Selection")
        target = st.session_state.get('target_element', 'CU')
        top_corr = st.slider("Top correlated features (%)", 10, 50, 30, 5)
    
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training model..."):
            try:
                # Get correlated features
                top_features = analyze_element_correlations(
                    st.session_state.df, 
                    target, 
                    top_percent=top_corr
                )
                
                # Prepare data
                cor_df = st.session_state.df[['OBJECTID', 'SAMPLENO', 'LONGITUDE', 
                                            'LATITUDE', 'geometry', 'TOPOSHEET', 
                                            'REGION', target] + top_features]
                
                # Split and scale data
                X_train_scaled, X_test_scaled, y_train, y_test, lon_test, lat_test, scaler, feature_cols = prepare_data(
                    cor_df, 
                    target=target, 
                    test_size=test_size
                )
                
                # Train model
                rf_model = train_random_forest(X_train_scaled, y_train)
                
                # Make predictions
                rf_pred = rf_model.predict(X_test_scaled)
                
                # Evaluate model
                metrics = evaluate_model(
                    rf_model, 
                    X_train_scaled, 
                    X_test_scaled, 
                    y_train, 
                    y_test, 
                    "Random Forest",
                    target_range=y_test.max()
                )
                
                # Store results in session state
                st.session_state.model_results = {
                    'model': rf_model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'test_data': {
                        'X_test': X_test_scaled,
                        'y_test': y_test,
                        'lon_test': lon_test,
                        'lat_test': lat_test,
                        'predictions': rf_pred
                    },
                    'metrics': metrics
                }
                
                # Display results
                display_model_results()
                
                # Add cross-validation results
                add_cross_validation(rf_model, X_train_scaled, y_train)
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

def visualization_section():
    """Handle spatial visualization of results."""
    if 'model_results' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Model Training tab")
        return
        
    st.subheader("Spatial Visualization")
    
    # Get data from session state
    test_data = st.session_state.model_results['test_data']
    target = st.session_state.get('target_element', 'CU')
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Plot Configuration")
        plot_title = st.text_input("Plot Title", f"Concentration Analysis for {target}")
        region = st.text_input("Region/Toposheet", "Study Area")
        
    with col2:
        st.write("Grid Configuration")
        grid_resolution = st.slider("Grid Resolution", 50, 200, 100, 10)
        smoothing = st.slider("Smoothing Factor", 0.1, 2.0, 1.0, 0.1)
    
    if st.button("Generate Visualization", key="generate_viz"):
        with st.spinner("Generating visualization..."):
            try:
                # Create contour plot
                fig, grid_x, grid_y, actual_grid, pred_grid = create_contour_plot_plotly(
                    test_data['lon_test'],
                    test_data['lat_test'],
                    test_data['y_test'].values,
                    test_data['predictions'],
                    plot_title,
                    region,
                    target,
                    st.session_state.get('mineralization_df', pd.DataFrame())
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add export options
                add_export_options(grid_x, grid_y, actual_grid, pred_grid, target, region)
                
                # Add error analysis
                st.subheader("Error Analysis")
                error = test_data['predictions'] - test_data['y_test']
                
                fig_error = px.histogram(
                    error,
                    title="Prediction Error Distribution",
                    labels={'value': 'Error', 'count': 'Frequency'},
                    marginal="box"
                )
                st.plotly_chart(fig_error, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")

def add_exploratory_analysis(df, target):
    """Add exploratory data analysis visualizations."""
    st.subheader("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic statistics
        st.write("Basic Statistics")
        st.dataframe(df[target].describe())
        
        # Distribution plot
        fig_dist = px.histogram(
            df, x=target,
            title=f"{target} Distribution",
            marginal="box"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Spatial distribution
        st.write("Spatial Distribution")
        fig_spatial = px.scatter_mapbox(
            df,
            lat='LATITUDE',
            lon='LONGITUDE',
            color=target,
            size=target,
            title=f"Spatial Distribution of {target}",
            mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_spatial, use_container_width=True)

def add_cross_validation(model, X, y, cv=5):
    """Add cross-validation results."""
    st.subheader("Cross-Validation Results")
    
    with st.spinner("Performing cross-validation..."):
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        st.write(f"Cross-validation R² scores (CV={cv}):")
        cv_df = pd.DataFrame({
            'Fold': range(1, cv+1),
            'R² Score': cv_scores
        })
        st.dataframe(cv_df)
        
        st.write(f"Mean R² Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

def add_export_options(grid_x, grid_y, actual_grid, pred_grid, target, region):
    """Add options for exporting results."""
    st.subheader("Export Options")
    
    export_format = st.session_state.get('settings', {}).get('export_format', 'Both')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if export_format in ['Shapefile', 'Both']:
            if st.button("Export as Shapefile", key="export_shp"):
                try:
                    with st.spinner("Generating shapefiles..."):
                        # Create results directory if it doesn't exist
                        os.makedirs('results', exist_ok=True)
                        
                        # Export actual values
                        save_grid_as_shapefile(
                            grid_x, grid_y, actual_grid,
                            'results', target, region, 'actual'
                        )
                        
                        # Export predicted values
                        save_grid_as_shapefile(
                            grid_x, grid_y, pred_grid,
                            'results', target, region, 'predicted'
                        )
                        st.success("✅ Shapefiles exported successfully to 'results' folder!")
                except Exception as e:
                    st.error(f"Error exporting shapefiles: {str(e)}")
    
    with col2:
        if export_format in ['CSV', 'Both']:
            try:
                # Prepare results DataFrame
                results_df = pd.DataFrame({
                    'Longitude': grid_x.flatten(),
                    'Latitude': grid_y.flatten(),
                    f'Actual_{target}': actual_grid.flatten(),
                    f'Predicted_{target}': pred_grid.flatten()
                })
                results_df = results_df.dropna()
                
                # Add download button
                st.download_button(
                    "📥 Download CSV Results",
                    results_df.to_csv(index=False).encode('utf-8'),
                    f"{target}_{region}_results.csv",
                    "text/csv",
                    key="download_csv"
                )
                
                # Also save to results folder
                os.makedirs('results', exist_ok=True)
                results_df.to_csv(f'results/{target}_{region}_results.csv', index=False)
                st.success("✅ CSV also saved to 'results' folder!")
            except Exception as e:
                st.error(f"Error exporting CSV: {str(e)}")

def display_model_results():
    """Display model training results and metrics."""
    if 'model_results' not in st.session_state:
        return
        
    metrics = st.session_state.model_results['metrics']
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Training Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'R² Score'],
            'Value': [
                f"{metrics['train']['mse']:.4f} ({metrics['train']['mse_pct']:.2f}%)",
                f"{metrics['train']['rmse']:.4f} ({metrics['train']['rmse_pct']:.2f}%)",
                f"{metrics['train']['r2']:.4f}"
            ]
        })
        st.table(metrics_df)
    
    with col2:
        st.write("Test Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'R² Score'],
            'Value': [
                f"{metrics['test']['mse']:.4f} ({metrics['test']['mse_pct']:.2f}%)",
                f"{metrics['test']['rmse']:.4f} ({metrics['test']['rmse_pct']:.2f}%)",
                f"{metrics['test']['r2']:.4f}"
            ]
        })
        st.table(metrics_df)
    
    # Feature importance plot
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'feature': st.session_state.model_results['feature_cols'],
        'importance': st.session_state.model_results['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(
        importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance Analysis'
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# FLORIDA NDVI/EVI VEGETATION INDEX ANALYSIS STREAMLIT APP
# Based on LST app structure and design patterns
# -----------------------------------------------------------------------------
import ee
import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import geemap.foliumap as geemap
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import json
import warnings
import geopandas as gpd
import fiona
import tempfile
import os
import uuid
from shapely.geometry import Polygon, shape
import numpy as np
from branca.colormap import linear
from scipy import stats
import time
import gc
import atexit
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="Florida NDVI/EVI Vegetation Analysis",
    layout="wide"
)
warnings.filterwarnings("ignore")

# Initialize session state
if "vegetation_results" not in st.session_state:
    st.session_state["vegetation_results"] = None
if "roi" not in st.session_state:
    st.session_state["roi"] = None
if "current_map_roi" not in st.session_state:
    st.session_state["current_map_roi"] = None
if "zoom_level" not in st.session_state:
    st.session_state["zoom_level"] = 4
if "analysis_params" not in st.session_state:
    st.session_state["analysis_params"] = {}
if "florida_counties_gdf" not in st.session_state:
    st.session_state["florida_counties_gdf"] = None
if "temp_files" not in st.session_state:
    st.session_state["temp_files"] = []

# -----------------------------------------------------------------------------
# CONSTANTS AND CONFIGURATION
# -----------------------------------------------------------------------------
# Band names for different Landsat missions
BAND_INFO = {
    'L8_9': {'NIR': 'SR_B5', 'Red': 'SR_B4', 'Blue': 'SR_B2'},  # Landsat 8 & 9
    'L5_7': {'NIR': 'SR_B4', 'Red': 'SR_B3', 'Blue': 'SR_B1'}   # Landsat 5 & 7
}

# Visualization parameters for NDVI and EVI
VIS_PARAMS = {
    'NDVI': {
        'min': -1.0, 'max': 1.0,
        'palette': ['beige', 'darkgreen'],
        'label': 'NDVI',
        'description': 'Normalized Difference Vegetation Index',
        'rank_high': '🌳 Top 5 Highest (Densest Vegetation)',
        'rank_low': '🍂 Top 5 Lowest (Least Vegetation)'
    },
    'EVI': {
        'min': -1.0, 'max': 1.0,
        'palette': ['#E6E6FA', '#006400'],  # Light purple to dark green
        'label': 'EVI',
        'description': 'Enhanced Vegetation Index',
        'rank_high': '🌳 Top 5 Highest (Healthiest Vegetation)',
        'rank_low': '🍂 Top 5 Lowest (Stressed/No Vegetation)'
    }
}

# -----------------------------------------------------------------------------
# FILE HANDLING UTILITIES (SAME AS LST APP)
# -----------------------------------------------------------------------------
def cleanup_temp_files():
    """Clean up temporary files safely"""
    if "temp_files" in st.session_state:
        for file_path in st.session_state["temp_files"]:
            try:
                if os.path.exists(file_path):
                    os.chmod(file_path, 0o777)
                    time.sleep(0.1)
                    os.remove(file_path)
            except Exception:
                try:
                    if os.path.exists(file_path):
                        shutil.move(file_path, file_path + ".deleted")
                except:
                    pass
        st.session_state["temp_files"] = []

def create_temp_file(data, extension):
    """Create temporary file with better handling"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        temp_dir = tempfile.gettempdir()
        custom_temp_dir = os.path.join(temp_dir, "streamlit_vegetation_app")
        
        os.makedirs(custom_temp_dir, exist_ok=True)
        file_path = os.path.join(custom_temp_dir, f"{file_id}{extension}")
        
        with open(file_path, "wb") as file:
            file.write(data.getbuffer())
            file.flush()
            os.fsync(file.fileno())
        
        if "temp_files" not in st.session_state:
            st.session_state["temp_files"] = []
        st.session_state["temp_files"].append(file_path)
        
        return file_path
        
    except Exception as e:
        st.error(f"Error creating temporary file: {e}")
        return None

# Register cleanup function
atexit.register(cleanup_temp_files)

# -----------------------------------------------------------------------------
# EARTH ENGINE AUTHENTICATION AND INITIALIZATION
# -----------------------------------------------------------------------------
def ee_authenticate_safe():
    """Initialize Earth Engine authentication with better error handling"""
    try:
        try:
            geemap.ee_initialize(token_name="EARTHENGINE_TOKEN")
            return True
        except:
            pass
        
        try:
            ee.Initialize()
            return True
        except:
            pass
        
        try:
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as auth_error:
            st.error(f"Earth Engine authentication failed: {auth_error}")
            st.info("Please ensure you have properly authenticated with Google Earth Engine.")
            return False
            
    except Exception as e:
        st.error(f"Critical error during Earth Engine initialization: {e}")
        return False

# -----------------------------------------------------------------------------
# FLORIDA COUNTIES AND GEOMETRY FUNCTIONS (SAME AS LST APP)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_florida_counties():
    """Get Florida counties from TIGER/2018 datasets"""
    try:
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties.filter(ee.Filter.eq('STATEFP', '12'))
        
        states = ee.FeatureCollection("TIGER/2018/States")
        florida = states.filter(ee.Filter.eq('NAME', 'Florida')).first()
        
        rois = {
            "Florida (All Counties)": florida.geometry()
        }
        
        county_features = florida_counties.getInfo()['features']
        
        for feature in county_features:
            county_name = feature['properties']['NAME']
            county_geom = ee.Geometry(feature['geometry'])
            if county_geom:
                rois[f"{county_name} County"] = county_geom
                
        return rois
    
    except Exception as e:
        st.error(f"Error loading Florida counties from GEE: {e}")
        return {
            "Florida (All Counties)": ee.Geometry.Rectangle([-87.634896, 24.396308, -79.974306, 31.000968])
        }

@st.cache_data(ttl=3600)
def get_florida_counties_gdf():
    """Get Florida counties as GeoDataFrame for map display"""
    try:
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties.filter(ee.Filter.eq('STATEFP', '12'))
        
        county_data = florida_counties.getInfo()['features']
        geometries = []
        names = []
        
        for feature in county_data:
            geom = shape(feature['geometry'])
            name = feature['properties']['NAME']
            geometries.append(geom)
            names.append(name)
        
        gdf = gpd.GeoDataFrame({
            'NAME': names,
            'geometry': geometries
        }, crs='EPSG:4326')
        
        return gdf
    except Exception as e:
        st.error(f"Error loading Florida counties GeoDataFrame: {e}")
        return None

def uploaded_file_to_gdf_safe(data):
    """Convert uploaded file to GeoDataFrame with safe file handling"""
    file_path = None
    try:
        _, file_extension = os.path.splitext(data.name)
        
        file_path = create_temp_file(data, file_extension)
        if not file_path:
            raise Exception("Could not create temporary file")
        
        time.sleep(0.1)
        
        if file_path.lower().endswith(".kml"):
            fiona.drvsupport.supported_drivers["KML"] = "rw"
            gdf = gpd.read_file(file_path, driver="KML")
        else:
            gdf = gpd.read_file(file_path)
        
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        return gdf
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# -----------------------------------------------------------------------------
# VEGETATION INDEX PROCESSING FUNCTIONS (FROM JUPYTER NOTEBOOK)
# -----------------------------------------------------------------------------
def mask_landsat_clouds(image):
    """Masks clouds and cloud shadows in Landsat Collection 2 images."""
    qa = image.select('QA_PIXEL')
    cloud_mask = (1 << 3) | (1 << 5)
    mask = qa.bitwiseAnd(cloud_mask).eq(0)
    return image.updateMask(mask)

def apply_scale_factors(image):
    """Applies scaling factors to optical (Surface Reflectance) bands."""
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical_bands, overwrite=True)

def add_vegetation_indices(image):
    """Calculates NDVI and EVI and adds them as bands to the image."""
    scaled_image = apply_scale_factors(image)
    spacecraft = image.get('SPACECRAFT_ID')

    # Select bands based on the satellite
    bands = ee.Dictionary(ee.Algorithms.If(
        ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(spacecraft),
        BAND_INFO['L8_9'],
        BAND_INFO['L5_7']
    ))

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = scaled_image.normalizedDifference([bands.getString('NIR'), bands.getString('Red')]).rename('NDVI')

    # Calculate EVI: 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    evi = scaled_image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': scaled_image.select(bands.getString('NIR')),
            'RED': scaled_image.select(bands.getString('Red')),
            'BLUE': scaled_image.select(bands.getString('Blue'))
        }).rename('EVI')

    return image.addBands([ndvi, evi])

def get_mean_image_for_year(year, months, geometry, index_name):
    """Creates a mean composite image for a given index, year, month range, and geometry."""
    start_date = ee.Date.fromYMD(year, months[0], 1)
    end_date = ee.Date.fromYMD(year, months[1], 1).advance(1, 'month').advance(-1, 'day')
    
    # Combine all Landsat collections
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    
    landsat_collection = l9.merge(l8).merge(l7).merge(l5)

    image_composite = (landsat_collection
                       .filterBounds(geometry)
                       .filterDate(start_date, end_date)
                       .map(mask_landsat_clouds)
                       .map(add_vegetation_indices)
                       .select(index_name)
                       .mean())
    
    return image_composite.set('year', year)

def get_yearly_stats_fc(year, months, index_name, collection, scale):
    """Calculates zonal statistics for a given year and returns a FeatureCollection."""
    mean_image = get_mean_image_for_year(year, months, collection.geometry(), index_name)
    
    return mean_image.reduceRegions(
        collection=collection,
        reducer=ee.Reducer.mean(),
        scale=scale
    ).map(lambda f: f.set('year', year))

def generate_year_range(start_year, end_year, delta_years):
    """Generate year range based on start, end, and delta"""
    years = list(range(start_year, end_year + 1, delta_years))
    if end_year not in years:
        years.append(end_year)
    return years

def analyze_vegetation_for_county(feature, years_to_process, month_range, index_name):
    """Analyze vegetation index for a single county feature"""
    geom = feature.geometry()
    
    properties = {
        'NAME': feature.get('NAME'),
    }
    
    for year in years_to_process:
        try:
            mean_image = get_mean_image_for_year(year, month_range, geom, index_name)
            stats = mean_image.reduceRegions(
                collection=ee.FeatureCollection([feature]),
                reducer=ee.Reducer.mean(),
                scale=200
            )
            
            mean_value = stats.first().get('mean')
            properties[f'{index_name}_{year}'] = mean_value
            
        except Exception:
            properties[f'{index_name}_{year}'] = None
    
    return feature.set(properties)

def analyze_region_vegetation_batch(region_geometry, region_name, years_to_process, month_range, index_name):
    """Analyze vegetation index for a region using batch processing"""
    
    try:
        if region_name == "Florida (All Counties)":
            counties_collection = ee.FeatureCollection("TIGER/2018/Counties")
            florida_counties = counties_collection.filter(ee.Filter.eq('STATEFP', '12'))
            
            county_list = florida_counties.toList(florida_counties.size())
            batch_size = 3
            n_counties = florida_counties.size().getInfo()
            
            all_features = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for start in range(0, n_counties, batch_size):
                batch_progress = (start + batch_size) / n_counties
                progress_bar.progress(min(batch_progress, 1.0))
                status_text.text(f"Processing counties {start + 1}-{min(start + batch_size, n_counties)} of {n_counties}...")
                
                try:
                    subset = ee.FeatureCollection(county_list.slice(start, start + batch_size))
                    subset_results = subset.map(
                        lambda f: analyze_vegetation_for_county(f, years_to_process, month_range, index_name)
                    ).getInfo()
                    all_features.extend(subset_results['features'])
                    
                    time.sleep(0.5)
                    
                except Exception as batch_error:
                    st.warning(f"Error processing batch {start + 1}-{min(start + batch_size, n_counties)}: {batch_error}")
                    continue
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            data_for_df = []
            for i, feature in enumerate(all_features):
                props = feature['properties']
                row_data = {'S.N.': i + 1, 'County': props['NAME']}
                
                for year in years_to_process:
                    value = props.get(f'{index_name}_{year}')
                    if value is not None:
                        try:
                            row_data[f'{index_name}_{year}'] = round(float(value), 4)
                        except (ValueError, TypeError):
                            row_data[f'{index_name}_{year}'] = None
                    else:
                        row_data[f'{index_name}_{year}'] = None
                
                data_for_df.append(row_data)
            
            return pd.DataFrame(data_for_df)
        
        else:
            # Single county or region analysis
            if hasattr(region_geometry, 'geometry'):
                if hasattr(region_geometry, 'first'):
                    geometry = region_geometry.first().geometry()
                else:
                    geometry = region_geometry.geometry()
            else:
                geometry = region_geometry
            
            results = {
                'S.N.': 1,
                'Region': region_name,
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, year in enumerate(years_to_process):
                progress = (i + 1) / len(years_to_process)
                progress_bar.progress(progress)
                status_text.text(f"Processing year {year}...")
                
                try:
                    mean_image = get_mean_image_for_year(year, month_range, geometry, index_name)
                    stats = mean_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=200,
                        maxPixels=1e13
                    )
                    
                    mean_value = stats.get(index_name).getInfo()
                    if mean_value is not None:
                        try:
                            results[f'{index_name}_{year}'] = round(float(mean_value), 4)
                        except (ValueError, TypeError):
                            results[f'{index_name}_{year}'] = None
                    else:
                        results[f'{index_name}_{year}'] = None
                        
                except Exception as year_error:
                    st.warning(f"Error processing year {year}: {year_error}")
                    results[f'{index_name}_{year}'] = None
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            return pd.DataFrame([results])
            
    except Exception as e:
        st.error(f"Critical error in vegetation analysis: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def create_vegetation_choropleth_map(m, results_df, florida_counties_gdf, latest_year, index_name):
    """Create choropleth map with vegetation index data"""
    if florida_counties_gdf is None or results_df is None or results_df.empty:
        return m
    
    try:
        # Find the latest index column
        index_columns = [col for col in results_df.columns if f'{index_name}_{latest_year}' in col]
        
        if not index_columns:
            return m
        
        latest_index_col = index_columns[0]
        
        # Merge results with geometry data
        merged_gdf = florida_counties_gdf.merge(
            results_df[['County', latest_index_col]], 
            left_on='NAME', 
            right_on='County', 
            how='left'
        )
        
        # Fill NaN values with median for visualization
        valid_data = merged_gdf[latest_index_col].dropna()
        if not valid_data.empty:
            merged_gdf[latest_index_col] = merged_gdf[latest_index_col].fillna(valid_data.median())
            
            # Create choropleth layer
            vmin = valid_data.min()
            vmax = valid_data.max()
            
            if vmax > vmin:
                # Select color scheme based on index
                color_scheme = 'YlGn' if index_name in ['NDVI', 'EVI'] else 'RdYlBu_r'
                
                choropleth = folium.Choropleth(
                    geo_data=merged_gdf.to_json(),
                    name=f'{index_name} Choropleth ({latest_year})',
                    data=merged_gdf,
                    columns=['NAME', latest_index_col],
                    key_on='feature.properties.NAME',
                    fill_color=color_scheme,
                    fill_opacity=0.7,
                    line_opacity=0.5,
                    legend_name=f'{index_name} {latest_year}',
                    highlight=True,
                    show=True,
                    overlay=True
                )
                choropleth.add_to(m)
                
                # Add interactive tooltips
                tooltip_style = """
                background-color: white; border: 2px solid black; border-radius: 3px;
                box-shadow: 3px; font-size: 14px; font-weight: bold;
                """
                
                geojson_layer = folium.GeoJson(
                    merged_gdf.to_json(),
                    name=f'County {index_name} Information',
                    style_function=lambda x: {'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['NAME', latest_index_col],
                        aliases=['County:', f'{index_name} ({latest_year}):'],
                        localize=True, sticky=False, labels=True,
                        style=tooltip_style, max_width=200
                    )
                )
                geojson_layer.add_to(m)
                
                folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
    except Exception as e:
        st.warning(f"Could not create choropleth map: {e}")
    
    return m

def display_geometry_on_map(m, selected_region, florida_rois, florida_counties_gdf):
    """Display selected geometry on map with appropriate zoom"""
    
    try:
        if selected_region == "Florida (All Counties)":
            if florida_counties_gdf is not None:
                county_style = {
                    'fillColor': '#E8F4FD', 'color': '#2E86AB', 'weight': 2,
                    'fillOpacity': 0.3, 'opacity': 0.8
                }
                
                folium.GeoJson(
                    florida_counties_gdf.to_json(),
                    name='Florida Counties',
                    style_function=lambda x: county_style,
                    tooltip=folium.GeoJsonTooltip(fields=['NAME'], aliases=['County:'], localize=True)
                ).add_to(m)
                
                bounds = florida_counties_gdf.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                
        elif selected_region != "Uploaded GeoJSON":
            geometry = florida_rois.get(selected_region)
            if geometry:
                try:
                    geojson_geom = geometry.getInfo()
                    shapely_geom = shape(geojson_geom)
                    gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[shapely_geom])
                    
                    roi_style = {
                        'fillColor': '#FFE5CC', 'color': '#FF6B35', 'weight': 3,
                        'fillOpacity': 0.4, 'opacity': 1
                    }
                    
                    folium.GeoJson(
                        gdf.to_json(),
                        name='Selected ROI',
                        style_function=lambda x: roi_style
                    ).add_to(m)
                    
                    bounds = gdf.total_bounds
                    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                except Exception as e:
                    st.warning(f"Could not display geometry: {e}")
    
    except Exception as e:
        st.warning(f"Error in map display: {e}")
    
    return m

def create_vegetation_time_series_plot(results_df, index_name):
    """Create time series plot for vegetation index data"""
    if results_df is None or results_df.empty:
        return None
    
    try:
        # Get index columns
        index_columns = [col for col in results_df.columns if f'{index_name}_' in col and col != 'County']
        years = [int(col.split('_')[1]) for col in index_columns]
        
        if len(index_columns) == 0:
            return None
        
        fig = go.Figure()
        
        if len(results_df) == 1:
            # Single region - line plot
            values = []
            for col in index_columns:
                val = results_df[col].iloc[0]
                if pd.notna(val):
                    values.append(val)
                else:
                    values.append(None)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=results_df['Region'].iloc[0] if 'Region' in results_df.columns else 'Single Region',
                line=dict(width=3),
                marker=dict(size=8),
                connectgaps=False
            ))
        else:
            # Multiple counties - show top 5 highest and lowest
            mean_values = {}
            for _, row in results_df.iterrows():
                county = row['County']
                values = [row[col] for col in index_columns if pd.notna(row[col])]
                if values:
                    mean_values[county] = np.mean(values)
            
            if mean_values:
                sorted_counties = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)
                top_5_high = sorted_counties[:5]
                top_5_low = sorted_counties[-5:]
                
                counties_to_plot = dict(top_5_high + top_5_low)
                
                for county, _ in counties_to_plot.items():
                    county_row = results_df[results_df['County'] == county].iloc[0]
                    values = []
                    for col in index_columns:
                        val = county_row[col]
                        if pd.notna(val):
                            values.append(val)
                        else:
                            values.append(None)
                    
                    color = 'darkgreen' if county in dict(top_5_high) else 'brown'
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=values,
                        mode='lines+markers',
                        name=county,
                        line=dict(width=2, color=color),
                        marker=dict(size=6),
                        connectgaps=False
                    ))
        
        fig.update_layout(
            title=f"{VIS_PARAMS[index_name]['description']} Time Series",
            xaxis_title="Year",
            yaxis_title=f"{index_name}",
            hovermode='x unified',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create time series plot: {e}")
        return None

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    # Clean up any existing temp files at start
    cleanup_temp_files()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        - **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation density and health
        - **EVI (Enhanced Vegetation Index)**: Improved vegetation monitoring, more sensitive to canopy structure
        - **Data Source**: Landsat Collection 2 Surface Reflectance (Landsat 5, 7, 8, and 9)
        - **Processing**: Cloud masking, surface reflectance scaling, vegetation index calculation
        """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        Vegetation Analysis Tool  
        📧 [Email](mailto:support@example.com)  
        💻 [GitHub](https://github.com/example/vegetation-analysis)
        """
    )
    
    # Main content
    st.title("Florida Vegetation Index Analysis (NDVI/EVI)")
    st.markdown(
        """
        An interactive web app for analyzing vegetation indices using 
        [Landsat satellite data](https://developers.google.com/earth-engine/datasets/catalog/landsat) from Google Earth Engine.
        
        **NDVI (Normalized Difference Vegetation Index)** measures vegetation density and health by comparing 
        near-infrared and red light reflectance. Values range from -1 to 1, with higher values indicating 
        healthier, denser vegetation.
        
        **EVI (Enhanced Vegetation Index)** provides improved vegetation monitoring compared to NDVI, 
        with better sensitivity to canopy structure and reduced atmospheric effects.
        
        **Data Source:** Landsat Collection 2 Surface Reflectance (Landsat 5, 7, 8, and 9)  
        **Processing:** Cloud masking, surface reflectance scaling, temporal aggregation
        """
    )
    
    # Initialize Earth Engine with better error handling
    if not ee_authenticate_safe():
        st.stop()
    
    # Get Florida ROIs and counties GDF with error handling
    try:
        florida_rois = get_florida_counties()
        if st.session_state["florida_counties_gdf"] is None:
            st.session_state["florida_counties_gdf"] = get_florida_counties_gdf()
        florida_counties_gdf = st.session_state["florida_counties_gdf"]
    except Exception as e:
        st.error(f"Could not load Florida counties data: {e}")
        st.stop()
    
    # Create layout
    row1_col1, row1_col2 = st.columns([2, 1])

    # Initialize map
    with row1_col1:
        try:
            m = geemap.Map(
                basemap="HYBRID",
                plugin_Draw=True,
                Draw_export=True,
                locate_control=True,
                plugin_LatLngPopup=False,
            )
            m.add_basemap("ROADMAP")
        except Exception as e:
            st.error(f"Could not initialize map: {e}")
            # Fallback to basic folium map
            m = folium.Map(location=[27.6648, -81.5158], zoom_start=6)

    with row1_col2:
        # Search for location section
        keyword = st.text_input("Search for a location:", "")
        if keyword:
            try:
                locations = geemap.geocode(keyword)
                if locations is not None and len(locations) > 0:
                    str_locations = [str(g)[1:-1] for g in locations]
                    location = st.selectbox("Select a location:", str_locations)
                    loc_index = str_locations.index(location)
                    selected_loc = locations[loc_index]
                    lat, lng = selected_loc.lat, selected_loc.lng
                    
                    folium.Marker(
                        location=[lat, lng], 
                        popup=location,
                        icon=folium.Icon(color='red', icon='search')
                    ).add_to(m)
                    m.set_center(lng, lat, 12)
            except Exception as e:
                st.warning(f"Could not geocode location: {e}")
        
        # Vegetation Index Selection

        
        selected_index = st.selectbox(
            "Select Vegetation Index:",
            ['NDVI', 'EVI'],
            help="NDVI: Vegetation density and health | EVI: Enhanced vegetation monitoring"
        )
        
        # Region selection
        roi_options = ["Uploaded GeoJSON"] + list(florida_rois.keys())
        
        selected_region = st.selectbox(
            "Select a sample ROI or upload a GeoJSON file:",
            roi_options,
            key="roi_selectbox"
        )
        
        # Time period inputs
    
        col1, col2, col3 = st.columns(3)
        with col1:
            start_year = st.number_input("Start Year:", min_value=1985, max_value=2024, value=2000, step=1)
        with col2:
            end_year = st.number_input("End Year:", min_value=1985, max_value=2024, value=2024, step=1)
        with col3:
            delta_years = st.selectbox("Delta (years):", list(range(1, 21)), index=4)  # Default to 5 years
        
        # Month range selection

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        col1, col2 = st.columns(2)
        with col1:
            start_month = st.selectbox("Start Month:", range(1, 13), index=3,  # Default to April
                                     format_func=lambda x: month_names[x-1])
        with col2:
            end_month = st.selectbox("End Month:", range(1, 13), index=8,  # Default to September
                                   format_func=lambda x: month_names[x-1])
        
        if start_year >= end_year:
            st.error("End year must be after start year!")
            st.stop()
        
        if start_month > end_month:
            st.warning("Start month is after end month. This will analyze across year boundaries.")
        
        # Generate year range
        years_to_process = generate_year_range(start_year, end_year, delta_years)
        month_range = [start_month, end_month]
        
        if len(years_to_process) < 1:
            st.warning("No years to process with current settings.")

    # File upload and map display
    with row1_col1:
        with st.expander("Upload Custom Region", False):
            st.info("Draw a rectangle on the map → Export as GeoJSON → Upload below → Click Submit")
            data = st.file_uploader(
                "Upload a GeoJSON file:", 
                type=["geojson", "kml", "zip"],
                help="Max 200MB • Supported: GeoJSON, KML, ZIP"
            )
        
        # Handle region selection and map display
        if selected_region == "Uploaded GeoJSON":
            if data:
                try:
                    gdf = uploaded_file_to_gdf_safe(data)
                    if gdf is not None:
                        st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
                        
                        roi_style = {
                            'fillColor': '#FFE5CC', 'color': '#FF6B35', 'weight': 3, 'fillOpacity': 0.4
                        }
                        
                        folium.GeoJson(
                            gdf.to_json(), name='Uploaded ROI', style_function=lambda x: roi_style
                        ).add_to(m)
                        
                        bounds = gdf.total_bounds
                        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                        st.success(f"✅ Loaded: {data.name}")
                    else:
                        st.error("❌ Could not process the uploaded file")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            else:
                st.info("👆 Upload a GeoJSON file above")
        else:
            try:
                geometry = florida_rois.get(selected_region)
                if geometry:
                    st.session_state["roi"] = geometry
                    m = display_geometry_on_map(m, selected_region, florida_rois, florida_counties_gdf)
            except Exception as e:
                st.warning(f"Could not display selected region: {e}")
        
        # Add choropleth overlay if results exist for the "All Counties" analysis
        try:
            if (st.session_state["vegetation_results"] is not None and 
                selected_region == "Florida (All Counties)" and
                st.session_state["analysis_params"]):
                latest_year = max(st.session_state["analysis_params"]["years_to_process"])
                analysis_index = st.session_state["analysis_params"]["selected_index"]
                m = create_vegetation_choropleth_map(
                    m, 
                    st.session_state["vegetation_results"], 
                    florida_counties_gdf, 
                    latest_year,
                    analysis_index
                )
        except Exception as e:
            st.warning(f"Could not add choropleth overlay: {e}")
        
        # Display map
        try:
            if hasattr(m, 'to_streamlit'):
                m.to_streamlit(height=600)
            else:
                # Fallback for basic folium maps
                st_folium = st.components.v1.html(m._repr_html_(), height=600)
        except Exception as e:
            st.error(f"Could not display map: {e}")

    # Submit analysis
    with row1_col2:
        # Custom CSS for submit button
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #28a745;
            color: white;
            border: 1px solid #28a745;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #218838;
            border-color: #1e7e34;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Analysis parameters summary
        st.subheader("📋 Analysis Summary")
        st.write(f"• **Index**: {selected_index} ({VIS_PARAMS[selected_index]['description']})")
        st.write(f"• **Region**: {selected_region}")
        st.write(f"• **Time Period**: {start_year}-{end_year} (every {delta_years} years)")
        st.write(f"• **Months**: {month_names[start_month-1]} to {month_names[end_month-1]}")
        st.write(f"• **Years to Process**: {len(years_to_process)}")
        
        # Submit button
        if st.button("Submit", type="primary"):
            roi = st.session_state.get("roi")
            if not roi and selected_region == "Uploaded GeoJSON":
                st.error("Please upload a GeoJSON file or select a region.")
                st.stop()
            
            if len(years_to_process) < 1:
                st.error("Need at least 1 year for analysis.")
                st.stop()
            
            # Clean up before analysis
            cleanup_temp_files()
            gc.collect()
            
            with st.spinner(f"Computing {selected_index} values... Please wait..."):
                try:
                    if selected_region == "Uploaded GeoJSON":
                        region_geometry = roi
                        region_name = "Custom Region"
                    else:
                        region_geometry = roi
                        region_name = selected_region
                    
                    results_df = analyze_region_vegetation_batch(
                        region_geometry, region_name, years_to_process, month_range, selected_index
                    )
                    
                    if not results_df.empty:
                        st.session_state["vegetation_results"] = results_df
                        st.session_state["analysis_params"] = {
                            "years_to_process": years_to_process,
                            "month_range": month_range,
                            "selected_region": selected_region,
                            "selected_index": selected_index
                        }
                        
                        st.success(f"✅ {selected_index} Analysis complete! {len(results_df)} region(s) processed.")
                        
                        # Auto-refresh to show choropleth for Florida
                        if selected_region == "Florida (All Counties)":
                            st.rerun()
                    else:
                        st.error("❌ Analysis returned no results. Please check your parameters and try again.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("Please try again with different parameters or check your Earth Engine authentication.")
                finally:
                    # Clean up after analysis
                    cleanup_temp_files()
                    gc.collect()

    # Display results
    if st.session_state["vegetation_results"] is not None:
        results_df = st.session_state["vegetation_results"]
        analysis_params = st.session_state.get("analysis_params", {})
        
        if not results_df.empty:
            selected_index = analysis_params.get("selected_index", "NDVI")
            vis_config = VIS_PARAMS[selected_index]
            
            st.header(f"🌱 {selected_index} Analysis Results")
            
            # Results summary
            region_count = len(results_df)
            years_analyzed = analysis_params.get("years_to_process", [])
            month_range = analysis_params.get("month_range", [1, 12])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Regions Analyzed", region_count)
            with col2:
                st.metric("Years Processed", len(years_analyzed))
            with col3:
                if len(years_analyzed) > 0:
                    st.metric("Year Range", f"{min(years_analyzed)}-{max(years_analyzed)}")
            with col4:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_str = f"{month_names[month_range[0]-1]}-{month_names[month_range[1]-1]}"
                st.metric("Season", month_str)
            
            # Statistics overview
            if region_count > 1:
                index_columns = [col for col in results_df.columns if f'{selected_index}_' in col and col != 'County']
                if index_columns:
                    latest_index_col = index_columns[-1]  # Get the most recent year
                    valid_data = results_df[latest_index_col].dropna()
                    
                    if not valid_data.empty:
                        st.subheader(f"📊 {selected_index} Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(f"Average {selected_index}", f"{valid_data.mean():.4f}")
                        with col2:
                            st.metric("Highest Value", f"{valid_data.max():.4f}")
                        with col3:
                            st.metric("Lowest Value", f"{valid_data.min():.4f}")
                        with col4:
                            st.metric("Value Range", f"{valid_data.max() - valid_data.min():.4f}")
                        
                        # Show highest and lowest counties
                        if 'County' in results_df.columns:
                            try:
                                highest_idx = results_df[latest_index_col].idxmax()
                                lowest_idx = results_df[latest_index_col].idxmin()
                                highest_county = results_df.loc[highest_idx, 'County']
                                lowest_county = results_df.loc[lowest_idx, 'County']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"🌳 **{vis_config['rank_high'].split(' ')[0]} Highest**: {highest_county} ({valid_data.max():.4f})")
                                with col2:
                                    st.info(f"🍂 **{vis_config['rank_low'].split(' ')[0]} Lowest**: {lowest_county} ({valid_data.min():.4f})")
                            except Exception:
                                pass
            
            # Time series plot
            if len(years_analyzed) > 1:
                st.subheader(f"📈 {selected_index} Time Series")
                fig = create_vegetation_time_series_plot(results_df, selected_index)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trend analysis for single region
                if region_count == 1 and len(years_analyzed) > 2:
                    index_columns = [col for col in results_df.columns if f'{selected_index}_' in col]
                    if index_columns:
                        values = []
                        valid_years = []
                        
                        for col in index_columns:
                            val = results_df[col].iloc[0]
                            if pd.notna(val):
                                values.append(val)
                                year = int(col.split('_')[1])
                                valid_years.append(year)
                        
                        if len(values) > 2:
                            try:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, values)
                                
                                trend_direction = "increasing" if slope > 0 else "decreasing"
                                trend_rate = abs(slope)
                                
                                if selected_index == 'NDVI':
                                    trend_meaning = "improving vegetation health" if slope > 0 else "declining vegetation health"
                                else:  # EVI
                                    trend_meaning = "enhancing vegetation vigor" if slope > 0 else "reducing vegetation vigor"
                                
                                st.info(f"📊 **{selected_index} Trend:** {trend_direction.title()} at {trend_rate:.5f} per year ({trend_meaning}) (R² = {r_value**2:.3f})")
                            except Exception:
                                try:
                                    x_mean = np.mean(valid_years)
                                    y_mean = np.mean(values)
                                    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(valid_years, values)) / sum((x - x_mean)**2 for x in valid_years)
                                    
                                    trend_direction = "increasing" if slope > 0 else "decreasing"
                                    trend_rate = abs(slope)
                                    
                                    st.info(f"📊 **{selected_index} Trend:** {trend_direction.title()} at {trend_rate:.5f} per year")
                                except:
                                    pass
            
            # Results table
            st.subheader(f"📋 Detailed Results ({region_count} region{'s' if region_count != 1 else ''})")
            
            # Format display
            results_display = results_df.copy()
            index_columns = [col for col in results_display.columns if f'{selected_index}_' in col]
            
            # Color-code the vegetation index values
            def highlight_vegetation(val):
                try:
                    if pd.isna(val):
                        return ''
                    elif val > 0.7:
                        return 'background-color: darkgreen; color: white'
                    elif val > 0.5:
                        return 'background-color: lightgreen'
                    elif val > 0.3:
                        return 'background-color: lightyellow'
                    elif val > 0.1:
                        return 'background-color: lightcoral'
                    else:
                        return 'background-color: lightgray'
                except:
                    return ''
            
            try:
                styled_df = results_display.style.applymap(highlight_vegetation, subset=index_columns)
                st.dataframe(styled_df, use_container_width=True)
            except Exception:
                # Fallback to regular dataframe if styling fails
                st.dataframe(results_display, use_container_width=True)
            
            # Vegetation rankings for Florida-wide analysis
            if region_count > 1 and 'County' in results_df.columns and index_columns:
                st.subheader(f"🏆 County {selected_index} Rankings")
                
                # Use the latest year for rankings
                latest_index_col = index_columns[-1]
                latest_year = latest_index_col.split('_')[1]
                
                valid_counties = results_df[['County', latest_index_col]].dropna()
                
                if not valid_counties.empty:
                    highest_counties = valid_counties.nlargest(10, latest_index_col)
                    lowest_counties = valid_counties.nsmallest(10, latest_index_col)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{vis_config['rank_high']} ({latest_year})**")
                        highest_display = highest_counties.copy()
                        highest_display[latest_index_col] = highest_display[latest_index_col].round(4)
                        highest_display['Rank'] = range(1, len(highest_display) + 1)
                        st.dataframe(highest_display[['Rank', 'County', latest_index_col]], 
                                   hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**{vis_config['rank_low']} ({latest_year})**")
                        lowest_display = lowest_counties.copy()
                        lowest_display[latest_index_col] = lowest_display[latest_index_col].round(4)
                        lowest_display['Rank'] = range(1, len(lowest_display) + 1)
                        st.dataframe(lowest_display[['Rank', 'County', latest_index_col]], 
                                   hide_index=True, use_container_width=True)
            
            # Distribution histogram for multi-county analysis
            if region_count > 5 and index_columns:
                st.subheader(f"📊 {selected_index} Distribution")
                
                latest_index_col = index_columns[-1]
                latest_year = latest_index_col.split('_')[1]
                
                valid_data = results_df[latest_index_col].dropna()
                
                if not valid_data.empty:
                    try:
                        fig_hist = px.histogram(
                            x=valid_data,
                            nbins=20,
                            title=f"Distribution of {selected_index} across Florida Counties ({latest_year})",
                            labels={'x': f'{selected_index}', 'y': 'Number of Counties'}
                        )
                        fig_hist.update_layout(showlegend=False)
                        fig_hist.update_traces(marker_color='lightgreen')
                        st.plotly_chart(fig_hist, use_container_width=True)
                    except Exception:
                        st.warning("Could not create distribution histogram")
            
            # Download section
            st.subheader("💾 Download Results")
            
            # Prepare download data
            try:
                csv_data = results_df.to_csv(index=False)
                region_name = analysis_params.get("selected_region", "Vegetation_Analysis").replace(" ", "_")
                filename = f"{selected_index}_Results_{region_name}_{min(years_analyzed)}-{max(years_analyzed)}.csv"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                
                with col2:
                    if st.button("🔄 Clear Results"):
                        st.session_state["vegetation_results"] = None
                        st.session_state["analysis_params"] = {}
                        cleanup_temp_files()
                        st.rerun()
            except Exception as e:
                st.error(f"Could not prepare download: {e}")
        else:
            st.warning("No results to display. Please run an analysis first.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please refresh the page and try again. If the error persists, check your Google Earth Engine authentication.")
        # Clean up on error
        cleanup_temp_files()
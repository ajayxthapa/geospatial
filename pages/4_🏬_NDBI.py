# -----------------------------------------------------------------------------
# FLORIDA NORMALIZED DIFFERENCE BUILT-UP INDEX (NDBI) ANALYSIS STREAMLIT APP
# Interactive web app for analyzing urban development using Landsat satellite data
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
    page_title="Florida NDBI Analysis",
    layout="wide"
)
warnings.filterwarnings("ignore")

# Initialize session state
if "ndbi_results" not in st.session_state:
    st.session_state["ndbi_results"] = None
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
# FILE HANDLING UTILITIES (FIXED FOR ACCESS ISSUES)
# -----------------------------------------------------------------------------
def cleanup_temp_files():
    """Clean up temporary files safely"""
    if "temp_files" in st.session_state:
        for file_path in st.session_state["temp_files"]:
            try:
                if os.path.exists(file_path):
                    os.chmod(file_path, 0o777)  # Change permissions
                    time.sleep(0.1)  # Small delay
                    os.remove(file_path)
            except Exception as e:
                try:
                    # Try alternative cleanup
                    if os.path.exists(file_path):
                        shutil.move(file_path, file_path + ".deleted")
                except:
                    pass  # Ignore if we can't delete
        st.session_state["temp_files"] = []

def create_temp_file(data, extension):
    """Create temporary file with better handling"""
    try:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Use a custom temp directory
        temp_dir = tempfile.gettempdir()
        custom_temp_dir = os.path.join(temp_dir, "streamlit_ndbi_app")
        
        # Create directory if it doesn't exist
        os.makedirs(custom_temp_dir, exist_ok=True)
        
        file_path = os.path.join(custom_temp_dir, f"{file_id}{extension}")
        
        # Write file with explicit binary mode
        with open(file_path, "wb") as file:
            file.write(data.getbuffer())
            file.flush()
            os.fsync(file.fileno())  # Force write to disk
        
        # Add to cleanup list
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
        # Try to initialize with token first
        try:
            geemap.ee_initialize(token_name="EARTHENGINE_TOKEN")
            return True
        except:
            pass
        
        # Try standard initialization
        try:
            ee.Initialize()
            return True
        except:
            pass
        
        # Try authentication flow
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
# FLORIDA COUNTIES AND GEOMETRY FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        # Get file extension
        _, file_extension = os.path.splitext(data.name)
        
        # Create temporary file safely
        file_path = create_temp_file(data, file_extension)
        if not file_path:
            raise Exception("Could not create temporary file")
        
        # Wait a moment for file to be fully written
        time.sleep(0.1)
        
        # Read the file
        if file_path.lower().endswith(".kml"):
            fiona.drvsupport.supported_drivers["KML"] = "rw"
            gdf = gpd.read_file(file_path, driver="KML")
        else:
            gdf = gpd.read_file(file_path)
        
        # Ensure CRS is set
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        return gdf
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None
    finally:
        # Don't delete here - let the cleanup function handle it
        pass

# -----------------------------------------------------------------------------
# NDBI PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
# Define band names for different Landsat missions for NDBI calculation
BAND_INFO = {
    'L8_9': {'NIR': 'SR_B5', 'SWIR': 'SR_B6'},  # Landsat 8 & 9
    'L5_7': {'NIR': 'SR_B4', 'SWIR': 'SR_B5'}   # Landsat 5 & 7
}

# Visualization parameters for NDBI
VIS_PARAMS = {
    'min': -0.5, 'max': 0.5,
    'palette': ['blue', 'white', 'brown']
}

def mask_landsat_clouds(image):
    """Masks clouds and cloud shadows in Landsat Collection 2 images."""
    qa = image.select('QA_PIXEL')
    # Bits 3 (Cloud) and 5 (Cloud Shadow) are set to 0
    cloud_mask = (1 << 3) | (1 << 5)
    mask = qa.bitwiseAnd(cloud_mask).eq(0)
    return image.updateMask(mask)

def apply_scale_factors(image):
    """Applies scaling factors to optical (Surface Reflectance) bands."""
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical_bands, overwrite=True)

def calculate_ndbi(image):
    """Calculates NDBI for a given image after applying scale factors."""
    spacecraft = image.get('SPACECRAFT_ID')
    
    # Use condition to select bands based on the satellite
    bands = ee.Dictionary(ee.Algorithms.If(
        ee.List(['LANDSAT_8', 'LANDSAT_9']).contains(spacecraft),
        BAND_INFO['L8_9'],
        BAND_INFO['L5_7']
    ))
    
    scaled_image = apply_scale_factors(image)
    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    ndbi = scaled_image.normalizedDifference([
        bands.getString('SWIR'), 
        bands.getString('NIR')
    ]).rename('NDBI')
    
    return image.addBands(ndbi)

def get_mean_ndbi_for_year(year, months, geometry):
    """
    Calculates the mean NDBI for a given year, month range, and geometry.
    """
    start_date = ee.Date.fromYMD(year, months[0], 1)
    end_date = ee.Date.fromYMD(year, months[1], 1).advance(1, 'month').advance(-1, 'day')

    # Combine imagery from all relevant Landsat missions
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    
    landsat_collection = l9.merge(l8).merge(l7).merge(l5)

    # Filter, map, and reduce the image collection to a single mean NDBI image
    image_composite = (landsat_collection
                       .filterBounds(geometry)
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.calendarRange(months[0], months[1], 'month'))
                       .map(mask_landsat_clouds)
                       .map(calculate_ndbi)
                       .select('NDBI')
                       .mean())
                       
    return image_composite.set('year', year)

def generate_year_range(start_year, end_year, delta_years):
    """Generate year range based on start, end, and delta"""
    years = list(range(start_year, end_year + 1, delta_years))
    # Ensure end year is included if not already in the list
    if end_year not in years:
        years.append(end_year)
    return sorted(years)

def analyze_ndbi_for_county(feature, years_to_process, month_range):
    """Analyze NDBI for a single county feature"""
    geom = feature.geometry()
    
    properties = {
        'NAME': feature.get('NAME'),
    }
    
    for year in years_to_process:
        try:
            mean_ndbi_image = get_mean_ndbi_for_year(year, month_range, geom)
            stats = mean_ndbi_image.reduceRegions(
                collection=ee.FeatureCollection([feature]),
                reducer=ee.Reducer.mean(),
                scale=500  # Higher scale for NDBI analysis
            )
            
            mean_ndbi = stats.first().get('mean')
            properties[f'NDBI_{year}'] = mean_ndbi
            
        except Exception as e:
            properties[f'NDBI_{year}'] = None
    
    return feature.set(properties)

def analyze_region_ndbi_batch(region_geometry, region_name, years_to_process, month_range):
    """Analyze NDBI for a region using batch processing with better error handling"""
    
    try:
        if region_name == "Florida (All Counties)":
            counties_collection = ee.FeatureCollection("TIGER/2018/Counties")
            florida_counties = counties_collection.filter(ee.Filter.eq('STATEFP', '12'))
            
            county_list = florida_counties.toList(florida_counties.size())
            batch_size = 3  # Reduced batch size for stability
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
                    subset_results = subset.map(lambda f: analyze_ndbi_for_county(f, years_to_process, month_range)).getInfo()
                    all_features.extend(subset_results['features'])
                    
                    # Small delay to prevent overwhelming the API
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
                    ndbi_value = props.get(f'NDBI_{year}')
                    if ndbi_value is not None:
                        try:
                            row_data[f'NDBI_{year}'] = round(float(ndbi_value), 4)
                        except (ValueError, TypeError):
                            row_data[f'NDBI_{year}'] = None
                    else:
                        row_data[f'NDBI_{year}'] = None
                
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
                    mean_ndbi_image = get_mean_ndbi_for_year(year, month_range, geometry)
                    stats = mean_ndbi_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=500,
                        maxPixels=1e13
                    )
                    
                    mean_ndbi = stats.get('NDBI').getInfo()
                    if mean_ndbi is not None:
                        try:
                            results[f'NDBI_{year}'] = round(float(mean_ndbi), 4)
                        except (ValueError, TypeError):
                            results[f'NDBI_{year}'] = None
                    else:
                        results[f'NDBI_{year}'] = None
                        
                except Exception as year_error:
                    st.warning(f"Error processing year {year}: {year_error}")
                    results[f'NDBI_{year}'] = None
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            return pd.DataFrame([results])
            
    except Exception as e:
        st.error(f"Critical error in NDBI analysis: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def create_ndbi_choropleth_map(m, results_df, florida_counties_gdf, latest_year):
    """Create choropleth map with NDBI data"""
    if florida_counties_gdf is None or results_df is None or results_df.empty:
        return m
    
    try:
        # Find the latest NDBI column
        ndbi_columns = [col for col in results_df.columns if f'NDBI_{latest_year}' in col]
        
        if not ndbi_columns:
            return m
        
        latest_ndbi_col = ndbi_columns[0]
        
        # Merge results with geometry data
        merged_gdf = florida_counties_gdf.merge(
            results_df[['County', latest_ndbi_col]], 
            left_on='NAME', 
            right_on='County', 
            how='left'
        )
        
        # Fill NaN values with median for visualization
        valid_data = merged_gdf[latest_ndbi_col].dropna()
        if not valid_data.empty:
            merged_gdf[latest_ndbi_col] = merged_gdf[latest_ndbi_col].fillna(valid_data.median())
            
            # Create choropleth layer
            vmin = valid_data.min()
            vmax = valid_data.max()
            
            if vmax > vmin:
                # Add choropleth - Brown to Blue (high to low built-up)
                choropleth = folium.Choropleth(
                    geo_data=merged_gdf.to_json(),
                    name=f'NDBI Choropleth ({latest_year})',
                    data=merged_gdf,
                    columns=['NAME', latest_ndbi_col],
                    key_on='feature.properties.NAME',
                    fill_color='RdYlGn_r',  # Red-Yellow-Green reversed (built-up to natural)
                    fill_opacity=0.7,
                    line_opacity=0.5,
                    legend_name=f'NDBI {latest_year}',
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
                
                # Create GeoJson layer with tooltips
                geojson_layer = folium.GeoJson(
                    merged_gdf.to_json(),
                    name='County NDBI Information',
                    style_function=lambda x: {'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['NAME', latest_ndbi_col],
                        aliases=['County:', f'NDBI ({latest_year}):'],
                        localize=True, sticky=False, labels=True,
                        style=tooltip_style, max_width=200
                    )
                )
                geojson_layer.add_to(m)
                
                # Add layer control
                folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
    except Exception as e:
        st.warning(f"Could not create choropleth map: {e}")
    
    return m

def display_geometry_on_map(m, selected_region, florida_rois, florida_counties_gdf):
    """Display selected geometry on map with appropriate zoom"""
    
    try:
        if selected_region == "Florida (All Counties)":
            # Display all Florida counties
            if florida_counties_gdf is not None:
                # Add county boundaries
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
                
                # Zoom to Florida bounds
                bounds = florida_counties_gdf.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                
        elif selected_region != "Uploaded GeoJSON":
            # Display individual county or region
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

def create_ndbi_time_series_plot(results_df):
    """Create time series plot for NDBI data"""
    if results_df is None or results_df.empty:
        return None
    
    try:
        # Get NDBI columns
        ndbi_columns = [col for col in results_df.columns if 'NDBI_' in col]
        years = [int(col.split('_')[1]) for col in ndbi_columns]
        
        if len(ndbi_columns) == 0:
            return None
        
        fig = go.Figure()
        
        if len(results_df) == 1:
            # Single region - line plot
            ndbi_values = []
            for col in ndbi_columns:
                val = results_df[col].iloc[0]
                if pd.notna(val):
                    ndbi_values.append(val)
                else:
                    ndbi_values.append(None)
            
            fig.add_trace(go.Scatter(
                x=years,
                y=ndbi_values,
                mode='lines+markers',
                name=results_df['Region'].iloc[0] if 'Region' in results_df.columns else 'Single Region',
                line=dict(width=3),
                marker=dict(size=8),
                connectgaps=False
            ))
        else:
            # Multiple counties - show top 5 most and least built-up
            mean_ndbi = {}
            for _, row in results_df.iterrows():
                county = row['County']
                ndbi_values = [row[col] for col in ndbi_columns if pd.notna(row[col])]
                if ndbi_values:
                    mean_ndbi[county] = np.mean(ndbi_values)
            
            if mean_ndbi:
                sorted_counties = sorted(mean_ndbi.items(), key=lambda x: x[1], reverse=True)
                top_5_built = sorted_counties[:5]
                top_5_natural = sorted_counties[-5:]
                
                counties_to_plot = dict(top_5_built + top_5_natural)
                
                for county, _ in counties_to_plot.items():
                    county_row = results_df[results_df['County'] == county].iloc[0]
                    ndbi_values = []
                    for col in ndbi_columns:
                        val = county_row[col]
                        if pd.notna(val):
                            ndbi_values.append(val)
                        else:
                            ndbi_values.append(None)
                    
                    color = 'brown' if county in dict(top_5_built) else 'green'
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=ndbi_values,
                        mode='lines+markers',
                        name=county,
                        line=dict(width=2, color=color),
                        marker=dict(size=6),
                        connectgaps=False
                    ))
        
        fig.update_layout(
            title="Normalized Difference Built-up Index (NDBI) Time Series",
            xaxis_title="Year",
            yaxis_title="NDBI",
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
        - Web App URL: [floridandbi.streamlit.app](https://floridandbi.streamlit.app)  
        - GitHub repository: [github.com/ajayxthapa/floridandbi](https://github.com/ajayxthapa/floridandbi)
        """
    )

    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        Ajay Thapa  
        📧 [Email](mailto:athapa2024@fau.edu)  
        💻 [GitHub](https://github.com/ajayxthapa)
        """
    )
    
    # Main content
    st.title("Florida Normalized Difference Built-up Index (NDBI) Analysis")
    st.markdown(
        """
        An interactive web app for analyzing urban development and built-up areas using 
        [Landsat satellite data](https://developers.google.com/earth-engine/datasets/catalog/landsat) from Google Earth Engine. 
        
        **Normalized Difference Built-up Index (NDBI)** is calculated as **(SWIR - NIR) / (SWIR + NIR)**, 
        where higher values indicate more built-up/urban areas and lower values indicate vegetation or natural areas.
        
        **Data Source:** Landsat Collection 2 Surface Reflectance (Landsat 5, 7, 8, and 9)  
        **Processing:** Cloud masking, surface reflectance scaling, NDBI calculation, temporal aggregation
        
        **NDBI Interpretation:**
        - **Positive values (0 to 1):** Built-up areas, urban development, bare soil
        - **Negative values (-1 to 0):** Vegetation, water bodies, natural areas
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
            start_month = st.selectbox("Start Month:", range(1, 13), index=3, 
                                     format_func=lambda x: month_names[x-1])  # Default April
        with col2:
            end_month = st.selectbox("End Month:", range(1, 13), index=8, 
                                   format_func=lambda x: month_names[x-1])  # Default September
        
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
            if (st.session_state["ndbi_results"] is not None and 
                selected_region == "Florida (All Counties)" and
                st.session_state["analysis_params"]):
                latest_year = max(st.session_state["analysis_params"]["years_to_process"])
                m = create_ndbi_choropleth_map(
                    m, 
                    st.session_state["ndbi_results"], 
                    florida_counties_gdf, 
                    latest_year
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
            background-color: #f0f2f6;
            color: #262730;
            border: 1px solid #d1d5db;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #e5e7eb;
            border-color: #9ca3af;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Analysis parameters summary
        st.markdown("**Analysis Summary:**")
        st.write(f"• Region: {selected_region}")
        st.write(f"• Time Period: {start_year}-{end_year} (every {delta_years} years)")
        st.write(f"• Months: {month_names[start_month-1]} to {month_names[end_month-1]}")
        st.write(f"• Years to Process: {len(years_to_process)}")
        
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
            gc.collect()  # Force garbage collection
            
            with st.spinner("Computing NDBI... Please wait..."):
                try:
                    if selected_region == "Uploaded GeoJSON":
                        region_geometry = roi
                        region_name = "Custom Region"
                    else:
                        region_geometry = roi
                        region_name = selected_region
                    
                    results_df = analyze_region_ndbi_batch(
                        region_geometry, region_name, years_to_process, month_range
                    )
                    
                    if not results_df.empty:
                        st.session_state["ndbi_results"] = results_df
                        st.session_state["analysis_params"] = {
                            "years_to_process": years_to_process,
                            "month_range": month_range,
                            "selected_region": selected_region
                        }
                        
                        st.success(f"✅ NDBI Analysis complete! {len(results_df)} region(s) processed.")
                        
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
    if st.session_state["ndbi_results"] is not None:
        results_df = st.session_state["ndbi_results"]
        analysis_params = st.session_state.get("analysis_params", {})
        
        if not results_df.empty:
            st.header("🏙️ NDBI Analysis Results")
            
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
                ndbi_columns = [col for col in results_df.columns if 'NDBI_' in col]
                if ndbi_columns:
                    latest_ndbi_col = ndbi_columns[-1]  # Get the most recent year
                    valid_data = results_df[latest_ndbi_col].dropna()
                    
                    if not valid_data.empty:
                        st.subheader("📊 NDBI Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Average NDBI", f"{valid_data.mean():.4f}")
                        with col2:
                            st.metric("Most Built-up", f"{valid_data.max():.4f}")
                        with col3:
                            st.metric("Least Built-up", f"{valid_data.min():.4f}")
                        with col4:
                            st.metric("NDBI Range", f"{valid_data.max() - valid_data.min():.4f}")
                        
                        # Show most and least built-up counties
                        if 'County' in results_df.columns:
                            try:
                                most_built_idx = results_df[latest_ndbi_col].idxmax()
                                least_built_idx = results_df[latest_ndbi_col].idxmin()
                                most_built_county = results_df.loc[most_built_idx, 'County']
                                least_built_county = results_df.loc[least_built_idx, 'County']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"🏙️ **Most Built-up:** {most_built_county} ({valid_data.max():.4f})")
                                with col2:
                                    st.info(f"🌿 **Least Built-up:** {least_built_county} ({valid_data.min():.4f})")
                            except Exception as e:
                                st.warning("Could not identify most/least built-up counties")
            
            # Time series plot
            if len(years_analyzed) > 1:
                st.subheader("📈 NDBI Time Series")
                fig = create_ndbi_time_series_plot(results_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Urban development trend analysis
                if region_count == 1 and len(years_analyzed) > 2:
                    ndbi_columns = [col for col in results_df.columns if 'NDBI_' in col]
                    if ndbi_columns:
                        ndbi_values = []
                        valid_years = []
                        
                        for col in ndbi_columns:
                            ndbi_val = results_df[col].iloc[0]
                            if pd.notna(ndbi_val):
                                ndbi_values.append(ndbi_val)
                                year = int(col.split('_')[1])
                                valid_years.append(year)
                        
                        if len(ndbi_values) > 2:
                            # Calculate linear trend
                            try:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_years, ndbi_values)
                                
                                trend_direction = "increasing urbanization" if slope > 0 else "decreasing urbanization"
                                trend_rate = abs(slope)
                                
                                st.info(f"📊 **Development Trend:** {trend_direction.title()} at {trend_rate:.6f} NDBI units/year (R² = {r_value**2:.3f})")
                            except Exception as e:
                                # Fallback to simple slope calculation
                                try:
                                    x_mean = np.mean(valid_years)
                                    y_mean = np.mean(ndbi_values)
                                    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(valid_years, ndbi_values)) / sum((x - x_mean)**2 for x in valid_years)
                                    
                                    trend_direction = "increasing urbanization" if slope > 0 else "decreasing urbanization"
                                    trend_rate = abs(slope)
                                    
                                    st.info(f"📊 **Development Trend:** {trend_direction.title()} at {trend_rate:.6f} NDBI units/year")
                                except:
                                    pass
            
            # Results table
            st.subheader(f"📋 Detailed Results ({region_count} region{'s' if region_count != 1 else ''})")
            
            # Format display
            results_display = results_df.copy()
            ndbi_columns = [col for col in results_display.columns if 'NDBI_' in col]
            
            # Color-code the NDBI values
            def highlight_ndbi(val):
                try:
                    if pd.isna(val):
                        return ''
                    elif val > 0.2:
                        return 'background-color: lightcoral'  # High built-up
                    elif val > 0:
                        return 'background-color: lightyellow'  # Moderate built-up
                    elif val > -0.2:
                        return 'background-color: lightgreen'  # Mixed/sparse vegetation
                    else:
                        return 'background-color: lightblue'   # Dense vegetation/water
                except:
                    return ''
            
            try:
                styled_df = results_display.style.applymap(highlight_ndbi, subset=ndbi_columns)
                st.dataframe(styled_df, use_container_width=True)
            except Exception as e:
                # Fallback to regular dataframe if styling fails
                st.dataframe(results_display, use_container_width=True)
            
            # NDBI rankings for Florida-wide analysis
            if region_count > 1 and 'County' in results_df.columns and ndbi_columns:
                st.subheader("🏆 County Development Rankings")
                
                # Use the latest year for rankings
                latest_ndbi_col = ndbi_columns[-1]
                latest_year = latest_ndbi_col.split('_')[1]
                
                valid_counties = results_df[['County', latest_ndbi_col]].dropna()
                
                if not valid_counties.empty:
                    most_built_counties = valid_counties.nlargest(10, latest_ndbi_col)
                    least_built_counties = valid_counties.nsmallest(10, latest_ndbi_col)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**🏙️ Top 10 Most Built-up Counties ({latest_year})**")
                        most_built_display = most_built_counties.copy()
                        most_built_display[latest_ndbi_col] = most_built_display[latest_ndbi_col].round(4)
                        most_built_display['Rank'] = range(1, len(most_built_display) + 1)
                        st.dataframe(most_built_display[['Rank', 'County', latest_ndbi_col]], 
                                   hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**🌿 Top 10 Most Natural Counties ({latest_year})**")
                        least_built_display = least_built_counties.copy()
                        least_built_display[latest_ndbi_col] = least_built_display[latest_ndbi_col].round(4)
                        least_built_display['Rank'] = range(1, len(least_built_display) + 1)
                        st.dataframe(least_built_display[['Rank', 'County', latest_ndbi_col]], 
                                   hide_index=True, use_container_width=True)
            
            # Distribution histogram for multi-county analysis
            if region_count > 5 and ndbi_columns:
                st.subheader("📊 NDBI Distribution")
                
                latest_ndbi_col = ndbi_columns[-1]
                latest_year = latest_ndbi_col.split('_')[1]
                
                valid_data = results_df[latest_ndbi_col].dropna()
                
                if not valid_data.empty:
                    try:
                        fig_hist = px.histogram(
                            x=valid_data,
                            nbins=20,
                            title=f"Distribution of NDBI across Florida Counties ({latest_year})",
                            labels={'x': 'NDBI', 'y': 'Number of Counties'},
                            color_discrete_sequence=['brown']
                        )
                        fig_hist.update_layout(showlegend=False)
                        
                        # Add vertical lines for interpretation
                        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                                         annotation_text="Built-up threshold")
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Add interpretation guide
                        st.markdown("""
                        **NDBI Interpretation Guide:**
                        - **> 0.2:** Highly built-up areas (urban centers, commercial districts)
                        - **0 to 0.2:** Moderate development (suburban areas, mixed land use)
                        - **-0.2 to 0:** Sparse development with vegetation
                        - **< -0.2:** Natural areas (forests, water bodies, dense vegetation)
                        """)
                        
                    except Exception as e:
                        st.warning("Could not create distribution histogram")
            
            # Download section
            st.subheader("💾 Download Results")
            
            # Prepare download data
            try:
                csv_data = results_df.to_csv(index=False)
                region_name = analysis_params.get("selected_region", "NDBI_Analysis").replace(" ", "_")
                filename = f"NDBI_Results_{region_name}_{min(years_analyzed)}-{max(years_analyzed)}.csv"
                
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
                        st.session_state["ndbi_results"] = None
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
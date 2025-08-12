import ee
import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import geemap.foliumap as geemap
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
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

# Configure Streamlit page
st.set_page_config(
    page_title="Urban Expansion Intensity Index (UEII) Calculator",
    layout="wide"
)
warnings.filterwarnings("ignore")

# Initialize session state
if "ueii_results" not in st.session_state:
    st.session_state["ueii_results"] = None
if "roi" not in st.session_state:
    st.session_state["roi"] = None
if "current_map_roi" not in st.session_state:
    st.session_state["current_map_roi"] = None
if "zoom_level" not in st.session_state:
    st.session_state["zoom_level"] = 4
if "year_intervals" not in st.session_state:
    st.session_state["year_intervals"] = []
if "florida_counties_gdf" not in st.session_state:
    st.session_state["florida_counties_gdf"] = None
if "selected_region_changed" not in st.session_state:
    st.session_state["selected_region_changed"] = False

@st.cache_data
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    """Initialize Earth Engine authentication"""
    try:
        geemap.ee_initialize(token_name=token_name)
    except Exception as e:
        try:
            ee.Authenticate()
            ee.Initialize()
        except Exception as auth_error:
            st.error(f"Earth Engine authentication failed: {auth_error}")
            st.stop()

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def uploaded_file_to_gdf(data):
    """Convert uploaded file to GeoDataFrame"""
    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf

def generate_year_intervals(start_year, end_year, delta_years):
    """Generate year intervals based on start, end, and delta"""
    intervals = []
    current_year = start_year
    
    while current_year < end_year:
        next_year = min(current_year + delta_years, end_year)
        intervals.append((current_year, next_year))
        current_year = next_year
        
        if current_year >= end_year:
            break
    
    return intervals

def get_available_nlcd_years():
    """Get available NLCD years"""
    return [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

def filter_years_for_intervals(intervals, available_years):
    """Filter intervals to only include years with NLCD data"""
    filtered_intervals = []
    
    for start_year, end_year in intervals:
        start_available = min(available_years, key=lambda x: abs(x - start_year))
        end_available = min(available_years, key=lambda x: abs(x - end_year))
        
        if (start_available != end_available and 
            abs(start_available - start_year) <= 3 and 
            abs(end_available - end_year) <= 3):
            filtered_intervals.append((start_available, end_available))
    
    return list(set(filtered_intervals))

def calculate_developed_area(image, geometry, scale=30):
    """Calculate developed area from NLCD image"""
    developed_classes = [21, 22, 23, 24]
    sqm_to_sqmi = 3.86102159e-7
    
    mask = image.remap(developed_classes, [1] * len(developed_classes), 0).selfMask()
    area_image = mask.multiply(ee.Image.pixelArea()).rename('area')
    
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13
    )
    
    return ee.Number(stats.get('area')).multiply(sqm_to_sqmi)

def compute_ueii(area_t1, area_t2, delta_t, total_area):
    """Compute Urban Expansion Intensity Index"""
    return (ee.Number(area_t2).subtract(area_t1)
            .divide(ee.Number(total_area).multiply(delta_t))).multiply(100)

def analyze_county(feature, year_intervals, nlcd_collection):
    """Analyze UEII for a single county feature"""
    sqm_to_sqmi = 3.86102159e-7
    
    geom = feature.geometry()
    county_area_sqmi = ee.Number(geom.area(maxError=1)).multiply(sqm_to_sqmi)
    
    unique_years = sorted(list(set([year for interval in year_intervals for year in interval])))
    dev_areas = {}
    
    for year in unique_years:
        try:
            image = nlcd_collection.filter(ee.Filter.eq('year', year)).first().select('b1')
            dev_areas[year] = calculate_developed_area(image, geom)
        except:
            continue
    
    properties = {
        'County Area (sq mi)': county_area_sqmi,
    }
    
    for year in unique_years:
        if year in dev_areas:
            properties[f'Developed Area {year} (sq mi)'] = dev_areas[year]
    
    for start_year, end_year in year_intervals:
        if start_year in dev_areas and end_year in dev_areas:
            delta_t = end_year - start_year
            ueii_value = compute_ueii(
                dev_areas[start_year], 
                dev_areas[end_year], 
                delta_t, 
                county_area_sqmi
            )
            properties[f'UEII {start_year}-{end_year} (%)'] = ueii_value
    
    return feature.set(properties)

def analyze_region_batch(region_geometry, region_name, year_intervals):
    """Analyze UEII for a region using batch processing"""
    
    nlcd_collection = ee.ImageCollection("projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER")
    
    if region_name == "Florida (All Counties)":
        counties_collection = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties_collection.filter(ee.Filter.eq('STATEFP', '12'))
        
        county_list = florida_counties.toList(florida_counties.size())
        batch_size = 5
        n_counties = florida_counties.size().getInfo()
        
        all_features = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for start in range(0, n_counties, batch_size):
            batch_progress = (start + batch_size) / n_counties
            progress_bar.progress(min(batch_progress, 1.0))
            status_text.text(f"Processing counties {start + 1}-{min(start + batch_size, n_counties)} of {n_counties}...")
            
            subset = ee.FeatureCollection(county_list.slice(start, start + batch_size))
            subset_results = subset.map(lambda f: analyze_county(f, year_intervals, nlcd_collection)).getInfo()
            all_features.extend(subset_results['features'])
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        data_for_df = []
        for i, feature in enumerate(all_features):
            props = feature['properties']
            row_data = {'S.N.': i + 1, 'Region': props['NAME']}
            row_data.update(props)
            data_for_df.append(row_data)
        
        return pd.DataFrame(data_for_df)
    
    else:
        sqm_to_sqmi = 3.86102159e-7
        
        if hasattr(region_geometry, 'geometry'):
            if hasattr(region_geometry, 'first'):
                geometry = region_geometry.first().geometry()
            else:
                geometry = region_geometry.geometry()
        else:
            geometry = region_geometry
        
        region_area_sqmi = ee.Number(geometry.area(maxError=1)).multiply(sqm_to_sqmi)
        
        unique_years = sorted(list(set([year for interval in year_intervals for year in interval])))
        developed_areas = {}
        
        for year in unique_years:
            try:
                image = nlcd_collection.filter(ee.Filter.eq('year', year)).first().select('b1')
                developed_areas[year] = calculate_developed_area(image, geometry)
            except:
                continue
        
        results = {
            'S.N.': 1,
            'Region': region_name,
            'Region Area (sq mi)': region_area_sqmi.getInfo()
        }
        
        for year in unique_years:
            if year in developed_areas:
                results[f'Developed Area {year} (sq mi)'] = developed_areas[year].getInfo()
        
        for start_year, end_year in year_intervals:
            if start_year in developed_areas and end_year in developed_areas:
                delta_t = end_year - start_year
                ueii_value = compute_ueii(
                    developed_areas[start_year], 
                    developed_areas[end_year], 
                    delta_t, 
                    region_area_sqmi
                ).getInfo()
                
                results[f'UEII {start_year}-{end_year} (%)'] = ueii_value
        
        return pd.DataFrame([results])

def create_choropleth_map(m, results_df, florida_counties_gdf, year_intervals):
    """Create choropleth map with UEII data"""
    if florida_counties_gdf is None or results_df is None:
        return m
    
    # Get the latest UEII column
    ueii_columns = []
    sorted_intervals = sorted(year_intervals)
    for start_year, end_year in sorted_intervals:
        col_name = f'UEII {start_year}-{end_year} (%)'
        if col_name in results_df.columns:
            ueii_columns.append(col_name)
    
    if not ueii_columns:
        return m
    
    latest_ueii_col = ueii_columns[-1]
    period_name = latest_ueii_col.replace('UEII ', '').replace(' (%)', '')
    
    # Merge results with geometry data
    merged_gdf = florida_counties_gdf.merge(
        results_df[['Region', latest_ueii_col]], 
        left_on='NAME', 
        right_on='Region', 
        how='left'
    )
    
    # Fill NaN values with 0 for visualization
    merged_gdf[latest_ueii_col] = merged_gdf[latest_ueii_col].fillna(0)
    
    # Create choropleth layer
    vmin = merged_gdf[latest_ueii_col].min()
    vmax = merged_gdf[latest_ueii_col].max()
    
    if vmax > vmin:
        # Add choropleth
        choropleth = folium.Choropleth(
            geo_data=merged_gdf.to_json(),
            name=f'UEII Choropleth ({period_name})',
            data=merged_gdf,
            columns=['NAME', latest_ueii_col],
            key_on='feature.properties.NAME',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.5,
            legend_name=f'UEII {period_name} (%)',
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
            name='County Information',
            style_function=lambda x: {'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(
                fields=['NAME', latest_ueii_col],
                aliases=['County:', f'UEII ({period_name}):'],
                localize=True, sticky=False, labels=True,
                style=tooltip_style, max_width=200
            )
        )
        geojson_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    return m

def display_geometry_on_map(m, selected_region, florida_rois, florida_counties_gdf):
    """Display selected geometry on map with appropriate zoom"""
    
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
                st.error(f"Error displaying geometry: {e}")
    
    return m

def main():
    # Sidebar
    # Sidebar content
    st.sidebar.title("About")
    st.sidebar.info(
        """
        - Web App URL: [urbanheatisland.streamlit.app](https://urbanheatisland.streamlit.app)  
        - GitHub repository: [github.com/ajayxthapa/urbanheatisland](https://github.com/ajayxthapa/urbanheatisland)
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
    st.title("Urban Expansion Intensity Index (UEII) Analysis")
    st.markdown(
        """
        An interactive web app for calculating the Urban Expansion Intensity Index (UEII) using 
        [USGS NLCD](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2019_REL_NLCD) data from Google Earth Engine. 
        The app was built using [streamlit](https://streamlit.io), [geemap](https://geemap.org), and [Google Earth Engine](https://earthengine.google.com).
        \n
        The **Urban Expansion Intensity Index (UEII)** measures the rate of urban 
        development relative to the total area of a region over a specific time period.
        
        **Formula:** $UEII = \\frac{Area_{t2} - Area_{t1}}{Total\\_Area \\times \\Delta T} \\times 100$
        
        **Data Source:** USGS Annual NLCD Landcover
        """
    )
    
    # Initialize Earth Engine
    ee_authenticate(token_name="EARTHENGINE_TOKEN")
    
    # Get Florida ROIs and counties GDF
    florida_rois = get_florida_counties()
    if st.session_state["florida_counties_gdf"] is None:
        st.session_state["florida_counties_gdf"] = get_florida_counties_gdf()
    florida_counties_gdf = st.session_state["florida_counties_gdf"]
    
    # Create layout
    row1_col1, row1_col2 = st.columns([2, 1])

    # Initialize map
    with row1_col1:
        m = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=True,
            Draw_export=True,
            locate_control=True,
            plugin_LatLngPopup=False,
        )
        m.add_basemap("ROADMAP")

    with row1_col2:
        # Search for location section
        keyword = st.text_input("Search for a location:", "")
        if keyword:
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
        
        # Region selection with automatic geometry display
        roi_options = ["Uploaded GeoJSON"] + list(florida_rois.keys())
        
        selected_region = st.selectbox(
            "Select a sample ROI or upload a GeoJSON file:",
            roi_options,
            key="roi_selectbox"
        )
        
        # Time period inputs in a clean layout
        available_years = get_available_nlcd_years()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start_year = st.selectbox("Start Year:", available_years, index=0)
        with col2:
            end_year = st.selectbox("End Year:", available_years, index=len(available_years) - 1)
        with col3:
            delta_years = st.selectbox("Delta (years):", list(range(1, 21)), index=9)
        
        if start_year >= end_year:
            st.error("End year must be after start year!")
            st.stop()
        
        # Generate intervals
        year_intervals = generate_year_intervals(start_year, end_year, delta_years)
        filtered_intervals = filter_years_for_intervals(year_intervals, available_years)
        
        if len(filtered_intervals) < 1:
            st.warning("No valid intervals with available NLCD data.")

    # File upload and map display
    with row1_col1:
        with st.expander("Upload Custom Region", False):
            st.info("Draw a rectangle on the map → Export as GeoJSON → Upload below → Click Submit")
            data = st.file_uploader(
                "Upload a GeoJSON file:", 
                type=["geojson", "kml", "zip"],
                help="Max 200MB • Supported: GeoJSON, KML, ZIP"
            )
        
        # This logic now correctly handles drawing the base map and overlaying the choropleth.
        if selected_region == "Uploaded GeoJSON":
            if data:
                try:
                    gdf = uploaded_file_to_gdf(data)
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
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.info("Upload a GeoJSON file above")
        else:
            # This handles all pre-selected Florida regions (individual counties or all).
            geometry = florida_rois.get(selected_region)
            if geometry:
                st.session_state["roi"] = geometry
                # Always display the base geometry. This ensures that on a rerun (after analysis), 
                # the base map with county outlines is drawn before the choropleth layer is added.
                m = display_geometry_on_map(m, selected_region, florida_rois, florida_counties_gdf)
        
        # Add choropleth overlay if results exist for the "All Counties" analysis.
        # This now correctly draws ON TOP of the base layer created above.
        if (st.session_state["ueii_results"] is not None and 
            selected_region == "Florida (All Counties)"):
            m = create_choropleth_map(
                m, 
                st.session_state["ueii_results"], 
                florida_counties_gdf, 
                st.session_state["year_intervals"]
            )
        
        m.to_streamlit(height=600)

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
        
        # Simple submit button
        if st.button("Submit", type="primary"):
            roi = st.session_state.get("roi")
            if not roi and selected_region == "Uploaded GeoJSON":
                st.error("Please upload a GeoJSON file or select a region.")
                st.stop()
            
            if len(filtered_intervals) < 1:
                st.error("Need at least 1 valid time interval for analysis.")
                st.stop()
            
            with st.spinner("Computing UEII... Please wait..."):
                try:
                    if selected_region == "Uploaded GeoJSON":
                        region_geometry = roi
                        region_name = "Custom Region"
                    else:
                        region_geometry = roi
                        region_name = selected_region
                    
                    results_df = analyze_region_batch(region_geometry, region_name, filtered_intervals)
                    
                    st.session_state["ueii_results"] = results_df
                    st.session_state["year_intervals"] = filtered_intervals
                    
                    st.success(f"✅ Analysis complete! {len(results_df)} region(s) processed.")
                    
                    # Auto-refresh to show choropleth
                    if selected_region == "Florida (All Counties)":
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # Display results
    if st.session_state["ueii_results"] is not None:
        results_df = st.session_state["ueii_results"]
        
        st.header("📋 Results")
        
        # Results table
        region_count = len(results_df)
        st.subheader(f"📊 Detailed Results ({region_count} region{'s' if region_count != 1 else ''})")
        
        # Format display
        results_display = results_df.copy()
        numeric_columns = results_display.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if 'Area' in col or 'UEII' in col:
                results_display[col] = results_display[col].round(4)
        
        st.dataframe(results_display, use_container_width=True)
        
        # Download
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"ueii_results_{selected_region.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please refresh the page and try again. If the error persists, check your Google Earth Engine authentication.")
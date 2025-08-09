import streamlit as st
import ee
import geemap
import geemap.foliumap as geemap
import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import shape
import fiona
import os
import tempfile
import warnings
from streamlit_folium import st_folium
import json

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(layout="wide", page_title="Florida UEII Analyzer")
warnings.filterwarnings("ignore")
st.markdown("""
    <style>
    .export-button-container {
        position: absolute;
        top: 15px;
        right: 15px;
        z-index: 9999;
    }
    .export-button-container button {
        background-color: #4CAF50;
        color: white;
        padding: 6px 12px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------
# GEE Authentication (Cached)
# ----------------------
@st.cache_data
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    """Initializes the Earth Engine API."""
    geemap.ee_initialize(token_name=token_name)

# ----------------------
# Sidebar Content
# ----------------------
with st.sidebar:
    st.title("Contact")
    st.success("""
        **Author:** Ajay Thapa
        - **Email:** [athapa2024@fau.edu](mailto:athapa2024@fau.edu)
        - **GitHub:** [ajayxthapa](https://github.com/ajayxthapa)
    """)

# ----------------------
# GEE Data & Parameters
# ----------------------
NLCD_ID = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER"
DEVELOPED_CLASSES = [21, 22, 23, 24]
SCALE = 30
SQM_TO_SQMI = 3.86102159e-7
COUNTY_FEATURE_COLLECTION = "TIGER/2018/Counties"
STATE_FIPS = "12"

# ----------------------
# GEE Helper Functions
# ----------------------
@st.cache_data
def get_nlcd_image(year):
    """Gets the NLCD land cover image for a specific year."""
    image = ee.ImageCollection(NLCD_ID).filter(ee.Filter.eq('year', year)).first()
    return image.select('b1') if image else ee.Image().rename('b1')

@st.cache_data
def get_florida_rois():
    """Fetches Florida state and county boundaries from GEE and caches them."""
    try:
        states = ee.FeatureCollection("TIGER/2018/States")
        florida_state = states.filter(ee.Filter.eq('NAME', 'Florida')).first()
        counties = ee.FeatureCollection(COUNTY_FEATURE_COLLECTION)
        florida_counties = counties.filter(ee.Filter.eq('STATEFP', STATE_FIPS))
        rois = {"Florida State": florida_state.geometry().getInfo()}
        county_features = florida_counties.getInfo()['features']
        for feature in sorted(county_features, key=lambda f: f['properties']['NAME']):
            county_name = feature['properties']['NAME']
            rois[f"{county_name} County"] = feature['geometry']
        return rois
    except Exception as e:
        st.error(f"Could not load Florida boundaries from GEE: {e}")
        return {"Florida State": ee.Geometry.Rectangle([-87.6, 24.4, -79.9, 31.0]).getInfo()}

def calculate_developed_area(image, geometry):
    """Calculates developed area in square miles."""
    developed_mask = image.remap(DEVELOPED_CLASSES, [1] * len(DEVELOPED_CLASSES), 0).selfMask()
    area_image = developed_mask.multiply(ee.Image.pixelArea()).rename("area")
    stats = area_image.reduceRegion(reducer=ee.Reducer.sum(), geometry=geometry, scale=SCALE, maxPixels=1e13)
    return ee.Number(stats.get("area", 0)).multiply(SQM_TO_SQMI)

def compute_ueii_for_feature(feature, start_year, end_year):
    """Computes UEII for a single GEE Feature."""
    geom = feature.geometry()
    delta_t = end_year - start_year
    total_area_sqmi = ee.Number(geom.area(maxError=1)).multiply(SQM_TO_SQMI)
    img_t1, img_t2 = get_nlcd_image(start_year), get_nlcd_image(end_year)
    area_t1, area_t2 = calculate_developed_area(img_t1, geom), calculate_developed_area(img_t2, geom)
    ueii = ee.Number(area_t2).subtract(area_t1).divide(total_area_sqmi.multiply(delta_t)).multiply(100)
    return feature.set({'UEII': ueii, 'NAME': feature.get('NAME')})

@st.cache_data
def uploaded_file_to_gdf(data):
    """Converts an uploaded file to a GeoDataFrame."""
    try:
        _, file_extension = os.path.splitext(data.name)
        file_id = os.urandom(8).hex()
        file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")
        with open(file_path, "wb") as file:
            file.write(data.getbuffer())
        if file_path.lower().endswith(".kml"):
            fiona.drvsupport.supported_drivers["KML"] = "rw"
            gdf = gpd.read_file(file_path, driver="KML")
        else:
            gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        return None

# ----------------------
# APP LAYOUT
# ----------------------
st.title("Urban Expansion Intensity Index (UEII) Analyzer")

st.info("This app analyzes urban expansion using the UEII, calculated with NLCD data via Google Earth Engine.")

with st.expander("What is the Urban Expansion Intensity Index (UEII)?"):
    st.markdown("""
        The UEII measures the intensity of urban land growth over time. It quantifies the percentage of new urban land developed annually relative to a region's total area.
        **Formula:** $$ UEII = \\frac{UA_{t2} - UA_{t1}}{TotalArea \\times \\Delta t} \\times 100 $$
    """)
st.divider()

ee_authenticate()

col_map, col_controls = st.columns([0.7, 0.3])

m = geemap.Map(center=[28.5, -83.5], zoom=6, plugin_Draw=True, draw_export=True)
m.add_basemap("HYBRID")

with col_controls:
    st.header("Control Panel")

    search_term = st.text_input("Search for a location:")
    if search_term:
        search_roi = geemap.geocode_to_ee(search_term)
        if search_roi is not None:
            m.center_object(search_roi, zoom=10)

    florida_rois = get_florida_rois()
    # MODIFIED: Removed "Drawn on map" from the options list
    roi_options = ["Uploaded GeoJSON"] + list(florida_rois.keys())
    
    # Use session state to manage the dropdown's selection index
    if 'roi_selection_index' not in st.session_state:
        st.session_state.roi_selection_index = 1 # Default to "Florida State"

    with st.form("control_form"):
        # The dropdown's index is now controlled by session state
        selected_roi_name = st.selectbox("Select a Region of Interest (ROI)", roi_options, index=st.session_state.roi_selection_index)
        
        time_period_options = {
            "1985-1995": (1985, 1995), "1995-2005": (1995, 2005),
            "2005-2015": (2005, 2015), "2015-2022": (2015, 2022)
        }
        selected_period = st.selectbox("Select Time Period", list(time_period_options.keys()))
        start_year, end_year = time_period_options[selected_period]
        opacity = st.slider("Map Layer Opacity", 0.0, 1.0, 0.7, 0.05)
        submitted = st.form_submit_button("Generate Map")

with col_map:
    st.header("Interactive Map")
    
    uploaded_file = st.file_uploader("Upload a GeoJSON, KML, or Shapefile (zipped)", type=["geojson", "kml", "zip"])

    if uploaded_file:
        gdf = uploaded_file_to_gdf(uploaded_file)
        if gdf is not None:
            st.session_state['custom_gdf'] = gdf
            st.session_state.roi_selection_index = 0  # Set dropdown to "Uploaded GeoJSON"
            m.add_gdf(gdf, layer_name="Uploaded ROI")
            m.center_object(geemap.gdf_to_ee(gdf), zoom=10)

   
     # ✅ Export button inside the map panel when a geometry is drawn
    if st.session_state.get("custom_gdf_drawn") is not None:
        st.markdown("### Export Drawn Geometry")
        drawn_gdf = st.session_state["custom_gdf_drawn"]
        geojson_str = drawn_gdf.to_json()
        st.download_button(
            label="📤 Export as GeoJSON",
            data=geojson_str,
            file_name="drawn_roi.geojson",
            mime="application/json",
            use_container_width=True
    )


    if 'final_map' in st.session_state and not submitted:
         map_data = st_folium(st.session_state['final_map'], use_container_width=True, key="final_map_rerun")
    else:
         map_data = st_folium(m, use_container_width=True, height=600, key="base_map")

    # When a user draws, update the session state to make it the active ROI
    if map_data.get('all_drawings'):
        drawn_geojson = map_data['all_drawings'][0]
        # Convert the drawn GeoJSON dict to a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features([drawn_geojson])
        st.session_state['custom_gdf'] = gdf
        st.session_state['custom_gdf_drawn'] = gdf # For the export button
        st.session_state.roi_selection_index = 0 # Set dropdown to "Uploaded GeoJSON"

    if submitted:
        roi, feature_collection = None, None
        with st.spinner("Processing your request..."):
            try:
                if selected_roi_name == "Uploaded GeoJSON":
                    if 'custom_gdf' in st.session_state and st.session_state['custom_gdf'] is not None:
                        gdf = st.session_state['custom_gdf']
                        # Assign a default name if one doesn't exist
                        if 'NAME' not in gdf.columns:
                            gdf['NAME'] = 'Custom ROI'
                        feature_collection = geemap.gdf_to_ee(gdf)
                        roi = feature_collection.geometry()
                    else:
                        st.warning("Please draw a shape or upload a file to use this option.")
                else: 
                    geojson_geom = florida_rois[selected_roi_name]
                    roi = ee.Geometry(geojson_geom)
                    if "County" in selected_roi_name:
                        county_name = selected_roi_name.replace(" County", "")
                        feature_collection = ee.FeatureCollection(COUNTY_FEATURE_COLLECTION).filter(ee.Filter.And(ee.Filter.eq('STATEFP', STATE_FIPS), ee.Filter.eq('NAME', county_name)))
                    else: 
                        feature_collection = ee.FeatureCollection(COUNTY_FEATURE_COLLECTION).filter(ee.Filter.eq('STATEFP', STATE_FIPS))

                if feature_collection:
                    results_fc = feature_collection.map(lambda f: compute_ueii_for_feature(f, start_year, end_year))
                    results_geojson = results_fc.getInfo()
                    gdf_results = gpd.GeoDataFrame.from_features(results_geojson['features'])
                    gdf_results.crs = "EPSG:4326"
                    gdf_results['UEII'] = pd.to_numeric(gdf_results['UEII'], errors='coerce')
                    
                    map_center = roi.centroid(maxError=1).coordinates().reverse().getInfo()
                    final_map = folium.Map(location=map_center, zoom_start=9, tiles="CartoDB positron")
                    folium.Choropleth(geo_data=gdf_results, data=gdf_results, columns=['NAME', 'UEII'], key_on='feature.properties.NAME', fill_color='YlOrRd', fill_opacity=opacity, line_opacity=0.3, nan_fill_color='grey', legend_name=f'UEII (%) {start_year}–{end_year}').add_to(final_map)
                    folium.LayerControl().add_to(final_map)

                    st.session_state['final_map'] = final_map
                    st.success("Map generated successfully!")
                    st_folium(final_map, use_container_width=True, key="final_map_display")
                    st.stop()
                elif submitted:
                    st.error("Could not determine a valid ROI. Please select an option, draw, or upload a file.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
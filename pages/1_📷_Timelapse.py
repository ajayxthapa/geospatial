import ee
import json
import os
import warnings
import datetime
import fiona
import geopandas as gpd
import folium
import streamlit as st
import geemap.colormaps as cm
import geemap.foliumap as geemap
from datetime import date, timedelta
from shapely.geometry import Polygon, shape

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")


@st.cache_data
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    geemap.ee_initialize(token_name=token_name)


@st.cache_data
def get_collection_date_range(collection_id):
    """Gets the start and end dates of an Earth Engine image collection."""
    try:
        collection = ee.ImageCollection(collection_id)
        first_image = collection.sort('system:time_start').first()
        last_image = collection.sort('system:time_start', False).first()
        start_date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        end_date = ee.Date(last_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        return datetime.datetime.strptime(start_date, '%Y-%m-%d').date(), datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    except Exception as e:
        # Return a default range if the collection is empty or doesn't have the property
        print(f"Could not fetch date range for {collection_id}: {e}")
        return datetime.date(1980, 1, 1), date.today()

@st.cache_data
def get_full_landsat_date_range():
    """
    Returns a fixed, hardcoded date range for the Landsat program to ensure
    instantaneous loading of the UI. This avoids slow GEE queries on startup.
    """
    # Landsat 5, the first collection used in the app, began operations in March 1984.
    start_date = datetime.date(1984, 3, 1)
    
    # The end date can be set to today. The app will simply not find images
    # for the last few days if they haven't been processed and added yet.
    end_date = date.today()
    
    return start_date, end_date

def apply_mask(img, use_radsat=False):
    """Applies a cloud, shadow, and optional radiometric saturation mask."""
    # Cloud and shadow mask (bits 3 and 4)
    qa = img.select('QA_PIXEL').bitwiseAnd(0b11000).eq(0)
    masked = img.updateMask(qa)
    
    # Optional radiometric saturation mask
    if use_radsat:
        flag = masked.bandNames().contains('QA_RADSAT')
        # Use ee.Algorithms.If to avoid errors on collections without QA_RADSAT
        masked = ee.Image(
            ee.Algorithms.If(
                flag,
                masked.updateMask(masked.select('QA_RADSAT').eq(0)),
                masked
            )
        )
    return masked

def resample_to_500m(image):
    """
    Resamples an image to 500m resolution, first ensuring it has a projection.
    """
    # Get the projection from the first band of the image.
    proj = image.select(0).projection()
    
    # Resample using the image's own projection as the starting point.
    return image.setDefaultProjection(proj).reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024
    ).reproject(crs='EPSG:4326', scale=500)

def get_annual_rgb_composite(year, region):
    """Computes an annual median true-color RGB composite from Landsat, harmonizing bands."""
    year = ee.Number(year)
    landsat_info = get_landsat_info(year)
    collection = landsat_info['collection']
    
    # Determine the correct RGB bands based on the satellite/year
    is_l8_or_l9 = year.gte(2014)
    rgb_bands = ee.List(ee.Algorithms.If(
        is_l8_or_l9,
        ['SR_B4', 'SR_B3', 'SR_B2'],  # For Landsat 8/9
        ['SR_B3', 'SR_B2', 'SR_B1']   # For Landsat 5/7
    ))
    
    # Define the new, consistent band names
    new_band_names = ee.List(['red', 'green', 'blue'])

    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)

    # Filter and mask the collection
    filtered_collection = (collection
                           .filterBounds(region)
                           .filterDate(start_date, end_date)
                           .map(lambda img: apply_mask(img, use_radsat=False)))

    # Function to select the correct bands and rename them
    def harmonize_bands(img):
        return img.select(rgb_bands).rename(new_band_names)

    # Map the harmonization, create a median composite, and clip
    median_image = filtered_collection.map(harmonize_bands).median().clip(region)
    
    return median_image

def create_landsat_rgb_timelapse(roi, out_gif, start_date, end_date, vis_params, **kwargs):
    """Creates a true-color Landsat timelapse using annual composites."""
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    years = ee.List.sequence(start_year, end_year)

    # Map over the years to create a collection of annual RGB composites
    def create_yearly_composite(year):
        annual_composite = get_annual_rgb_composite(year, roi)
        return annual_composite.set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis())

    composites = ee.ImageCollection.fromImages(years.map(create_yearly_composite))

    # Generate the timelapse using the constructed collection of annual composites
    geemap.create_timelapse(
        collection=composites,
        start_date=start_date,
        end_date=end_date,
        out_gif=out_gif,
        region=roi,
        vis_params=vis_params,
        **kwargs
    )
    return out_gif

# Helper to get the correct collection and band names for a given year
def get_landsat_info(year):
    """Returns the appropriate collection and band names for a given year."""
    year = ee.Number(year)
    
    # Define collections
    l9_col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    l8_col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    l7_col = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    l5_col = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    
    # Select collection based on year
    collection = ee.ImageCollection(ee.Algorithms.If(
        year.gte(2023), l9_col,
        ee.Algorithms.If(year.gte(2014), l8_col,
        ee.Algorithms.If(year.gte(2012), l7_col, l5_col))
    ))
    
    # Select band names based on year (Landsat 8/9 vs 5/7)
    is_l8_or_l9 = year.gte(2014)
    info = {
        'collection': collection,
        'lst_band': ee.String(ee.Algorithms.If(is_l8_or_l9, 'ST_B10', 'ST_B6')),
        'nir_band': ee.String(ee.Algorithms.If(is_l8_or_l9, 'SR_B5', 'SR_B4')),
        'red_band': ee.String(ee.Algorithms.If(is_l8_or_l9, 'SR_B4', 'SR_B3')),
        'swir_band': ee.String(ee.Algorithms.If(is_l8_or_l9, 'SR_B6', 'SR_B5')),
    }
    return info

# A single function to generate any annual index
def get_annual_composite(year, region, index_type):
    """
    Computes an annual median composite for a specified index (LST, NDVI, NDBI).
    This version includes the .copyProperties() fix for the projection error.
    """
    year = ee.Number(year)
    landsat_info = get_landsat_info(year)
    collection = landsat_info['collection']
    
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year, 12, 31)

    filtered_collection = (collection
                           .filterBounds(region)
                           .filterDate(start_date, end_date)
                           .map(lambda img: apply_mask(img, use_radsat=(index_type == 'LST'))))

    def calculate_index(image):
        # This 'image' argument is the masked Landsat image with projection info.
        
        # Calculate the desired index.
        if index_type == 'LST':
            calc_image = image.select(landsat_info['lst_band']) \
                            .multiply(0.00341802).add(149.0).subtract(273.15) \
                            .rename('LST')
        elif index_type == 'NDVI':
            calc_image = image.normalizedDifference([landsat_info['nir_band'], landsat_info['red_band']]) \
                            .rename('NDVI')
        elif index_type == 'NDBI':
            calc_image = image.normalizedDifference([landsat_info['swir_band'], landsat_info['nir_band']]) \
                            .rename('NDBI')
        else:
            return image

        # THE FIX: Copy all properties (including projection and timestamp) from the
        # source 'image' to the newly calculated 'calc_image'.
        return calc_image.copyProperties(image, image.propertyNames())

    final_image = (filtered_collection
                   .map(calculate_index)
                   .map(resample_to_500m)
                   .median()
                   .clip(region)
                   .selfMask())

    return final_image

# (Replace the old create_optimized_landsat_lst_timelapse function)
def create_optimized_landsat_lst_timelapse(
    roi, out_gif, start_date, end_date, **kwargs
):
    """Creates a timelapse of Landsat Land Surface Temperature using efficient annual composites."""
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    years = ee.List.sequence(start_year, end_year)

    def create_yearly_composite(year):
        # Call the new helper function for each year
        annual_composite = get_annual_composite(year, roi, 'LST')
        return annual_composite.set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis())

    # Create a collection of annual composites
    composites = ee.ImageCollection.fromImages(years.map(create_yearly_composite))

    # Define visualization parameters for LST
    vis_params = {
        'min': -10, 'max': 40, 'palette': kwargs.get('palette') or ['blue','green','yellow','red']
    }
    
    # Generate the timelapse using the collection of annual composites
    geemap.create_timelapse(
        collection=composites,
        start_date=start_date,
        end_date=end_date,
        out_gif=out_gif,
        region=roi,
        bands=['LST'],
        vis_params=vis_params,
        **kwargs
    )
    return out_gif


def create_landsat_index_timelapse(
    roi, out_gif, index_type, start_year, end_year, **kwargs
):
    """Creates a timelapse for Landsat indices (NDVI, NDBI, UTFVI, UHS) using efficient annual composites."""
    
    # For complex indices requiring per-image stats, we must use a client-side loop
    # to avoid "Too many concurrent aggregations" GEE error.
    if index_type in ['UTFVI', 'UHS']:
        st.warning(f"Info: {index_type} calculation is complex and will be processed year-by-year, which may be slow. Please be patient.")
        
        python_years = range(start_year, end_year + 1)
        image_list = []

        # This is a client-side loop (slower but necessary)
        for year in python_years:
            if index_type == 'UTFVI':
                lst_annual = get_annual_composite(year, roi, 'LST')
                # Use .getInfo() to bring the mean value from the server to the client
                # The 'lst_annual' image is already a 500m composite, so we use it directly.
                lst_mean = lst_annual.reduceRegion(ee.Reducer.mean(), roi, 500).get('LST').getInfo()
                
                # Check if the calculation was successful before proceeding
                if lst_mean is not None:
                    utfvi = lst_annual.subtract(lst_mean).divide(lst_mean).rename('UTFVI')
                    image_list.append(utfvi.set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis()))

            elif index_type == 'UHS':
                lst_annual = get_annual_composite(year, roi, 'LST')
                # Get stats for the year
                stats = lst_annual.reduceRegion(
                    ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True),
                    roi, 30
                ).getInfo()
                
                if stats and stats.get('LST_mean') is not None and stats.get('LST_stdDev') is not None:
                    mean = stats.get('LST_mean')
                    std_dev = stats.get('LST_stdDev')
                    threshold = mean + (std_dev * 2)
                    hotspot = lst_annual.gt(threshold).rename('UHS')
                    image_list.append(hotspot.set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis()))

        if not image_list:
            st.error("Could not generate any images. The ROI might be too small or have too much cloud cover over the time period.")
            st.stop()
            
        composites = ee.ImageCollection.fromImages(image_list)

    # For simple indices, the fast server-side .map() is fine
    else:
        years = ee.List.sequence(start_year, end_year)
        
        def create_yearly_composite(year):
            year = ee.Number(year)
            return get_annual_composite(year, roi, index_type)

        image_list = years.map(
            lambda year: create_yearly_composite(year).set('system:time_start', ee.Date.fromYMD(year, 6, 1).millis())
        )
        composites = ee.ImageCollection.fromImages(image_list)

    # Visualization parameters
    palettes = {
        'NDVI': ['FFFFFF', 'CE7E45', 'FCD163', '66A000', '207401'],
        'NDBI': ['brown', 'white', 'blue'],
        'UTFVI': ['green', 'white', 'red'],
        'UHS': ['lightgray', 'red']
    }
    vis_params = {'min': -0.5, 'max': 0.5, 'palette': kwargs.get('palette') or palettes[index_type]}
    if index_type == 'NDVI':
        vis_params.update({'min': 0, 'max': 1})
    if index_type == 'UHS':
        vis_params.update({'min': 0, 'max': 1})

    # Generate the timelapse
    geemap.create_timelapse(
        collection=composites,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        out_gif=out_gif,
        region=roi,
        bands=[index_type],
        vis_params=vis_params,
        **kwargs
    )
    return out_gif

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

landsat_rois = {
    "Aral Sea": Polygon(
        [
            [57.667236, 43.834527],
            [57.667236, 45.996962],
            [61.12793, 45.996962],
            [61.12793, 43.834527],
            [57.667236, 43.834527],
        ]
    ),
}

# Function to get Florida counties dynamically from TIGER datasets
@st.cache_data
def get_florida_counties():
    """
    Get Florida counties from TIGER/2018 datasets.
    
    This version calls .getInfo() on the geometries to make them
    serializable for Streamlit's cache.
    """
    try:
        # Get Florida state boundary
        states = ee.FeatureCollection("TIGER/2018/States")
        florida = states.filter(ee.Filter.eq('NAME', 'Florida')).first()
        
        # Get Florida counties
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties.filter(ee.Filter.eq('STATEFP', '12'))  # Florida FIPS code
        
        # Create ROI dictionary. Use .getInfo() to fetch the geometry as a GeoJSON dict.
        rois = {
            "Florida State": florida.geometry().getInfo()
        }
        
        # Get all Florida counties as a list of features
        county_features = florida_counties.getInfo()['features']
        
        # Loop through the features client-side to build the dictionary
        for feature in county_features:
            county_name = feature['properties']['NAME']
            # The geometry is already a GeoJSON dictionary in the feature
            county_geom = feature['geometry']
            if county_geom:
                rois[f"{county_name} County"] = county_geom
                
        return rois
    
    except Exception as e:
        st.error(f"Error loading Florida counties from GEE: {e}. Using a fallback boundary.")
        # Fallback returns a GeoJSON dictionary, which is serializable
        return {
            "Florida State": ee.Geometry.Rectangle([-87.634896, 24.396308, -79.974306, 31.000968]).getInfo()
        }

# Get Florida ROIs dynamically

modis_rois = {
    "World": Polygon(
        [
            [-171.210938, -57.136239],
            [-171.210938, 79.997168],
            [177.539063, 79.997168],
            [177.539063, -57.136239],
            [-171.210938, -57.136239],
        ]
    ),
    "Africa": Polygon(
        [
            [-18.6983, 38.1446],
            [-18.6983, -36.1630],
            [52.2293, -36.1630],
            [52.2293, 38.1446],
        ]
    ),
    "USA": Polygon(
        [
            [-127.177734, 23.725012],
            [-127.177734, 50.792047],
            [-66.269531, 50.792047],
            [-66.269531, 23.725012],
            [-127.177734, 23.725012],
        ]
    ),
}

@st.cache_data
def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

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


def app():

    today = date.today()

    st.title("Create Satellite Timelapse")

    st.markdown(
        """
        An interactive web app for creating [Landsat](https://developers.google.com/earth-engine/datasets/catalog/landsat) timelapse for any location around the globe.
        The app was built using [streamlit](https://streamlit.io), [geemap](https://geemap.org), and [Google Earth Engine](https://earthengine.google.com).
        """
    )
    
    ee_authenticate(token_name="EARTHENGINE_TOKEN")
    
    # Get Florida ROIs and update the modis_rois dictionary
    florida_rois = get_florida_counties()
    if florida_rois:
        modis_rois.update(florida_rois)

    row1_col1, row1_col2 = st.columns([2, 1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    # Reset session state variables
    st.session_state["ee_asset_id"] = None
    st.session_state["bands"] = None
    st.session_state["palette"] = None
    st.session_state["vis_params"] = None

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

        keyword = st.text_input("Search for a location:", "")
        if keyword:
            locations = geemap.geocode(keyword)
            if locations is not None and len(locations) > 0:
                str_locations = [str(g)[1:-1] for g in locations]
                location = st.selectbox("Select a location:", str_locations)
                loc_index = str_locations.index(location)
                selected_loc = locations[loc_index]
                lat, lng = selected_loc.lat, selected_loc.lng
                folium.Marker(location=[lat, lng], popup=location).add_to(m)
                m.set_center(lng, lat, 12)
                st.session_state["zoom_level"] = 12

        collection_options = {
            "Landsat TM-ETM-OLI Surface Reflectance": "LANDSAT/LC08/C02/T1_L2", # Placeholder
            "Landsat Land Surface Temperature (LST)": "LANDSAT/LT05/C02/T1_L2", # Placeholder
            "Normalized Difference Vegetation Index (NDVI)": "LANDSAT/LC08/C02/T1_L2", # Placeholder
            "Normalized Difference Built-up Index (NDBI)": "LANDSAT/LC08/C02/T1_L2", # Placeholder
            "Urban Thermal Field Variance Index (UTFVI)": "LANDSAT/LC08/C02/T1_L2", # Placeholder
            "Urban Hotspots (UHS)": "LANDSAT/LC08/C02/T1_L2", # Placeholder
            "Any Earth Engine ImageCollection": None
        }

        collection = st.selectbox(
            "Select a satellite image source: ",
            list(collection_options.keys()),
            index=0,
        )
        
        asset_id_for_date = None
        if collection == "Any Earth Engine ImageCollection":
            asset_id_for_date = st.text_input("Enter an ee.ImageCollection asset ID to get date range:", "")
            if asset_id_for_date:
                st.session_state["ee_asset_id"] = asset_id_for_date
        else:
            asset_id_for_date = None

        min_date, max_date = datetime.date(1980, 1, 1), date.today()
        if asset_id_for_date:
            with st.spinner(f"Fetching date range for {collection}..."):
                min_date, max_date = get_collection_date_range(asset_id_for_date)
        elif "Landsat" in collection or collection.startswith("Normalized") or collection.startswith("Urban"):
            with st.spinner(f"Fetching full Landsat date range..."):
                min_date, max_date = get_full_landsat_date_range()
        
        roi_options = ["Uploaded GeoJSON"] + list(modis_rois.keys())

        if collection == "Any Earth Engine ImageCollection":
            keyword = st.text_input("Enter a keyword to search (e.g., MODIS):", "")
            if keyword:
                assets = geemap.search_ee_data(keyword)
                ee_assets = [asset for asset in assets if asset["ee_id_snippet"].startswith("ee.ImageCollection")]
                asset_titles = [x["title"] for x in ee_assets]
                if asset_titles:
                    dataset = st.selectbox("Select a dataset:", asset_titles)
                    index = asset_titles.index(dataset)
                    ee_id = ee_assets[index]["id"]
                    st.session_state["ee_assets"] = ee_assets
                    st.session_state["asset_titles"] = asset_titles
                    with st.expander("Show dataset details", False):
                        html = geemap.ee_data_html(ee_assets[index])
                        st.markdown(html, True)
                else:
                    ee_id = ""
            else:
                ee_id = ""
            
            if not asset_id_for_date:
                asset_id = st.text_input("Enter an ee.ImageCollection asset ID:", ee_id)
                st.session_state["ee_asset_id"] = asset_id

        elif collection in ["Normalized Difference Vegetation Index (NDVI)", "Normalized Difference Built-up Index (NDBI)", "Urban Thermal Field Variance Index (UTFVI)", "Urban Hotspots (UHS)"]:
            with st.expander("Show dataset details", False):
                if collection == "Normalized Difference Vegetation Index (NDVI)":
                    st.markdown("**NDVI (Normalized Difference Vegetation Index)**: $(NIR - Red) / (NIR + Red)$")
                elif collection == "Normalized Difference Built-up Index (NDBI)":
                    st.markdown("**NDBI (Normalized Difference Built-up Index)**: $(SWIR - NIR) / (SWIR + NIR)$")
                elif collection == "Urban Thermal Field Variance Index (UTFVI)":
                    st.markdown("**UTFVI (Urban Thermal Field Variance Index)**: Combines LST and NDVI.")
                else:
                    st.markdown("**UHS (Urban Hotspots)**: Binary classification of urban heat islands.")

        sample_roi = st.selectbox("Select a sample ROI or upload a GeoJSON file:", roi_options, index=0)
        
        add_outline = st.checkbox("Overlay an administrative boundary on timelapse", False)
        if add_outline:
            with st.expander("Customize administrative boundary", True):
                overlay_options = {"User-defined": None, "Continents": "continents", "Countries": "countries", "US States": "us_states", "China": "china"}
                overlay = st.selectbox("Select an administrative boundary:", list(overlay_options.keys()), index=2)
                overlay_data = overlay_options[overlay]
                if overlay_data is None:
                    overlay_data = st.text_input("Enter a URL to a GeoJSON or an ee.FeatureCollection asset id:", "https://raw.githubusercontent.com/giswqs/geemap/master/examples/data/countries.geojson")
                overlay_color = st.color_picker("Boundary color:", "#000000")
                overlay_width = st.slider("Boundary line width:", 1, 20, 1)
                overlay_opacity = st.slider("Boundary opacity:", 0.0, 1.0, 1.0, 0.05)
        else:
            overlay_data, overlay_color, overlay_width, overlay_opacity = None, "black", 1, 1

    with row1_col1:
        with st.expander("Steps: Draw a rectangle on the map -> Export it as a GeoJSON -> Upload it back to the app -> Click the Submit button.", False):
            video_empty = st.empty()

        data = st.file_uploader("Upload a GeoJSON file to use as an ROI.", type=["geojson", "kml", "zip"])
        
        crs = "epsg:4326"
        if sample_roi == "Uploaded GeoJSON":
            if data:
                gdf = uploaded_file_to_gdf(data)
                st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
                m.add_gdf(gdf, "ROI")
                bounds = gdf.total_bounds
                m.set_center((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 9)
        else:
            geometry = modis_rois.get(sample_roi)
            if geometry:
                geojson_geom = geometry.getInfo() if hasattr(geometry, 'getInfo') else geometry
                shapely_geom = shape(geojson_geom)
                gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[shapely_geom])
                st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
                m.add_gdf(gdf, "ROI")
                bounds = gdf.total_bounds
                zoom = 2 if sample_roi == "World" else 9
                m.set_center((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, zoom)
        
        m.to_streamlit(height=600)

    with row1_col2:
        if collection != "Landsat TM-ETM-OLI Surface Reflectance":          
            palette_options = st.selectbox("Color palette", cm.list_colormaps(), index=90)
            palette_values = cm.get_palette(palette_options, 15)
            palette = st.text_area("Enter a custom palette:", palette_values)
            st.write(cm.plot_colormap(cmap=palette_options, return_fig=True))
            try:
                custom_palette = json.loads(palette.replace("'", '"'))
                st.session_state["palette"] = custom_palette
            except Exception as e:
                st.error(f"Invalid palette format: {e}")
                st.stop()
        
        # Determine a unique form key based on the collection type
        if collection in ["Landsat TM-ETM-OLI Surface Reflectance", "Any Earth Engine ImageCollection"]:
            form_key = "submit_generic_form"
        elif collection == "Landsat Land Surface Temperature (LST)":
            form_key = "submit_landsat_lst_form"
        elif "NDVI" in collection or "NDBI" in collection or "UTFVI" in collection or "UHS" in collection:
            index_type = collection.split('(')[-1][:-1]
            form_key = f"submit_{index_type.lower()}_form"
        else:
            form_key = "submit_other_form"

        with st.form(form_key):
            with st.expander("Customize timelapse", expanded=True):
                title = st.text_input("Title:", f"{collection} Timelapse")
                
                start_date = st.date_input("Start date:", value=min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("End date:", value=max_date, min_value=min_date, max_value=max_date)

                optimized_products = [
                    "Landsat TM-ETM-OLI Surface Reflectance", # Added for consistency
                    "Landsat Land Surface Temperature (LST)",
                    "Normalized Difference Vegetation Index (NDVI)",
                    "Normalized Difference Built-up Index (NDBI)",
                    "Urban Thermal Field Variance Index (UTFVI)",
                    "Urban Hotspots (UHS)"
                ]

                if collection in optimized_products:
                    frequency = "year"
                    reducer = "median"
                    st.info("Info: Frequency is fixed to 'Year' and Reducer to 'Median' for this product to improve performance.")
                else:
                    frequency = st.selectbox("Frequency:", ["year", "quarter", "month"], index=0)
                    reducer = st.selectbox("Reducer:", ["median", "mean", "min", "max"], index=0)

                speed = st.slider("Frames per second:", 1, 30, 5)
                font_size = st.slider("Font size:", 10, 50, 20)
                font_color = st.color_picker("Font color:", "#ffffff")
                add_progress_bar = st.checkbox("Add progress bar", True)
                progress_bar_color = st.color_picker("Progress bar color:", "#0000ff")
                mp4 = st.checkbox("Save as MP4", True)
                
                if collection in ["Landsat TM-ETM-OLI Surface Reflectance", "Any Earth Engine ImageCollection"]:
                    vis_params_text = st.text_area("Visualization Parameters (JSON):", '{"bands": ["red", "green", "blue"], "min": 0, "max": 16000, "gamma": 1.4}')
            
            submitted = st.form_submit_button("Submit")

            if submitted:
                roi = st.session_state.get("roi")
                if not roi:
                    st.error("Please select or upload a Region of Interest (ROI).")
                    st.stop()
                
                out_gif = geemap.temp_file_path(".gif")
                empty_text = st.empty()
                empty_image = st.empty()
                empty_video = st.container()

                empty_text.text("Computing... Please wait...")
                
                # --- CORRECTED SUBMISSION LOGIC ---
                try:
                    if collection == "Landsat TM-ETM-OLI Surface Reflectance":
                        vis_params = json.loads(vis_params_text)
                        create_landsat_rgb_timelapse(
                            roi=roi, out_gif=out_gif, start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"), vis_params=vis_params,
                            frames_per_second=speed, title=title, font_size=font_size,
                            font_color=font_color, add_progress_bar=add_progress_bar,
                            progress_bar_color=progress_bar_color, mp4=mp4,
                            overlay_data=overlay_data, overlay_color=overlay_color,
                            overlay_width=overlay_width, overlay_opacity=overlay_opacity
                        )
                    elif collection == "Landsat Land Surface Temperature (LST)":
                        create_optimized_landsat_lst_timelapse(
                            roi=roi, out_gif=out_gif, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"),
                            palette=custom_palette, frames_per_second=speed, title=title,
                            font_size=font_size, font_color=font_color, add_progress_bar=add_progress_bar, progress_bar_color=progress_bar_color, mp4=mp4,
                            overlay_data=overlay_data, overlay_color=overlay_color, overlay_width=overlay_width, overlay_opacity=overlay_opacity
                        )

                    elif "NDVI" in collection or "NDBI" in collection or "UTFVI" in collection or "UHS" in collection:
                        create_landsat_index_timelapse(
                            roi=roi, out_gif=out_gif, index_type=index_type, start_year=start_date.year, end_year=end_date.year,
                            palette=custom_palette, frames_per_second=speed, title=title,
                            font_size=font_size, font_color=font_color, add_progress_bar=add_progress_bar, progress_bar_color=progress_bar_color, mp4=mp4,
                            overlay_data=overlay_data, overlay_color=overlay_color, overlay_width=overlay_width, overlay_opacity=overlay_opacity
                        )

                    else: # Handles "Any Earth Engine ImageCollection"
                        vis_params = json.loads(vis_params_text) if 'vis_params_text' in locals() else {}
                        
                        geemap.create_timelapse(
                            st.session_state.get("ee_asset_id"),
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            region=roi, frequency=frequency, reducer=reducer, out_gif=out_gif, vis_params=vis_params,
                            frames_per_second=speed, title=title, font_size=font_size, font_color=font_color,
                            add_progress_bar=add_progress_bar, progress_bar_color=progress_bar_color, mp4=mp4,
                            overlay_data=overlay_data, overlay_color=overlay_color, overlay_width=overlay_width, overlay_opacity=overlay_opacity
                        )

                    if os.path.exists(out_gif):
                        geemap.reduce_gif_size(out_gif)
                        empty_text.text("Right-click the GIF to save it to your computer.")
                        empty_image.image(out_gif)
                        if mp4:
                            out_mp4 = out_gif.replace(".gif", ".mp4")
                            if os.path.exists(out_mp4):
                                with empty_video:
                                    st.text("Right-click the MP4 to save it.")
                                    st.video(out_mp4)
                    else:
                        empty_text.error("Timelapse generation failed. The output file was not created.")

                except Exception as e:
                    empty_text.error(f"An error occurred: {e}. Try reducing the ROI or timespan.")
try:
    app()
except Exception as e:
    st.error(f"An unexpected error occurred in the application: {e}")

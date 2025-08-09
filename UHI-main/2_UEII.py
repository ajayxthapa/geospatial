# app_fixed.py
import streamlit as st
import ee
import pandas as pd
import geopandas as gpd
import geemap.foliumap as geemap
import geemap.colormaps as gmc
from shapely.geometry import shape
import folium
import fiona
import os
import tempfile
import uuid
import branca
from matplotlib import cm as mpl_cm
from matplotlib.colors import to_hex

# -----------------------
# Page config / styles
# -----------------------
st.set_page_config(layout="wide", page_title="Florida UEII (fixed)", page_icon="🏞️")
st.markdown("""
    <style>
        .main .block-container { padding: 1rem 1.5rem; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Earth Engine init
# -----------------------
@st.cache_resource
def ee_initialize():
    try:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate()
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
ee_initialize()

# -----------------------
# Constants & helpers
# -----------------------
NLCD_COLLECTION = "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER"
COUNTIES_COLLECTION = ee.FeatureCollection("TIGER/2018/Counties")
DEVELOPED_CLASSES = [21, 22, 23, 24]
TARGET_YEARS = [1985, 1995, 2005, 2015, 2024]
SCALE = 30
SQM_TO_SQMI = 3.86102159e-7

@st.cache_resource
def get_nlcd_images():
    return ee.Dictionary({
        str(year): ee.ImageCollection(NLCD_COLLECTION)
                     .filter(ee.Filter.eq("year", year))
                     .first()
                     .select("b1")
        for year in TARGET_YEARS
    })
NLCD_IMAGES = get_nlcd_images()

def reset_app_state():
    keys = list(st.session_state.keys())
    for k in keys:
        # keep nothing so app is fully fresh
        st.session_state.pop(k, None)

def clear_results():
    for k in ["analysis_result", "active_roi", "uploaded_gdf"]:
        st.session_state.pop(k, None)

@st.cache_data
def get_florida_county_names():
    florida_counties = COUNTIES_COLLECTION.filter(ee.Filter.eq("STATEFP", "12"))
    return florida_counties.aggregate_array("NAME").sort().getInfo()

@st.cache_data
def uploaded_file_to_gdf(uploaded_file):
    _, ext = os.path.splitext(uploaded_file.name)
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    with open(tmp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if tmp.lower().endswith(".kml"):
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(tmp, driver="KML")
    else:
        gdf = gpd.read_file(tmp)
    # ensure EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

# --- GEE analysis functions (same logic) ---
def calculate_developed_area(image, geometry):
    mask = image.remap(DEVELOPED_CLASSES, [1]*len(DEVELOPED_CLASSES), 0).selfMask()
    area_image = mask.multiply(ee.Image.pixelArea()).rename("area")
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry, scale=SCALE, maxPixels=1e13
    )
    return ee.Number(stats.get("area")).multiply(SQM_TO_SQMI)

def compute_ueii(area_t1, area_t2, delta_t, total_area):
    return (ee.Number(area_t2).subtract(area_t1)
            .divide(ee.Number(total_area).multiply(delta_t))).multiply(100)

def analyze_feature(feature):
    geom = feature.geometry()
    area_sqmi = ee.Number(geom.area(maxError=1)).multiply(SQM_TO_SQMI)
    dev_areas = ee.Dictionary({
        str(year): calculate_developed_area(ee.Image(NLCD_IMAGES.get(str(year))), geom)
        for year in TARGET_YEARS
    })
    ueii_periods = {
        'UEII 1985-1995 (%)': compute_ueii(dev_areas.get('1985'), dev_areas.get('1995'), 10, area_sqmi),
        'UEII 1995-2005 (%)': compute_ueii(dev_areas.get('1995'), dev_areas.get('2005'), 10, area_sqmi),
        'UEII 2005-2015 (%)': compute_ueii(dev_areas.get('2005'), dev_areas.get('2015'), 10, area_sqmi),
        'UEII 2015-2024 (%)': compute_ueii(dev_areas.get('2015'), dev_areas.get('2024'), 9, area_sqmi)
    }
    props = {
        'ROI Area (sq mi)': area_sqmi,
        **{f'Developed Area {y} (sq mi)': dev_areas.get(str(y)) for y in TARGET_YEARS}
    }
    props.update(ueii_periods)
    name = ee.String(ee.Algorithms.If(feature.get('NAME'), feature.get('NAME'), 'Custom ROI'))
    return feature.set(props).set({'NAME': name})

@st.cache_data
def run_analysis(_roi_collection):
    results = _roi_collection.map(analyze_feature).getInfo()
    features = results.get("features", [])
    rows = []
    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        row = {'S.N.': i + 1, 'County': props.get('NAME', f'feature_{i}')}
        row.update(props)
        row['geometry'] = shape(feat['geometry'])
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    # convert numeric columns to floats where possible
    for c in gdf.columns:
        if c not in ['County', 'geometry']:
            gdf[c] = pd.to_numeric(gdf[c], errors='coerce')
    return gdf.round(6)

# -----------------------
# Robust color helper (returns list of hex)
# -----------------------
def make_color_list(palette, n=8):
    # Try geemap's palettes first (ColorBrewer style)
    try:
        pal = gmc.get_palette(palette, n)
        if pal and isinstance(pal, (list, tuple)):
            # geemap returns list of hex strings already
            return pal
    except Exception:
        pass
    # Fallback to matplotlib colormap (supports cividis/viridis)
    try:
        cmap = mpl_cm.get_cmap(palette)
        return [to_hex(cmap(i/(n-1))) for i in range(n)]
    except Exception:
        # Final fallback: YlOrRd small list
        return ['#ffffcc','#ffeda0','#feb24c','#fd8d3c','#f03b20','#bd0026'][:n]

def add_choropleth_to_map(m, gdf, column_to_plot, legend_name, palette):
    # Use GeoJson + branca LinearColormap + style_function (no folium.Choropleth palette strings)
    plot_gdf = gdf.dropna(subset=[column_to_plot]).copy()
    if plot_gdf.empty:
        st.warning(f"No data to plot for '{column_to_plot}'. Showing ROI only.")
        folium.GeoJson(gdf.__geo_interface__, name="ROI").add_to(m)
        m.center_object(ee.Geometry(gdf.unary_union.__geo_interface__), zoom=6)
        return

    # ensure numeric
    plot_gdf[column_to_plot] = pd.to_numeric(plot_gdf[column_to_plot], errors='coerce')
    vals = plot_gdf[column_to_plot].dropna().astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    colors = make_color_list(palette, n=8)
    colormap = branca.colormap.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=legend_name).to_step(8)
    colormap.add_to(m)

    def style_function(feature):
        props = feature.get('properties', {})
        val = props.get(column_to_plot)
        try:
            val = float(val)
            fill = colormap(val)
        except Exception:
            fill = "rgba(0,0,0,0)"  # transparent if missing
        return {
            "fillColor": fill,
            "color": "#444444",
            "weight": 0.6,
            "fillOpacity": 0.75,
        }

    tooltip_fields = []
    if 'County' in plot_gdf.columns:
        tooltip_fields = ['County', column_to_plot]
        aliases = ['County:', legend_name + ':']
    else:
        tooltip_fields = [column_to_plot]
        aliases = [legend_name + ':']

    folium.GeoJson(
        plot_gdf.__geo_interface__,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases),
        name=legend_name
    ).add_to(m)

    # center map
    m.center_object(ee.Geometry(plot_gdf.unary_union.__geo_interface__), zoom=7 if len(plot_gdf) > 1 else 9)

# -----------------------
# App UI
# -----------------------
with st.sidebar:
    st.title("About")
    st.info("UEII analyzer — upload an ROI (default) or pick a Florida county.")
    if st.button("Reset App State"):
        reset_app_state()
        st.experimental_rerun()

st.title("🏞️ Urban Expansion Intensity (UEII) Analyzer (fixed)")

# File uploader
data = st.file_uploader("Upload a GeoJSON, KML, or zipped Shapefile", type=["geojson", "kml", "zip"])

# layout
map_col, ctrl_col = st.columns([2, 1])

with ctrl_col:
    st.header("Controls")
    county_names = [name + " County" for name in get_florida_county_names()]
    roi_options = ["Uploaded GeoJSON", "Florida State"] + county_names

    # Default always shows "Uploaded GeoJSON" on load/refresh
    selected_roi = st.selectbox("1) Select a Region of Interest (ROI):", roi_options, index=0)

    # If user changed selection between runs, clear previous results
    if st.session_state.get("prev_selected_roi") != selected_roi:
        clear_results()
    st.session_state["prev_selected_roi"] = selected_roi

    palettes = [
        'YlOrRd', 'YlGnBu', 'viridis', 'plasma', 'inferno',
        'magma', 'cividis', 'Greys', 'Purples', 'Blues'
    ]
    palette_choice = st.selectbox("3) Select color palette:", palettes, index=0)
    # palette preview (best-effort)
    try:
        st.write(gmc.plot_colormap(cmap=palette_choice, return_fig=True))
    except Exception:
        try:
            cmap = mpl_cm.get_cmap(palette_choice)
            # quick preview using geemap helper (works often), else skip
            st.write(gmc.plot_colormap(cmap=palette_choice, return_fig=True))
        except Exception:
            st.write(f"Preview not available for `{palette_choice}`")

    ueii_periods_for_map = [
        'UEII 2015-2024 (%)', 'UEII 2005-2015 (%)',
        'UEII 1995-2005 (%)', 'UEII 1985-1995 (%)',
        'Developed Area 2024 (sq mi)'
    ]
    selected_period = st.selectbox("2) Select data for map:", ueii_periods_for_map, index=0)
    submitted = st.button("Submit")

# -----------------------
# Prepare active ROI (uploaded or county)
# -----------------------
if selected_roi == "Uploaded GeoJSON":
    if data:
        try:
            gdf = uploaded_file_to_gdf(data)
            st.session_state["uploaded_gdf"] = gdf
            fc = geemap.gdf_to_ee(gdf, geodesic=False)
            st.session_state["active_roi"] = fc
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.session_state["active_roi"] = None
    else:
        # no uploaded file yet
        st.session_state.pop("active_roi", None)
else:
    county_name = selected_roi.replace(" County", "")
    if county_name == "Florida State":
        st.session_state["active_roi"] = COUNTIES_COLLECTION.filter(ee.Filter.eq('STATEFP', '12'))
    else:
        st.session_state["active_roi"] = COUNTIES_COLLECTION.filter(
            ee.Filter.And(ee.Filter.eq('STATEFP', '12'), ee.Filter.eq('NAME', county_name))
        )

# -----------------------
# If submitted -> run analysis and save result
# -----------------------
if submitted:
    if "active_roi" not in st.session_state or st.session_state["active_roi"] is None:
        st.error("Upload a GeoJSON (or pick a county) before submitting.")
    else:
        with st.spinner("Running GEE analysis..."):
            try:
                gdf_results = run_analysis(st.session_state["active_roi"])
                if gdf_results is None:
                    st.error("No results — ROI might be outside coverage or too small.")
                    st.session_state.pop("analysis_result", None)
                else:
                    st.session_state["analysis_result"] = gdf_results
                    st.success("Analysis complete — see map.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state.pop("analysis_result", None)

# -----------------------
# Build and render single map (only one m.to_streamlit call)
# -----------------------
with map_col:
    st.header("🗺️ Map")
    # create a map object (fine to recreate each run as long as we render once)
    m = geemap.Map(basemap="HYBRID", plugin_Draw=True, Draw_export=True, locate_control=True)
    m.add_basemap("CARTO_POSITRON")

    if "analysis_result" in st.session_state:
        add_choropleth_to_map(m, st.session_state["analysis_result"], selected_period, selected_period, palette_choice)
        st.subheader(f"Choropleth Map: {selected_period} for {selected_roi}")
    else:
        # show ROI geometry only
        if selected_roi == "Uploaded GeoJSON" and st.session_state.get("uploaded_gdf") is not None:
            m.add_gdf(st.session_state["uploaded_gdf"], layer_name="Uploaded ROI")
            m.center_object(st.session_state["active_roi"], zoom=8)
        elif st.session_state.get("active_roi") is not None:
            m.addLayer(st.session_state["active_roi"], {}, selected_roi)
            m.center_object(st.session_state["active_roi"], zoom=6)
        else:
            # default Florida
            m.addLayer(COUNTIES_COLLECTION.filter(ee.Filter.eq("STATEFP","12")), {}, "Florida")
            m.center_object(COUNTIES_COLLECTION.filter(ee.Filter.eq('STATEFP','12')), zoom=6)

    # render once
    m.to_streamlit(height=650, key="ueii_map_fixed")

# -----------------------
# Show data table if available
# -----------------------
if "analysis_result" in st.session_state:
    with st.expander("📊 View Full Data Table", expanded=True):
        df_view = st.session_state["analysis_result"].drop(columns=["geometry"], errors="ignore")
        st.dataframe(df_view, use_container_width=True)

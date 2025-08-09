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
    """
    Get Florida counties from TIGER/2018 datasets.
    Returns a dictionary with county names and their geometries.
    """
    try:
        # Get Florida counties
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties.filter(ee.Filter.eq('STATEFP', '12'))  # Florida FIPS code
        
        # Get Florida state boundary
        states = ee.FeatureCollection("TIGER/2018/States")
        florida = states.filter(ee.Filter.eq('NAME', 'Florida')).first()
        
        # Create ROI dictionary
        rois = {
            "Florida shapefile geometry": florida.geometry().getInfo()
        }
        
        # Get all Florida counties as a list of features
        county_features = florida_counties.getInfo()['features']
        
        # Loop through the features to build the dictionary
        for feature in county_features:
            county_name = feature['properties']['NAME']
            county_geom = feature['geometry']
            if county_geom:
                rois[f"{county_name} County"] = county_geom
                
        return rois
    
    except Exception as e:
        st.error(f"Error loading Florida counties from GEE: {e}. Using a fallback boundary.")
        # Fallback returns a GeoJSON dictionary
        return {
            "Florida shapefile geometry": ee.Geometry.Rectangle([-87.634896, 24.396308, -79.974306, 31.000968]).getInfo()
        }

@st.cache_data
def uploaded_file_to_gdf(data):
    """Convert uploaded file to GeoDataFrame"""
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

def calculate_developed_area(image, geometry, scale=30):
    """Calculate developed area from NLCD image"""
    developed_classes = [21, 22, 23, 24]  # NLCD developed classes
    sqm_to_sqmi = 3.86102159e-7
    
    # Create mask for developed areas
    mask = image.remap(developed_classes, [1] * len(developed_classes), 0).selfMask()
    area_image = mask.multiply(ee.Image.pixelArea()).rename('area')
    
    # Calculate total developed area
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

def analyze_county(feature, target_years, nlcd_images):
    """Analyze UEII for a single county feature"""
    sqm_to_sqmi = 3.86102159e-7
    
    geom = feature.geometry()
    county_area_sqmi = ee.Number(geom.area(maxError=1)).multiply(sqm_to_sqmi)
    
    # Calculate developed areas for each year
    dev_areas = ee.Dictionary({
        str(year): calculate_developed_area(ee.Image(nlcd_images.get(str(year))), geom)
        for year in target_years
    })
    
    # Compute UEII for different periods
    properties = {
        'County Area (sq mi)': county_area_sqmi,
    }
    
    # Add developed areas
    for year in target_years:
        properties[f'Developed Area {year} (sq mi)'] = dev_areas.get(str(year))
    
    # Calculate UEII for consecutive periods
    for i in range(len(target_years) - 1):
        year1, year2 = target_years[i], target_years[i + 1]
        delta_t = year2 - year1
        ueii_value = compute_ueii(
            dev_areas.get(str(year1)), 
            dev_areas.get(str(year2)), 
            delta_t, 
            county_area_sqmi
        )
        properties[f'UEII {year1}-{year2} (%)'] = ueii_value
    
    return feature.set(properties)

def analyze_region_batch(region_geometry, region_name, target_years):
    """Analyze UEII for a region using batch processing"""
    
    # Get NLCD collection
    nlcd_collection = ee.ImageCollection("projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER")
    
    # Get images for target years
    nlcd_images = ee.Dictionary({
        str(year): nlcd_collection.filter(ee.Filter.eq('year', year)).first().select('b1')
        for year in target_years
    })
    
    # Check if analyzing Florida (all counties) or single region
    if region_name == "Florida shapefile geometry":
        # Get all Florida counties
        counties_collection = ee.FeatureCollection("TIGER/2018/Counties")
        florida_counties = counties_collection.filter(ee.Filter.eq('STATEFP', '12'))
        
        # Process counties in batches
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
            subset_results = subset.map(lambda f: analyze_county(f, target_years, nlcd_images)).getInfo()
            all_features.extend(subset_results['features'])
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Build results DataFrame
        data_for_df = []
        for i, feature in enumerate(all_features):
            props = feature['properties']
            row_data = {'S.N.': i + 1, 'Region': props['NAME']}
            row_data.update(props)
            data_for_df.append(row_data)
        
        return pd.DataFrame(data_for_df)
    
    else:
        # Single region analysis
        sqm_to_sqmi = 3.86102159e-7
        region_area_sqmi = ee.Number(region_geometry.area(maxError=1)).multiply(sqm_to_sqmi)
        
        # Calculate developed areas for each year
        developed_areas = {}
        for year in target_years:
            year_str = str(year)
            image = nlcd_collection.filter(ee.Filter.eq('year', year)).first().select('b1')
            developed_areas[year_str] = calculate_developed_area(image, region_geometry)
        
        # Compute results
        results = {
            'S.N.': 1,
            'Region': region_name,
            'Region Area (sq mi)': region_area_sqmi.getInfo()
        }
        
        # Add developed areas to results
        for year in target_years:
            year_str = str(year)
            if year_str in developed_areas:
                results[f'Developed Area {year} (sq mi)'] = developed_areas[year_str].getInfo()
        
        # Calculate UEII for consecutive periods
        for i in range(len(target_years) - 1):
            year1, year2 = target_years[i], target_years[i + 1]
            year1_str, year2_str = str(year1), str(year2)
            
            if year1_str in developed_areas and year2_str in developed_areas:
                delta_t = year2 - year1
                ueii_value = compute_ueii(
                    developed_areas[year1_str], 
                    developed_areas[year2_str], 
                    delta_t, 
                    region_area_sqmi
                ).getInfo()
                
                results[f'UEII {year1}-{year2} (%)'] = ueii_value
        
        return pd.DataFrame([results])

def create_ueii_visualization(results_df):
    """Create interactive visualizations for UEII results"""
    
    # Extract UEII columns
    ueii_columns = [col for col in results_df.columns if 'UEII' in col and '(%)' in col]
    
    if not ueii_columns:
        st.warning("No UEII data found for visualization.")
        return
    
    # Create time series plot
    st.subheader("📊 UEII Time Series Analysis")
    
    # Prepare data for plotting
    plot_data = []
    for _, row in results_df.iterrows():
        region_name = row['Region']
        for col in ueii_columns:
            period = col.replace('UEII ', '').replace(' (%)', '')
            value = row[col]
            if pd.notna(value):
                plot_data.append({
                    'Region': region_name,
                    'Period': period,
                    'UEII (%)': value
                })
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # For many regions, show top 10 only
        if len(results_df) > 10:
            latest_col = ueii_columns[-1] if ueii_columns else None
            if latest_col:
                top_regions = results_df.nlargest(10, latest_col)['Region'].tolist()
                plot_df = plot_df[plot_df['Region'].isin(top_regions)]
                st.info("Showing top 10 regions with highest UEII in the latest period.")
        
        # Create interactive line plot
        fig = px.line(
            plot_df, 
            x='Period', 
            y='UEII (%)',
            color='Region',
            title='Urban Expansion Intensity Index Over Time',
            markers=True
        )
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="UEII (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create heatmap for Florida counties if analyzing all counties
        if len(results_df) > 10 and len(ueii_columns) >= 2:
            st.subheader("🗺️ UEII Heatmap - Top and Bottom Counties")
            
            latest_col = ueii_columns[-1]
            top5 = results_df.nlargest(5, latest_col)
            bottom5 = results_df.nsmallest(5, latest_col)
            heatmap_df = pd.concat([top5, bottom5])
            
            # Melt for heatmap
            heatmap_long = heatmap_df.melt(
                id_vars=['Region'],
                value_vars=ueii_columns,
                var_name='Time Period',
                value_name='UEII (%)'
            )
            
            # Pivot for plotting
            heatmap_data = heatmap_long.pivot(index='Region', columns='Time Period', values='UEII (%)')
            
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='YlOrRd',
                aspect='auto',
                title='Top 5 and Bottom 5 Regions by UEII'
            )
            fig_heatmap.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Region"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

def create_summary_statistics(results_df):
    """Create summary statistics for UEII analysis"""
    st.subheader("📊 Summary Statistics")
    
    # Extract UEII columns
    ueii_columns = [col for col in results_df.columns if 'UEII' in col and '(%)' in col]
    
    if ueii_columns:
        # Calculate summary statistics
        summary_stats = results_df[ueii_columns].describe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistical Summary:**")
            st.dataframe(summary_stats)
        
        with col2:
            # Find regions with highest and lowest UEII in latest period
            if len(ueii_columns) > 0:
                latest_period = ueii_columns[-1]
                latest_data = results_df[['Region', latest_period]].dropna()
                
                if not latest_data.empty:
                    highest = latest_data.loc[latest_data[latest_period].idxmax()]
                    lowest = latest_data.loc[latest_data[latest_period].idxmin()]
                    
                    st.write("**Key Insights:**")
                    st.write(f"🔥 **Highest Expansion:** {highest['Region']}")
                    st.write(f"📊 **UEII:** {highest[latest_period]:.3f}%")
                    st.write("")
                    st.write(f"🌿 **Lowest Expansion:** {lowest['Region']}")
                    st.write(f"📊 **UEII:** {lowest[latest_period]:.3f}%")

def main():
    # Sidebar
    st.sidebar.title("About UEII")
    st.sidebar.info(
        """
        The **Urban Expansion Intensity Index (UEII)** measures the rate of urban 
        development relative to the total area of a region over a specific time period.
        
        **Formula:** UEII = ((Area_t2 - Area_t1) / (Total_Area × ΔT)) × 100
        
        **Data Source:** USGS Annual NLCD Landcover
        
        - Web App: Built with Streamlit & Google Earth Engine
        - GitHub: Contact for source code access
        """
    )
    
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        📧 [Email](mailto:contact@example.com)  
        💻 [GitHub](https://github.com/username)
        """
    )
    
    # Main content
    st.title("Create Urban Expansion Intensity Index (UEII) Analysis")
    st.markdown(
        """
        An interactive web app for calculating the Urban Expansion Intensity Index (UEII) using 
        [USGS NLCD](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2019_REL_NLCD) data from Google Earth Engine. 
        The app was built using [streamlit](https://streamlit.io), [geemap](https://geemap.org), and [Google Earth Engine](https://earthengine.google.com).
        """
    )
    
    # Initialize Earth Engine
    ee_authenticate(token_name="EARTHENGINE_TOKEN")
    
    # Get Florida ROIs
    florida_rois = get_florida_counties()
    
    # Create layout
    row1_col1, row1_col2 = st.columns([2, 1])
    
    # Reset session state variables
    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

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
        st.header("⚙️ Configuration")
        
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
                folium.Marker(location=[lat, lng], popup=location).add_to(m)
                m.set_center(lng, lat, 12)
                st.session_state["zoom_level"] = 12
        
        # Region selection with proper options
        roi_options = ["Uploaded GeoJSON"] + list(florida_rois.keys())
        
        selected_region = st.selectbox(
            "Select a sample ROI or upload a GeoJSON file:",
            roi_options,
            index=0
        )
        
        # Time period selection with full year range
        st.subheader("📅 Time Period Configuration")
        available_years = list(range(1985, 2025))  # 1985 to 2024
        
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox("Start Year:", available_years, index=0)  # Default to 1985
        with col2:
            end_year = st.selectbox("End Year:", available_years, index=len(available_years) - 1)  # Default to 2024
        
        if start_year >= end_year:
            st.error("End year must be after start year!")
            st.stop()
        
        # Filter to available NLCD years within the selected range
        nlcd_years = [1985, 1995, 2005, 2015, 2024]
        target_years = [year for year in nlcd_years if start_year <= year <= end_year]
        
        if len(target_years) < 2:
            st.warning(f"Selected range contains fewer than 2 NLCD data points. Available years: {nlcd_years}")
            st.info(f"Current selection will use: {target_years}")
        else:
            st.info(f"Analysis will use {len(target_years)} time points: {target_years}")
        
        # Analysis options
        st.subheader("🔬 Analysis Options")
        include_visualization = st.checkbox("Create Visualizations", True)
        include_summary = st.checkbox("Generate Summary Statistics", True)

    # File upload section above the map
    with row1_col1:
        with st.expander("Steps: Draw a rectangle on the map → Export it as a GeoJSON → Upload it back to the app → Click the Submit button.", False):
            st.info("Use the drawing tools on the map to create a custom region, export as GeoJSON, then upload below.")

        data = st.file_uploader(
            "Upload a GeoJSON file to use as an ROI.", 
            type=["geojson", "kml", "zip"],
            help="Limit 200MB per file • GEOJSON, KML, ZIP"
        )
        
        crs = "epsg:4326"
        if selected_region == "Uploaded GeoJSON":
            if data:
                try:
                    gdf = uploaded_file_to_gdf(data)
                    st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
                    m.add_gdf(gdf, "ROI")
                    bounds = gdf.total_bounds
                    m.set_center((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 9)
                    st.success(f"✅ Uploaded GeoJSON: {data.name}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
            else:
                st.info("👆 Please upload a GeoJSON file above.")
        else:
            geometry = florida_rois.get(selected_region)
            if geometry:
                geojson_geom = geometry.getInfo() if hasattr(geometry, 'getInfo') else geometry
                shapely_geom = shape(geojson_geom)
                gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[shapely_geom])
                st.session_state["roi"] = geemap.gdf_to_ee(gdf, geodesic=False)
                m.add_gdf(gdf, "ROI")
                bounds = gdf.total_bounds
                zoom = 6 if "Florida" in selected_region else 9
                m.set_center((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, zoom)
        
        m.to_streamlit(height=600)

    # Submit button and analysis
    with row1_col2:
        with st.form("submit_ueii_form"):
            with st.expander("Customize UEII Analysis", True):
                title = st.text_input("Analysis Title:", f"UEII Analysis - {selected_region}")
                
                # Processing options
                st.write("**Processing Options:**")
                if selected_region == "Florida shapefile geometry":
                    st.info("Info: Florida analysis will process all 67 counties individually.")
                elif "County" in selected_region:
                    st.info(f"Info: Single county analysis for {selected_region}.")
                else:
                    st.info("Info: Custom region analysis.")
                
                include_maps = st.checkbox("Generate Choropleth Maps (Florida only)", True)
                
            submitted = st.form_submit_button("🚀 Submit UEII Analysis", type="primary")
            
            if submitted:
                roi = st.session_state.get("roi")
                if not roi and selected_region == "Uploaded GeoJSON":
                    st.error("Please upload a GeoJSON file or select a predefined region.")
                    st.stop()
                
                if len(target_years) < 2:
                    st.error("Need at least 2 time points for UEII calculation.")
                    st.stop()
                
                empty_text = st.empty()
                empty_text.text("🔄 Computing UEII... Please wait...")
                
                try:
                    # Get region geometry
                    if selected_region == "Uploaded GeoJSON":
                        region_geometry = roi
                        region_name = "Custom Region"
                    else:
                        region_geometry = geemap.gdf_to_ee(gdf, geodesic=False) if 'gdf' in locals() else roi
                        region_name = selected_region
                    
                    # Perform UEII analysis
                    results_df = analyze_region_batch(region_geometry, region_name, target_years)
                    
                    # Store results in session state
                    st.session_state["ueii_results"] = results_df
                    
                    empty_text.text("✅ Analysis completed successfully!")
                    st.success(f"🎉 UEII analysis completed for {len(results_df)} region(s)!")
                    
                except Exception as e:
                    empty_text.text(f"❌ Analysis failed: {str(e)}")
                    st.error(f"Error details: {str(e)}")

    # Display results if available
    if st.session_state["ueii_results"] is not None:
        results_df = st.session_state["ueii_results"]
        
        st.header("📋 UEII Analysis Results")
        
        # Display results table
        st.subheader(f"📊 Detailed Results ({len(results_df)} regions)")
        
        # Format numeric columns
        numeric_columns = results_df.select_dtypes(include=['float64', 'int64']).columns
        results_display = results_df.copy()
        for col in numeric_columns:
            if 'Area' in col or 'UEII' in col:
                results_display[col] = results_display[col].round(4)
        
        st.dataframe(results_display, use_container_width=True)
        
        # Download button
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"ueii_results_{selected_region.replace(' ', '_').replace('/', '_')}.csv",
            mime="text/csv"
        )
        
        # Create visualizations if requested
        if include_visualization and len(results_df) > 0:
            create_ueii_visualization(results_df)
        
        # Generate summary statistics if requested
        if include_summary and len(results_df) > 0:
            create_summary_statistics(results_df)
        
        # Key metrics
        st.header("🔍 Key Findings")
        ueii_columns = [col for col in results_df.columns if 'UEII' in col and '(%)' in col]
        
        if ueii_columns and len(results_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Regions Analyzed",
                    value=len(results_df),
                    help="Total number of regions in the analysis"
                )
            
            with col2:
                latest_period = ueii_columns[-1]
                avg_ueii = results_df[latest_period].mean()
                st.metric(
                    label="Average UEII",
                    value=f"{avg_ueii:.3f}%",
                    help=f"Average UEII for {latest_period.replace('UEII ', '').replace(' (%)', '')}"
                )
            
            with col3:
                max_ueii = results_df[latest_period].max()
                st.metric(
                    label="Highest UEII",
                    value=f"{max_ueii:.3f}%",
                    help="Maximum urban expansion intensity observed"
                )
            
            with col4:
                time_span = f"{target_years[0]}-{target_years[-1]}"
                st.metric(
                    label="Analysis Period",
                    value=time_span,
                    help="Time period covered by the analysis"
                )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
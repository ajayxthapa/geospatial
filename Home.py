import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="Urban Heat Island Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# --- Main Title ---
st.markdown("<h1 style='text-align: center;'>Spatiotemporal Analysis of Urban Heat and Land Cover Dynamics in the U.S.</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>A GEE-Powered Web Application (1985–2024)</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Introduction ---
st.subheader("🌎 Welcome to the Urban Analysis Toolkit")
st.markdown(
    """
    This interactive web application leverages the power of **Google Earth Engine (GEE)** and **Streamlit** to explore nearly four decades of environmental change across the U.S.

    Analyze long-term trends in:
    - 🌡️ Land Surface Temperature (LST)
    - 🌿 Vegetation Health (NDVI)
    - 🏙️ Urban Expansion (NDBI)
    - 🔥 Urban Hotspots (UHS)
    - 🧪 UTFVI Index

    Use these tools to support climate change adaptation and urban planning.
    """
)

# --- Key Features ---
st.subheader("🧰 Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 Multi-Parameter Analysis")
    st.info("Investigate LST, NDVI, NDBI, UHS, and UTFVI with just a few clicks.")

with col2:
    st.markdown("### 🗺️ State & County Level Detail")
    st.info("Perform fine-scale analysis for specific locations and download visualizations.")

with col3:
    st.markdown("### 📽️ Dynamic Visualizations")
    st.info("Create and export maps and GIF animations based on a custom time range.")

# --- How It Works ---
st.subheader("⚙️ How It Works")
st.markdown(
    """
    This app is powered by:
    - **Landsat Satellite Imagery (1985–2024)**: Landsat 5, 7, 8, and 9
    - **Google Earth Engine (GEE)** for cloud-based geospatial processing
    - **Python + Streamlit** for interactive web delivery

    All data processing is cloud-based, eliminating the need for local downloads.
    """
)

# --- Timelapse Previews ---
st.subheader("🕓 Featured Visualizations")

gif_col1, gif_col2 = st.columns(2)
with gif_col1:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/las_vegas.gif", caption="Urban Growth in Las Vegas, NV")

with gif_col2:
    st.image("https://github.com/giswqs/data/raw/main/timelapse/spain.gif", caption="Reservoir Changes in Spain")


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px; padding: 10px;">
        © 2024 Ajay Kumar Thapa | <a href="https://github.com/ajayxthapa/urbanheatisland" target="_blank">GitHub</a> | <a href="mailto:athapa2024@fau.edu">Contact</a>
    </div>
    """,
    unsafe_allow_html=True
)

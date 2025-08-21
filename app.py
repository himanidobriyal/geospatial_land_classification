# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import rasterio
# import numpy as np
# import re
# from scipy.ndimage import binary_dilation

# # ---------------------------
# # Class Mapping and Color Palette
# # ---------------------------
# # Define mapping from class names to pixel values and assign display colors
# CLASS_MAPPING = {
#     "Outside": -1,
#     "Green Space": 0,
#     "Built Space": 1,
#     "Impervious Space": 2,
#     "Permeable Space": 3,
#     "Other": 4
# }

# CLASS_COLOR_PALETTE = {
#     "Outside": ("black", -1),
#     "Green Space": ("green", 0),
#     "Built Space": ("gray", 1),
#     "Impervious Space": ("yellow", 2),
#     "Permeable Space": ("red", 3),
#     "Other": ("blue", 4)
# }

# # Create a colormap and normalization for image display
# class_names = list(CLASS_MAPPING.keys())
# class_colors = [CLASS_COLOR_PALETTE[name][0] for name in class_names]
# cmap = ListedColormap(class_colors)
# bounds = [v - 0.5 for v in range(-1, len(class_colors))]
# norm = BoundaryNorm(bounds, cmap.N)

# # ---------------------------
# # Session State Initialization
# # ---------------------------
# # Initialize session state to manage instruction toggle
# if "show_instructions" not in st.session_state:
#     st.session_state.show_instructions = False

# # ---------------------------
# # Utility Functions
# # ---------------------------
# def extract_year(name):
#     """Extracts the first 4-digit year found in a filename."""
#     match = re.search(r"(19|20)\d{2}", name)
#     return match.group(0) if match else "Unknown"

# def get_class_mask(file, class_name):
#     """Returns a boolean mask where pixels match the specified class name."""
#     with rasterio.open(file) as src:
#         data = src.read(1)
#         nodata = src.nodata
#         if nodata is not None:
#             data = np.where(data == nodata, -1, data)
#         else:
#             data = np.where(data == 0, -1, data)
#         pixel_value = CLASS_MAPPING[class_name]
#         return (data == pixel_value)

# def calculate_area(file, class_name, debug=False):
#     """Calculates area in km² of a given class in the raster image."""
#     try:
#         with rasterio.open(file) as src:
#             image = src.read(1)
#             pixel_size_x = abs(src.res[0])
#             pixel_size_y = abs(src.res[1])
#             pixel_value = CLASS_MAPPING[class_name]
#             nodata = src.nodata

#             if nodata is not None:
#                 valid_mask = image != nodata
#             else:
#                 valid_mask = image != 0
#                 valid_mask = binary_dilation(valid_mask, iterations=3)

#             mask = (image == pixel_value) & valid_mask
#             num_pixels = np.sum(mask)

#             if debug:
#                 st.write(f"[DEBUG] Unique values: {np.unique(image)}")
#                 st.write(f"[DEBUG] '{class_name}' pixel count: {num_pixels}")
#                 st.write(f"[DEBUG] Nodata: {nodata}")

#             return num_pixels * pixel_size_x * pixel_size_y / 1e6

#     except Exception as e:
#         st.error(f"Error calculating area: {e}")
#         return 0

# def calculate_total_valid_area(file, debug=False):
#     """Calculates the total valid area (non-nodata) in km² for a raster image."""
#     try:
#         with rasterio.open(file) as src:
#             band = src.read(1)
#             pixel_size_x = abs(src.res[0])
#             pixel_size_y = abs(src.res[1])
#             nodata = src.nodata

#             if nodata is not None:
#                 valid_mask = band != nodata
#             else:
#                 valid_mask = band != 0
#                 valid_mask = binary_dilation(valid_mask, iterations=3)

#             num_valid_pixels = np.sum(valid_mask)
#             area_km2 = (num_valid_pixels * pixel_size_x * pixel_size_y) / 1e6

#             if debug:
#                 st.write(f"[DEBUG] CRS: {src.crs}")
#                 st.write(f"[DEBUG] Pixel Size: {pixel_size_x} × {pixel_size_y} meters")
#                 st.write(f"[DEBUG] Valid Pixels: {num_valid_pixels}")
#                 st.write(f"[DEBUG] Total Valid Area: {area_km2:.4f} km²")

#             return area_km2

#     except Exception as e:
#         st.error(f"Error calculating valid polygon area: {e}")
#         return 0

# def render_legend():
#     """Displays a legend showing the meaning of each land classification color."""
#     legend_elements = [
#         mpatches.Patch(color=color, label=label)
#         for label, (color, _) in CLASS_COLOR_PALETTE.items()
#     ]
#     fig, ax = plt.subplots(figsize=(2.5, 1.5))
#     ax.axis('off')
#     ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=8)
#     st.pyplot(fig)

# def display_results(metric_name, debug=False):
#     """Displays a table and line chart for the selected land classification area over years."""
#     rows = []
#     for file in uploaded_files:
#         year = extract_year(file.name)
#         area = calculate_area(file, metric_name, debug)
#         rows.append({"Year": year, "Area (km²)": round(area, 4)})
#     df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
#     st.markdown(f"### Area for {metric_name}")
#     st.dataframe(df, use_container_width=True)

#     st.markdown("### Area Over Time")
#     fig, ax = plt.subplots(figsize=(5, 3))
#     ax.plot(df["Year"], df["Area (km²)"], marker='o', color='green')
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Area (km²)")
#     ax.set_title(f"{metric_name} Over Time")
#     ax.grid(True)
#     st.pyplot(fig)

# def display_index(index_name):
#     """Calculates and displays the selected greenness index (GIP, GBP, GSI) over time."""
#     rows = []
#     for file in uploaded_files:
#         year = extract_year(file.name)
#         with rasterio.open(file) as src:
#             data = src.read(1)
#             unique, counts = np.unique(data, return_counts=True)
#             class_counts = dict(zip(unique.tolist(), counts.tolist()))

#         green = class_counts.get(CLASS_MAPPING["Green Space"], 0)
#         built = class_counts.get(CLASS_MAPPING["Built Space"], 0)
#         impervious = class_counts.get(CLASS_MAPPING["Impervious Space"], 0)

#         if debug_mode:
#             st.write(f"[DEBUG] Year: {year} | Green: {green}, Built: {built}, Impervious: {impervious}")

#         gip = green / (green + impervious) if (green + impervious) > 0 else 0
#         gbp = green / (green + built) if (green + built) > 0 else 0
#         gsi = (gip + gbp) / 2 if (gip + gbp) > 0 else 0

#         value = {"GIP": gip, "GBP": gbp, "GSI": gsi}[index_name]
#         rows.append({"Year": year, index_name: round(value, 4)})

#     df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
#     st.markdown(f"### {index_name} Index Over Time")
#     st.dataframe(df, use_container_width=True)

#     st.markdown("### Index Trend")
#     fig, ax = plt.subplots()
#     ax.plot(df["Year"], df[index_name], marker='o', color='green')
#     ax.set_xlabel("Year")
#     ax.set_ylabel(index_name)
#     ax.set_title(f"{index_name} Trend Over Time")
#     ax.grid(True)
#     st.pyplot(fig)

# def display_total_area(debug=False):
#     """Displays total valid area (non-nodata) per image over time."""
#     rows = []
#     for file in uploaded_files:
#         year = extract_year(file.name)
#         area = calculate_total_valid_area(file, debug)
#         rows.append({"Year": year, "Total Valid Polygon Area (km²)": round(area, 4)})
#     df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
#     st.markdown("### Total Valid Polygon Area (Clipped Extent)")
#     st.dataframe(df, use_container_width=True)

# # ---------------------------
# # Streamlit Page Setup
# # ---------------------------
# st.set_page_config(page_title="Land Classification", layout="wide")

# if st.session_state.show_instructions:
#     st.title(" Export Instructions")
#     st.markdown("""
#     Use this Earth Engine code to export your classified raster:

#     ```javascript
#     Export.image.toDrive({
#       image: classified.clip(kota).rename('classification'),
#       description: 'classified_kota_' + year,
#       folder: 'EarthEngineExports',
#       fileNamePrefix: 'kota_classified_' + year,
#       region: kota.geometry(),
#       scale: 10,
#       crs: 'EPSG:32643',
#       fileFormat: 'GeoTIFF',
#       maxPixels: 1e13
#     });
#     ```
#     ✔️ **Adjust `scale` and `crs` as needed.**  
#     ✔️ **Check *Tasks* tab to download the export.**  
#     ✔️ **Always clip to your polygon for accurate area.**
#     """)
#     st.button("⬅Back", on_click=lambda: st.session_state.update({"show_instructions": False}))
#     st.stop()

# # ---------------------------
# # Application Main Interface
# # ---------------------------
# st.title("Land Classification Area Calculator")

# # Sidebar interface
# st.sidebar.header("Upload & Analyze")
# if st.sidebar.button(" Instructions"):
#     st.session_state.show_instructions = True

# uploaded_files = st.sidebar.file_uploader(
#     "Upload Classified .tif Files",
#     type=["tif", "tiff"],
#     accept_multiple_files=True
# )
# debug_mode = st.sidebar.toggle("Enable Debug Logs", value=False)

# button_pressed = None

# st.sidebar.subheader("Area Calculation")
# for metric in ["Green Space", "Built Space", "Impervious Space", "Permeable Space", "Other"]:
#     if st.sidebar.button(metric):
#         button_pressed = metric

# st.sidebar.subheader("Index Calculation")
# for index in ["GIP", "GBP", "GSI"]:
#     if st.sidebar.button(index):
#         button_pressed = index

# st.sidebar.subheader(" Shapefile Extent")
# if st.sidebar.button("Total Valid Area"):
#     button_pressed = "Total Area"

# # ---------------------------
# # Main Display Area
# # ---------------------------
# if uploaded_files:
#     if button_pressed in CLASS_MAPPING and button_pressed != "Outside":
#         display_results(button_pressed, debug=debug_mode)
#     elif button_pressed in ["GIP", "GBP", "GSI"]:
#         display_index(button_pressed)
#     elif button_pressed == "Total Area":
#         display_total_area(debug=debug_mode)

#     st.markdown("## Legend")
#     def render_legend_small():
#         legend_elements = [
#             mpatches.Patch(color=color, label=label)
#             for label, (color, _) in CLASS_COLOR_PALETTE.items()
#         ]
#         fig, ax = plt.subplots(figsize=(2.5, 0.8))
#         ax.axis('off')
#         ax.legend(
#             handles=legend_elements,
#             loc='center',
#             frameon=True,
#             fontsize=7,
#             ncol=3
#         )
#         st.pyplot(fig)

#     render_legend_small()

#     st.markdown("## All Uploaded Classified Images (Grid View)")
#     cols_per_row = 2
#     rows = (len(uploaded_files) + cols_per_row - 1) // cols_per_row

#     for row in range(rows):
#         cols = st.columns(cols_per_row)
#         for i in range(cols_per_row):
#             idx = row * cols_per_row + i
#             if idx < len(uploaded_files):
#                 file = uploaded_files[idx]
#                 with rasterio.open(file) as src:
#                     arr = src.read(1)
#                     nodata = src.nodata
#                     if nodata is not None:
#                         arr = np.where(arr == nodata, -1, arr)
#                     else:
#                         arr = np.where(arr == 0, -1, arr)
#                 with cols[i]:
#                     st.markdown(f"**{file.name}**")
#                     fig, ax = plt.subplots(figsize=(4, 4))
#                     ax.imshow(arr, cmap=cmap, norm=norm)
#                     ax.axis("off")
#                     st.pyplot(fig)
# else:
#     st.info("Please upload classified .tif files from the sidebar to begin analysis.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import numpy as np
import re
from scipy.ndimage import binary_dilation


# Class Mapping and Color Palette
CLASS_MAPPING = {
    "Outside": -1,
    "Green Space": 0,
    "Built Space": 1,
    "Impervious Space": 2,
    "Permeable Space": 3,
    "Other": 4
}

CLASS_COLOR_PALETTE = {
    "Outside": ("black", -1),
    "Green Space": ("green", 0),
    "Built Space": ("gray", 1),
    "Impervious Space": ("yellow", 2),
    "Permeable Space": ("red", 3),
    "Other": ("blue", 4)
}

class_names = list(CLASS_MAPPING.keys())
class_colors = [CLASS_COLOR_PALETTE[name][0] for name in class_names]
cmap = ListedColormap(class_colors)
bounds = [v - 0.5 for v in range(-1, len(class_colors))]
norm = BoundaryNorm(bounds, cmap.N)


# Session State Initialization
if "show_instructions" not in st.session_state:
    st.session_state.show_instructions = False


# Utility Functions
def extract_year(name):
    match = re.search(r"(19|20)\d{2}", name)
    return match.group(0) if match else "Unknown"

def get_class_mask(file, class_name):
    with rasterio.open(file) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, -1, data)
        else:
            data = np.where(data == 0, -1, data)
        pixel_value = CLASS_MAPPING[class_name]
        return (data == pixel_value)

def calculate_area(file, class_name, debug=False):
    try:
        with rasterio.open(file) as src:
            image = src.read(1)
            pixel_size_x = abs(src.res[0])
            pixel_size_y = abs(src.res[1])
            pixel_value = CLASS_MAPPING[class_name]
            nodata = src.nodata

            if nodata is not None:
                valid_mask = image != nodata
            else:
                valid_mask = image != 0
                valid_mask = binary_dilation(valid_mask, iterations=3)

            mask = (image == pixel_value) & valid_mask
            num_pixels = np.sum(mask)

            if debug:
                st.write(f"[DEBUG] Unique values: {np.unique(image)}")
                st.write(f"[DEBUG] '{class_name}' pixel count: {num_pixels}")
                st.write(f"[DEBUG] Nodata: {nodata}")

            return num_pixels * pixel_size_x * pixel_size_y / 1e6

    except Exception as e:
        st.error(f"Error calculating area: {e}")
        return 0

def calculate_total_valid_area(file, debug=False):
    try:
        with rasterio.open(file) as src:
            band = src.read(1)
            pixel_size_x = abs(src.res[0])
            pixel_size_y = abs(src.res[1])
            nodata = src.nodata

            if nodata is not None:
                valid_mask = band != nodata
            else:
                valid_mask = band != 0
                valid_mask = binary_dilation(valid_mask, iterations=3)

            num_valid_pixels = np.sum(valid_mask)
            area_km2 = (num_valid_pixels * pixel_size_x * pixel_size_y) / 1e6

            if debug:
                st.write(f"[DEBUG] CRS: {src.crs}")
                st.write(f"[DEBUG] Pixel Size: {pixel_size_x} × {pixel_size_y} meters")
                st.write(f"[DEBUG] Valid Pixels: {num_valid_pixels}")
                st.write(f"[DEBUG] Total Valid Area: {area_km2:.4f} km²")

            return area_km2

    except Exception as e:
        st.error(f"Error calculating valid polygon area: {e}")
        return 0

def render_legend():
    legend_elements = [
        mpatches.Patch(color=color, label=label)
        for label, (color, _) in CLASS_COLOR_PALETTE.items()
    ]
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=8)
    st.pyplot(fig)

def display_results(metric_name, debug=False):
    rows = []
    for file in uploaded_files:
        year = extract_year(file.name)
        area = calculate_area(file, metric_name, debug)
        rows.append({"Year": year, "Area (km²)": round(area, 4)})
    df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
    st.markdown(f"### Area for {metric_name}")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Area Over Time")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["Year"], df["Area (km²)"], marker='o', color='green')
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.set_title(f"{metric_name} Over Time")
    ax.grid(True)
    st.pyplot(fig)

def display_index(index_name):
    rows = []
    for file in uploaded_files:
        year = extract_year(file.name)
        with rasterio.open(file) as src:
            data = src.read(1)
            unique, counts = np.unique(data, return_counts=True)
            class_counts = dict(zip(unique.tolist(), counts.tolist()))

        green = class_counts.get(CLASS_MAPPING["Green Space"], 0)
        built = class_counts.get(CLASS_MAPPING["Built Space"], 0)
        impervious = class_counts.get(CLASS_MAPPING["Impervious Space"], 0)

        if debug_mode:
            st.write(f"[DEBUG] Year: {year} | Green: {green}, Built: {built}, Impervious: {impervious}")

        gip = green / (green + impervious) if (green + impervious) > 0 else 0
        gbp = green / (green + built) if (green + built) > 0 else 0
        gsi = (gip + gbp) / 2 if (gip + gbp) > 0 else 0

        value = {"GIP": gip, "GBP": gbp, "GSI": gsi}[index_name]
        rows.append({"Year": year, index_name: round(value, 4)})

    df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
    st.markdown(f"### {index_name} Index Over Time")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Index Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Year"], df[index_name], marker='o', color='green')
    ax.set_xlabel("Year")
    ax.set_ylabel(index_name)
    ax.set_title(f"{index_name} Trend Over Time")
    ax.grid(True)
    st.pyplot(fig)

def display_total_area(debug=False):
    rows = []
    for file in uploaded_files:
        year = extract_year(file.name)
        area = calculate_total_valid_area(file, debug)
        rows.append({"Year": year, "Total Valid Polygon Area (km²)": round(area, 4)})
    df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
    st.markdown("### Total Valid Polygon Area (Clipped Extent)")
    st.dataframe(df, use_container_width=True)


# Streamlit Page Setup
st.set_page_config(page_title="Land Classification", layout="wide")

if st.session_state.show_instructions:
    st.title("Export & Usage Instructions")
    st.markdown("""
    ### How to Use This App  
    1. **Upload Files** – On the left sidebar, click the **Browse** button under *Upload Classified .tif Files* and select the classified raster files you want to analyze.  
    2. **Wait for Loading** – Allow all selected images to load completely. Loading time may vary depending on file size and the area of the city.  
    3. **Calculate Areas** – Once images are loaded, click on the desired **Area Calculation** button (e.g., *Green Space*, *Built Space*, etc.) in the sidebar.  
    4. **Calculate Indexes** – Similarly, click on any **Index Calculation** button (*GIP*, *GBP*, *GSI*) to compute the selected index.  
    5. **Note** – Area or index calculations can only be performed once images have fully loaded.  

    ---

    ### Google Earth Engine Export Code  
    Use this Earth Engine code to export your classified raster:

    ```javascript
    Export.image.toDrive({
      image: classified.clip(kota).rename('classification'),
      description: 'classified_kota_' + year,
      folder: 'EarthEngineExports',
      fileNamePrefix: 'kota_classified_' + year,
      region: kota.geometry(),
      scale: 10,
      crs: 'EPSG:32643',
      fileFormat: 'GeoTIFF',
      maxPixels: 1e13
    });
    ```
    ✔️ **Adjust `scale` and `crs` as needed.**  
    ✔️ **Check the *Tasks* tab in Earth Engine to download the export.**  
    ✔️ **Always clip to your polygon for accurate area results.**
    """)
    st.button("⬅ Back", on_click=lambda: st.session_state.update({"show_instructions": False}))
    st.stop()

# Application Main Interface
st.title("Land Classification Area Calculator")

# Sidebar
st.sidebar.header("Upload & Analyze")
if st.sidebar.button(" Instructions"):
    st.session_state.show_instructions = True

uploaded_files = st.sidebar.file_uploader(
    "Upload Classified .tif Files",
    type=["tif", "tiff"],
    accept_multiple_files=True
)
debug_mode = st.sidebar.toggle("Enable Debug Logs", value=False)

button_pressed = None

st.sidebar.subheader("Area Calculation")
for metric in ["Green Space", "Built Space", "Impervious Space", "Permeable Space", "Other"]:
    if st.sidebar.button(metric):
        button_pressed = metric

st.sidebar.subheader("Index Calculation")
index_labels = {
    "GIP": "GIP – Green vs. Impervious Surface Proportion",
    "GBP": "GBP – Green vs. Built Spaces Proportion",
    "GSI": "GSI – Green Index"
}
for index_key, index_label in index_labels.items():
    if st.sidebar.button(index_label):
        button_pressed = index_key

st.sidebar.subheader(" Shapefile Extent")
if st.sidebar.button("Total Valid Area"):
    button_pressed = "Total Area"

# Main Display
if uploaded_files:
    if button_pressed in CLASS_MAPPING and button_pressed != "Outside":
        display_results(button_pressed, debug=debug_mode)
    elif button_pressed in ["GIP", "GBP", "GSI"]:
        display_index(button_pressed)
    elif button_pressed == "Total Area":
        display_total_area(debug=debug_mode)

    st.markdown("## Legend")
    def render_legend_small():
        legend_elements = [
            mpatches.Patch(color=color, label=label)
            for label, (color, _) in CLASS_COLOR_PALETTE.items()
        ]
        fig, ax = plt.subplots(figsize=(2.5, 0.8))
        ax.axis('off')
        ax.legend(
            handles=legend_elements,
            loc='center',
            frameon=True,
            fontsize=7,
            ncol=3
        )
        st.pyplot(fig)

    render_legend_small()

    st.markdown("## All Uploaded Classified Images (Grid View)")
    cols_per_row = 2
    rows = (len(uploaded_files) + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = row * cols_per_row + i
            if idx < len(uploaded_files):
                file = uploaded_files[idx]
                with rasterio.open(file) as src:
                    arr = src.read(1)
                    nodata = src.nodata
                    if nodata is not None:
                        arr = np.where(arr == nodata, -1, arr)
                    else:
                        arr = np.where(arr == 0, -1, arr)
                with cols[i]:
                    st.markdown(f"**{file.name}**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(arr, cmap=cmap, norm=norm)
                    ax.axis("off")
                    st.pyplot(fig)
else:
    st.info("Please upload classified .tif files from the sidebar to begin analysis.")
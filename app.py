import io
import os
import re
import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import rasterio
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
from contextlib import contextmanager

# Try to import fiona (for shapefile support)
try:
    import fiona
except Exception:
    fiona = None

# Class mapping & colors

CLASS_MAPPING = {
    "Outside": -1,
    "Green Space": 0,
    "Built Space": 1,
    "Impervious Space": 2,
    "Permeable Space": 3,
    "Other": 4,
}
CLASS_COLOR_PALETTE = {
    "Outside": ("black", -1),
    "Green Space": ("green", 0),
    "Built Space": ("gray", 1),
    "Impervious Space": ("yellow", 2),
    "Permeable Space": ("red", 3),
    "Other": ("blue", 4),
}
class_names = list(CLASS_MAPPING.keys())
class_colors = [CLASS_COLOR_PALETTE[name][0] for name in class_names]
cmap = ListedColormap(class_colors)
bounds = [v - 0.5 for v in range(-1, len(class_colors))]
norm = BoundaryNorm(bounds, cmap.N)

# Helpers

def extract_year(name: str):
    m = re.search(r"(19|20)\d{2}", name)
    return m.group(0) if m else "Unknown"

@contextmanager
def open_raster_with_reproject(uploaded_file, debug=False):
    """
    Open a raster; if CRS is geographic (e.g., EPSG:4326),
    reproject to a suitable UTM in-memory. Yields (dataset, meta).
    """
    src = rasterio.open(uploaded_file)
    try:
        if src.crs is None:
            src.close()
            raise ValueError("Source CRS is undefined for uploaded file.")

        crs_obj = CRS.from_user_input(src.crs)

        if crs_obj.is_projected:
            if debug: st.write(f"[DEBUG] CRS already projected: {src.crs}")
            yield src, {"original_crs": src.crs, "used_epsg": str(src.crs)}
            return

        # Compute UTM from centroid
        b = src.bounds
        lon = (b.left + b.right) / 2.0
        lat = (b.top + b.bottom) / 2.0
        zone = int((lon + 180) / 6) + 1
        epsg_code = 32600 + zone if lat >= 0 else 32700 + zone
        utm_crs = CRS.from_epsg(epsg_code)

        if debug:
            st.write(f"[DEBUG] Original CRS: {src.crs}")
            st.write(f"[DEBUG] Centroid lon/lat: {lon:.6f}, {lat:.6f}")
            st.write(f"[DEBUG] Auto UTM EPSG: {epsg_code}")

        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": utm_crs.to_wkt(), "transform": transform, "width": width, "height": height})

        memfile = MemoryFile()
        dst = memfile.open(**kwargs)
        try:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs.to_wkt(),
                    resampling=Resampling.nearest,
                )
            dst.close()
            reopened = memfile.open()
            try:
                yield reopened, {"original_crs": src.crs, "used_epsg": f"EPSG:{epsg_code}"}
            finally:
                reopened.close()
        finally:
            memfile.close()
    finally:
        try:
            src.close()
        except Exception:
            pass

def valid_data_mask(src):
    """Valid pixels from alpha/valid mask or nodata. If neither exists, all valid."""
    try:
        m = src.read_masks(1)
        return (m > 0)
    except Exception:
        pass
    band = src.read(1)
    nodata = src.nodata
    if nodata is not None:
        return (band != nodata)
    return np.ones_like(band, dtype=bool)

# ---- Shapefile ZIP → GeoJSON-like dict, detect EPSG ----
def load_shapefile_zip_return_geojson(uploaded_zip):
    """
    Reads a zipped shapefile and returns (FeatureCollection-like dict, detected_epsg).
    Requires Fiona; if not present, raises a helpful error.
    """
    if fiona is None:
        raise RuntimeError(
            "Reading shapefiles requires the 'fiona' package. "
            "Install it (pip install fiona) or use a GeoJSON instead."
        )

    mem = io.BytesIO(uploaded_zip.read())
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(mem) as zf:
            zf.extractall(td)
        shp_path = None
        for r, _, files in os.walk(td):
            for fn in files:
                if fn.lower().endswith(".shp"):
                    shp_path = os.path.join(r, fn)
                    break
            if shp_path:
                break
        if not shp_path:
            raise RuntimeError("No .shp found inside the ZIP.")

        feats = []
        with fiona.open(shp_path) as src:
            # Get CRS (WKT or dict); fall back to EPSG:4326 if absent
            crs_src = CRS.from_user_input(src.crs_wkt or src.crs) if (src.crs_wkt or src.crs) else CRS.from_epsg(4326)
            for ft in src:
                feats.append({
                    "type": "Feature",
                    "geometry": ft["geometry"],
                    "properties": ft.get("properties", {})
                })
        return {"type": "FeatureCollection", "features": feats}, (crs_src.to_epsg() or 4326)

# ---- Reproject AOI GeoJSON -> raster CRS and rasterize ----
def _transform_coords(coords, transformer):
    if isinstance(coords[0], (float, int)):
        x, y = coords[0], coords[1]
        X, Y = transformer.transform(x, y)  # GeoJSON is (lon,lat)
        return [X, Y]
    return [_transform_coords(c, transformer) for c in coords]

def reproject_geojson_to(src_crs, aoi_geojson, aoi_epsg):
    in_crs = CRS.from_epsg(aoi_epsg)
    out_crs = CRS.from_user_input(src_crs)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)

    geoms = []
    if aoi_geojson.get("type") == "FeatureCollection":
        feats = aoi_geojson["features"]
    elif aoi_geojson.get("type") == "Feature":
        feats = [aoi_geojson]
    else:
        feats = [{"type": "Feature", "geometry": aoi_geojson, "properties": {}}]

    for f in feats:
        geom = f["geometry"]
        if geom is None:
            continue
        new_geom = {
            "type": geom["type"],
            "coordinates": _transform_coords(geom["coordinates"], transformer)
        }
        geoms.append(new_geom)
    return geoms

def make_aoi_mask_from_geojson(src, aoi_geojson_dict, aoi_epsg):
    geoms = reproject_geojson_to(src.crs, aoi_geojson_dict, aoi_epsg)
    mask = geometry_mask(
        geoms,
        out_shape=(src.height, src.width),
        transform=src.transform,
        invert=True,          # True => inside polygons is True
        all_touched=False
    )
    return mask

def replace_nodata_with_outside_for_display(band, src, aoi_mask=None):
    """For preview: set masked-out/nodata/outside AOI to -1 (Outside)."""
    base_valid = valid_data_mask(src)
    if aoi_mask is not None:
        base_valid = base_valid & aoi_mask
    out = band.copy()
    out[~base_valid] = -1
    if src.nodata is not None:
        out[band == src.nodata] = -1
    return out


# Area & Index computations

def calculate_area(file, class_name, aoi_geo=None, aoi_epsg=4326, debug=False):
    try:
        with open_raster_with_reproject(file, debug=debug) as (src, meta):
            band = src.read(1)
            mask = valid_data_mask(src)
            if aoi_geo is not None:
                mask = mask & make_aoi_mask_from_geojson(src, aoi_geo, aoi_epsg)

            px, py = abs(src.res[0]), abs(src.res[1])
            class_val = CLASS_MAPPING[class_name]
            num = int(np.sum((band == class_val) & mask))

            if debug:
                st.write(f"[DEBUG] File: {getattr(file, 'name', 'uploaded')}")
                st.write(f"[DEBUG] Meta: {meta}")
                st.write(f"[DEBUG] Pixel size (m): {px:.3f} × {py:.3f}")
                st.write(f"[DEBUG] '{class_name}' pixels (inside AOI): {num}")
                st.write(f"[DEBUG] nodata: {src.nodata}")

            return (num * px * py) / 1e6  # km²
    except Exception as e:
        st.error(f"Error calculating area: {e}")
        return 0.0

def calculate_total_valid_area(file, aoi_geo=None, aoi_epsg=4326, debug=False):
    try:
        with open_raster_with_reproject(file, debug=debug) as (src, meta):
            mask = valid_data_mask(src)
            if aoi_geo is not None:
                mask = mask & make_aoi_mask_from_geojson(src, aoi_geo, aoi_epsg)

            px, py = abs(src.res[0]), abs(src.res[1])
            num_valid = int(np.sum(mask))
            area_km2 = (num_valid * px * py) / 1e6

            if debug:
                st.write(f"[DEBUG] Meta: {meta}")
                st.write(f"[DEBUG] Pixel size: {px:.3f} × {py:.3f} m")
                st.write(f"[DEBUG] Valid (inside AOI) pixels: {num_valid}")
                st.write(f"[DEBUG] Total Valid Area in AOI (km²): {area_km2:.6f}")

            return area_km2
    except Exception as e:
        st.error(f"Error calculating valid area: {e}")
        return 0.0

def compute_class_counts_from_file(file, aoi_geo=None, aoi_epsg=4326, debug=False):
    with open_raster_with_reproject(file, debug=debug) as (src, meta):
        band = src.read(1)
        mask = valid_data_mask(src)
        if aoi_geo is not None:
            mask = mask & make_aoi_mask_from_geojson(src, aoi_geo, aoi_epsg)
        vals = band[mask]
        unique, counts = np.unique(vals, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist())), meta

def calculate_indexes_for_file(file, aoi_geo=None, aoi_epsg=4326, debug=False):
    counts, meta = compute_class_counts_from_file(file, aoi_geo, aoi_epsg, debug)
    green = counts.get(CLASS_MAPPING["Green Space"], 0)
    built = counts.get(CLASS_MAPPING["Built Space"], 0)
    imperv = counts.get(CLASS_MAPPING["Impervious Space"], 0)

    if debug:
        st.write(f"[DEBUG] File: {getattr(file, 'name', 'uploaded')} | meta: {meta}")
        st.write(f"[DEBUG] (AOI) Green: {green}, Built: {built}, Impervious: {imperv}")

    gip = green / (green + imperv) if (green + imperv) > 0 else 0.0
    gbp = green / (green + built) if (green + built) > 0 else 0.0
    gsi = (gip + gbp) / 2.0 if (gip + gbp) > 0 else 0.0
    return {"GIP": gip, "GBP": gbp, "GSI": gsi}, meta


# Legend rendering

def render_legend_small():
    legend_elements = [mpatches.Patch(color=color, label=label)
                       for label, (color, _) in CLASS_COLOR_PALETTE.items()]
    fig, ax = plt.subplots(figsize=(3, 0.9))
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=8, ncol=3)
    st.pyplot(fig)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Land Classification )", layout="wide")
st.title("Land Classification Area & Index Calculator ")

# Sidebar controls
st.sidebar.header("Upload & Controls")

st.sidebar.markdown("---")
if st.sidebar.button("Show Instructions"):
    st.session_state.show_instructions = True
else:
    if "show_instructions" not in st.session_state:
        st.session_state.show_instructions = False

uploaded_files = st.sidebar.file_uploader(
    "Upload Classified GeoTIFF(s)", type=["tif", "tiff"], accept_multiple_files=True
)
debug_mode = st.sidebar.checkbox("Enable Debug Logs", value=False)

st.sidebar.subheader("City AOI — Shapefile (ZIP)")
aoi_zip = st.sidebar.file_uploader("Upload Shapefile as ZIP (.shp, .shx, .dbf, .prj)", type=["zip"])
aoi_geojson = None
aoi_epsg_detected = None

if aoi_zip is not None:
    try:
        aoi_geojson, aoi_epsg_detected = load_shapefile_zip_return_geojson(aoi_zip)
        st.sidebar.success(f"AOI loaded. Detected CRS: EPSG:{aoi_epsg_detected}")
    except Exception as e:
        st.sidebar.error(f"Failed to read shapefile: {e}")

st.sidebar.markdown("### Actions")
button_pressed = None

st.sidebar.subheader("Area Calculation")
for metric in ["Green Space", "Built Space", "Impervious Space", "Permeable Space", "Other"]:
    if st.sidebar.button(metric):
        button_pressed = metric

st.sidebar.subheader("Index Calculation")
index_labels = {"GIP": "GIP – Green vs Impervious", "GBP": "GBP – Green vs Built", "GSI": "GSI – Green Index"}
for idx_key, idx_label in index_labels.items():
    if st.sidebar.button(idx_label):
        button_pressed = idx_key

st.sidebar.subheader("Other")
if st.sidebar.button("Total Valid Area (inside AOI)"):
    button_pressed = "Total Area"

# st.sidebar.markdown("---")
# if st.sidebar.button("Show Instructions"):
#     st.session_state.show_instructions = True
# else:
#     if "show_instructions" not in st.session_state:
#         st.session_state.show_instructions = False

# Instructions panel
if st.session_state.get("show_instructions", False):
    st.header("Usage Instructions")
    st.markdown("""
                
    **Goal:** Compute areas & indices only *inside* your city boundary (AOI).

    1) Upload classified GeoTIFF(s).  
    2) Upload **Shapefile ZIP** containing `.shp .shx .dbf .prj`.  
    3) The app detects the shapefile CRS and clips to the AOI for **previews and calculations**.  
    4) Turn on debug logs for CRS/AOI details.
#   5) Wait for Loading Allow all selected images to load completely. Loading time may vary depending on file size and the area of the city.  
#   6)Calculate Areas Once images are loaded, click on the desired **Area Calculation** button (e.g., *Green Space*, *Built Space*, etc.) in the sidebar.  
#   7)Calculate Indexes Similarly, click on any **Index Calculation** button (*GIP*, *GBP*, *GSI*) to compute the selected index.  
#   8)**Note** – Area or index calculations can only be performed once images have fully loaded.  

#     ---

#     ### Google Earth Engine Export Code  
#     Use this Earth Engine code to export your classified raster:

#     ```javascript
#     Export.image.toDrive({
#       image: classified.clip(kota).rename('classification'),
#       description: 'classified_kota_' + year,
#       folder: 'EarthEngineExports',
#       fileNamePrefix: 'kota_classified_' + year,
#       region: kota.geometry(),
#       scale: 10,
#       crs: 'EPSG:4326',
#       fileFormat: 'GeoTIFF',
#       maxPixels: 1e13
#     });
#     ```
#     ✔️ **Adjust `scale` and `crs` as needed.**  
#     ✔️ **Check the *Tasks* tab in Earth Engine to download the export.**  
#     ✔️ **Always clip to your polygon for accurate area results.**
    """)
    if st.button("Close Instructions"):
        st.session_state.show_instructions = False
    st.stop()

# Main area
if uploaded_files:
    # Perform action if a button pressed
    if button_pressed and button_pressed in CLASS_MAPPING and button_pressed != "Outside":
        rows = []
        for file in uploaded_files:
            year = extract_year(getattr(file, "name", "uploaded"))
            area_km2 = calculate_area(file, button_pressed, aoi_geojson, aoi_epsg_detected or 4326, debug=debug_mode)
            rows.append({"Year": year, "Area (km²)": round(area_km2, 6)})
        df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
        st.markdown(f"### Area for {button_pressed} (inside AOI)")
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["Year"], df["Area (km²)"], marker='o')
        ax.set_xlabel("Year"); ax.set_ylabel("Area (km²)")
        ax.set_title(f"{button_pressed} Over Time (AOI)"); ax.grid(True)
        st.pyplot(fig)

    elif button_pressed in ["GIP", "GBP", "GSI"]:
        rows = []
        for file in uploaded_files:
            year = extract_year(getattr(file, "name", "uploaded"))
            idxs, _ = calculate_indexes_for_file(file, aoi_geojson, aoi_epsg_detected or 4326, debug=debug_mode)
            rows.append({"Year": year, button_pressed: round(idxs[button_pressed], 6)})
        df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
        st.markdown(f"### {button_pressed} Over Time (inside AOI)")
        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df["Year"], df[button_pressed], marker='o')
        ax.set_xlabel("Year"); ax.set_ylabel(button_pressed)
        ax.set_title(f"{button_pressed} Trend Over Time (AOI)"); ax.grid(True)
        st.pyplot(fig)

    elif button_pressed == "Total Area":
        rows = []
        for file in uploaded_files:
            year = extract_year(getattr(file, "name", "uploaded"))
            total_area = calculate_total_valid_area(file, aoi_geojson, aoi_epsg_detected or 4326, debug=debug_mode)
            rows.append({"Year": year, "Total Valid Area (km²)": round(total_area, 6)})
        df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
        st.markdown("### Total Valid Area (inside AOI)")
        st.dataframe(df, use_container_width=True)

    # Legend & previews
    st.markdown("## Legend")
    render_legend_small()

    st.markdown("## Uploaded Classified Images — Clipped Preview (AOI if provided)")
    cols_per_row = 2
    rows_count = (len(uploaded_files) + cols_per_row - 1) // cols_per_row
    for r in range(rows_count):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = r * cols_per_row + i
            if idx < len(uploaded_files):
                f = uploaded_files[idx]
                with open_raster_with_reproject(f, debug=debug_mode) as (src, meta):
                    band = src.read(1)
                    aoi_mask = None
                    if aoi_geojson is not None:
                        aoi_mask = make_aoi_mask_from_geojson(src, aoi_geojson, aoi_epsg_detected or 4326)
                    arr = replace_nodata_with_outside_for_display(band, src, aoi_mask=aoi_mask)

                    # optional sanity check
                    valid_vals = set(CLASS_MAPPING.values())
                    unknown = sorted(v for v in np.unique(arr) if v not in valid_vals)
                    if unknown:
                        st.warning(f"Unknown class values in {getattr(f,'name','uploaded')}: {unknown}")

                with cols[i]:
                    st.markdown(f"**{getattr(f, 'name', 'uploaded')}**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(arr, cmap=cmap, norm=norm)
                    ax.axis("off")
                    st.pyplot(fig)
else:
    st.info("Please upload classified .tif files from the sidebar to begin analysis.")




# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import rasterio
# import numpy as np
# import re
# from scipy.ndimage import binary_dilation


# # Class Mapping and Color Palette
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

# class_names = list(CLASS_MAPPING.keys())
# class_colors = [CLASS_COLOR_PALETTE[name][0] for name in class_names]
# cmap = ListedColormap(class_colors)
# bounds = [v - 0.5 for v in range(-1, len(class_colors))]
# norm = BoundaryNorm(bounds, cmap.N)


# # Session State Initialization
# if "show_instructions" not in st.session_state:
#     st.session_state.show_instructions = False


# # Utility Functions
# def extract_year(name):
#     match = re.search(r"(19|20)\d{2}", name)
#     return match.group(0) if match else "Unknown"

# def get_class_mask(file, class_name):
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
#     legend_elements = [
#         mpatches.Patch(color=color, label=label)
#         for label, (color, _) in CLASS_COLOR_PALETTE.items()
#     ]
#     fig, ax = plt.subplots(figsize=(2.5, 1.5))
#     ax.axis('off')
#     ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=8)
#     st.pyplot(fig)

# def display_results(metric_name, debug=False):
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
#     rows = []
#     for file in uploaded_files:
#         year = extract_year(file.name)
#         area = calculate_total_valid_area(file, debug)
#         rows.append({"Year": year, "Total Valid Polygon Area (km²)": round(area, 4)})
#     df = pd.DataFrame(sorted(rows, key=lambda x: x["Year"]))
#     st.markdown("### Total Valid Polygon Area (Clipped Extent)")
#     st.dataframe(df, use_container_width=True)


# # Streamlit Page Setup
# st.set_page_config(page_title="Land Classification", layout="wide")

# if st.session_state.show_instructions:
#     st.title("Export & Usage Instructions")
#     st.markdown("""
#     ### How to Use This App  
#     1. **Upload Files** – On the left sidebar, click the **Browse** button under *Upload Classified .tif Files* and select the classified raster files you want to analyze.  
#     2. **Wait for Loading** – Allow all selected images to load completely. Loading time may vary depending on file size and the area of the city.  
#     3. **Calculate Areas** – Once images are loaded, click on the desired **Area Calculation** button (e.g., *Green Space*, *Built Space*, etc.) in the sidebar.  
#     4. **Calculate Indexes** – Similarly, click on any **Index Calculation** button (*GIP*, *GBP*, *GSI*) to compute the selected index.  
#     5. **Note** – Area or index calculations can only be performed once images have fully loaded.  

#     ---

#     ### Google Earth Engine Export Code  
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
#     ✔️ **Check the *Tasks* tab in Earth Engine to download the export.**  
#     ✔️ **Always clip to your polygon for accurate area results.**
#     """)
#     st.button("⬅ Back", on_click=lambda: st.session_state.update({"show_instructions": False}))
#     st.stop()

# # Application Main Interface
# st.title("Land Classification Area Calculator")

# # Sidebar
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
# index_labels = {
#     "GIP": "GIP – Green vs. Impervious Surface Proportion",
#     "GBP": "GBP – Green vs. Built Spaces Proportion",
#     "GSI": "GSI – Green Index"
# }
# for index_key, index_label in index_labels.items():
#     if st.sidebar.button(index_label):
#         button_pressed = index_key

# st.sidebar.subheader(" Shapefile Extent")
# if st.sidebar.button("Total Valid Area"):
#     button_pressed = "Total Area"

# # Main Display
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





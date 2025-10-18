import numpy as np
import pandas as pd
from pathlib import Path
import shapely
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid conflicts with tkinter
import matplotlib.pyplot as plt

def analysis(directory):
    # Function description: Analysis of raw data
    # Input: directory of raw data
    # Output: Dataframe with statistical data, GeoDataFrame with routes as LineString and modes, list of figures containing plots, list of axes containing plots, folium map plotting the routes
    # Plots within axes show the data as seen in "y_label_names", one plot for each element

    routes = []
    figs = []
    axes = []
    y_axis_values = ["number_tracks", "distance_sum", "duration_sum", "distance_avg", "duration_avg", "speed_track_avg",
                     "speed_avg", "acc_avg", "speed_max", "acc_max"]
    y_label_names = ["Number of tracks", "Total distance [km]", "Total duration [h]", "Average distance [km]",
                     "Average duration [h]",
                     "Average speed per track [km/h]", "Average speed [km/h]", "Average acceleration [m/s²]",
                     "Maximum speed [km/h]",
                     "Maximum acceleration [m/s²]"]

    ### Dataframe für jeden Track mit wichtigsten Kennwerten aus Data + Verkehrsmittel
    main_data_tracks = pd.DataFrame(columns=["mode", "distance", "duration", "speed"])
    cont_data_tracks = pd.DataFrame(columns=["mode", "speed", "acceleration"])

    ### Data import
    ordner = Path(directory)

    for file in ordner.rglob("*.pkl"):
        vehicle_id, track_id, modality, modality_precission, data = pd.read_pickle(file)

        print("Processing file:", file.name)

        if not modality:
            modality = "car"

        ## Geopandas
        data_geo = data.loc[:, ["longitude", "latitude"]]
        data_geo = data_geo.dropna(subset=["longitude", "latitude"])

        # Debug: print sample coordinates to verify order
        if len(data_geo) > 0:
            sample_lon = data_geo.iloc[0]["longitude"]
            sample_lat = data_geo.iloc[0]["latitude"]
            print(f"\nDEBUG {file.name}:")
            print(f"  Raw from dataframe - longitude: {sample_lon}, latitude: {sample_lat}")
            print(f"  Check: |lat|={abs(sample_lat)}, |lon|={abs(sample_lon)}")

        # Linestring - swap if needed (Munich/London are ~48-51° lat, ~11° or -0.1° lon)
        # If the "longitude" column has values > 20, it's actually latitude (columns are mislabeled)
        if len(data_geo) > 0 and abs(data_geo.iloc[0]["longitude"]) > 20:
            print(f"  -> SWAPPING: longitude column contains latitude values")
            points = gpd.points_from_xy(data_geo.latitude, data_geo.longitude)
        else:
            print(f"  -> NOT swapping: columns appear correct")
            points = gpd.points_from_xy(data_geo.longitude, data_geo.latitude)

        # Print first point after creation
        if len(points) > 0:
            first_point = points[0]
            print(f"  Result point: x={first_point.x}, y={first_point.y}")

        route = shapely.LineString(points.tolist())
        routes.append({"mode": modality, "geometry": route})

        ## Kennwerte für Data berechnen und in Dataframe speichern
        # Total distance
        tot_dis = data["cal_distance_covered"].iloc[-1]

        # Total duration
        tot_duration = data.loc[data.index[-1], "time"] - data.loc[data.index[0], "time"]

        speed_avg = data["speed"].mean()

        # In erstes Dataframe schreiben
        main_data_tracks.loc[len(main_data_tracks)] = [modality, tot_dis, tot_duration, speed_avg]

        # Zweites Dataframe für kontinuierliche Werte
        # Outlier detection
        data.loc[data["speed"] > 100, "speed"] = np.nan

        temp = pd.DataFrame({
            "mode": modality,
            "speed": data["speed"].rolling(window=100, min_periods=1).mean(),
            "acceleration": (data["speed"].diff() / data["time"].diff().dt.total_seconds()) / 3.6
        })

        cont_data_tracks = pd.concat([cont_data_tracks, temp])

    main_agg = main_data_tracks.groupby("mode").agg(
        number_tracks=("mode", "size"),
        distance_sum=("distance", "sum"),
        duration_sum=("duration", "sum"),
        distance_avg=("distance", "mean"),
        duration_avg=("duration", "mean"),
        speed_track_avg=("speed", "mean")
    ).reset_index()

    cont_agg = cont_data_tracks.groupby("mode").agg(
        speed_avg=("speed", "mean"),
        acc_avg=("acceleration", lambda x: x.abs().mean()),
        speed_max=("speed", "max"),
        acc_max=("acceleration", "max")
    ).reset_index()

    # Beide zusammenführen
    data_out = pd.merge(main_agg, cont_agg, on="mode", how="outer")
    gdf = gpd.GeoDataFrame(routes, geometry="geometry", crs="EPSG:4326")

    # Plotting
    for i, val in enumerate(y_axis_values):
        fig, ax = plt.subplots()
        if "duration" in val:
            ydata = data_out[val].dt.total_seconds() / 3600
            data_out.assign(temp=ydata).plot.bar(x="mode", y="temp", ax=ax)
        else:
            data_out.plot.bar(x="mode", y=val, ax=ax)
        ax.set_xlabel("Mode")
        ax.set_ylabel(y_label_names[i])
        ax.set_title(f"{y_label_names[i]} per mode")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend().set_visible(False)
        figs.append(fig)
        axes.append(ax)

    print("basic analysis done")

    print("creating map...")
    # Create interactive map with folium
    import folium

    # Debug: print unique modes in the data
    unique_modes = gdf['mode'].unique()
    print(f"Unique modes in data: {unique_modes}")

    # Define colors for each modality
    color_map = {'car': 'red', 'bike': 'blue', 'walk': 'green', 'bus': 'orange', 'train': 'purple'}

    # Get center point for map
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    print(f"\nDEBUG - Map bounds: {bounds}")
    print(f"DEBUG - Center: lat={center_lat}, lon={center_lon}")
    print(f"DEBUG - First geometry sample coords: {list(gdf.iloc[0]['geometry'].coords)[:3]}")

    # Create folium map - folium expects [latitude, longitude]
    geo_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # Add each track to the map with color based on mode
    for idx, row in gdf.iterrows():
        color = color_map.get(row['mode'], 'gray')
        # Extract coordinates from LineString geometry - they're in (lon, lat) order
        coords = [(lat, lon) for lon, lat in row['geometry'].coords]
        if idx == 0:
            print(f"DEBUG - First 3 coords after swap: {coords[:3]}")
        folium.PolyLine(
            coords,
            color=color,
            weight=3,
            opacity=0.8,
            tooltip=row['mode']
        ).add_to(geo_map)

    # Add legend - only show modes that actually exist in the data
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 150px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <p style="margin:0; font-weight:bold;">Transportation Mode</p>
    '''
    # Add legend entries for each unique mode in the data
    for mode in unique_modes:
        color = color_map.get(mode, 'gray')
        legend_html += f'<p style="margin:5px 0;"><span style="color:{color};">■</span> {mode}</p>'
    legend_html += '</div>'
    geo_map.get_root().html.add_child(folium.Element(legend_html))

    print("map created")
    
    return data_out, gdf, figs, axes, geo_map
import numpy as np
import pandas as pd
from pathlib import Path
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import mapclassify


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

        if not modality:
            modality = "car"

        ## Geopandas
        data_geo = data.loc[:, ["longitude", "latitude"]]
        data_geo = data_geo.dropna(subset=["longitude", "latitude"])
        # Linestring
        points = gpd.points_from_xy(data_geo.longitude, data_geo.latitude)
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

    geo_map = gdf.explore("mode")

    return data_out, gdf, figs, axes, geo_map
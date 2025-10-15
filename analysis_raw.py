import numpy as np
import pandas as pd
from pathlib import Path
import shapely
import geopandas as gpd


def analysis(directory):

    routes = []

    ### Dataframe für jeden Track mit wichtigsten Kennwerten aus Data + Verkehrsmittel
    main_data_tracks = pd.DataFrame(columns=["mode", "distance", "duration"])
    cont_data_tracks = pd.DataFrame(columns=["mode", "speed", "acceleration"])

    ### Data import
    ordner = Path(directory)

    for file in ordner.rglob("*.pkl"):
        vehicle_id, track_id, modality, modality_precission, data = pd.read_pickle(file)

        if not modality:
            modality = "unknown"

        ## Geopandas
        data_geo = data.loc[:, ["longitude", "latitude"]]
        data_geo = data_geo.dropna(subset=["longitude", "latitude"])
        # Linestring
        points = gpd.points_from_xy(data_geo.longitude, data_geo.latitude)
        route = shapely.LineString(points.tolist())
        routes.append({"mode": modality, "geometry": route})

        ## Kennwerte für Data berechnen und in Dataframe speichern
        # Total distance
        tot_dis = data["cal_distance_covered"].iloc[-1]  # Einheit checken

        # Total duration
        tot_duration = data.loc[data.index[-1], "time"] - data.loc[data.index[0], "time"]

        # In erstes Dataframe schreiben
        main_data_tracks.loc[len(main_data_tracks)] = [modality, tot_dis, tot_duration]

        # Zweites Dataframe für kontinuierliche Werte
        # Outlier detection
        data.loc[data["speed"] > 100, "speed"] = np.nan

        temp = pd.DataFrame({
            "mode": modality,
            "speed": data["speed"].rolling(window=100, min_periods=1).mean(),
            "acceleration": data["speed"].diff() / data["time"].diff().dt.total_seconds()
        })

        cont_data_tracks = pd.concat([cont_data_tracks, temp])

    main_agg = main_data_tracks.groupby("mode").agg(
        number_tracks=("mode", "size"),
        distance_sum=("distance", "sum"),
        duration_sum=("duration", "sum"),
        distance_avg=("distance", "mean"),
        duration_avg=("duration", "mean")
    ).reset_index()

    cont_agg = cont_data_tracks.groupby("mode").agg(
        speed_avg=("speed", "mean"),
        acc_avg=("acceleration", "mean"),
        speed_max=("speed", "max"),
        acc_max=("acceleration", "max")
    ).reset_index()

    # Beide zusammenführen
    data_out = pd.merge(main_agg, cont_agg, on="mode", how="outer")
    gdf = gpd.GeoDataFrame(routes, geometry="geometry", crs="EPSG:4326")
    return data_out, gdf
from dashboard import create_dashboard
from analysis_raw import analysis

print("test1")

# Run analysis and get results
data_out, gdf, figs, axes, geo_map = analysis("./data/raw")

# Launch the dashboard with analysis results
create_dashboard(data_out, figs, geo_map)
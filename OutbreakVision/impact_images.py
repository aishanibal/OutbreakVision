import os
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# === Load World Map ===
shapefile_path = "data/countries_py/ne_110m_admin_0_countries.shp"  # Ensure correct path
world = gpd.read_file(shapefile_path)

# Ensure proper country column name
country_column = "ADMIN"  # This is the correct column name for country names

# Handle multipolygons (fix missing borders)
world = world.explode(index_parts=False)

# === Load Impact Score Data ===
impact_data = pd.read_csv("impact_overtime.csv")

# Handle potential mismatches in country names
country_mapping = {
    "United States": "United States of America",
    "Russia": "Russian Federation",
    "Czech Republic": "Czechia",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
    "Republic of Korea": "South Korea",
    "Iran": "Iran, Islamic Republic of",
    "Vietnam": "Viet Nam",
    "Syria": "Syrian Arab Republic",
    "Bolivia": "Bolivia, Plurinational State of",
    "Venezuela": "Venezuela, Bolivarian Republic of"
}

impact_data["Country"] = impact_data["Country"].replace(country_mapping)

# Rename existing timestamp columns
old_timestamps = impact_data.columns[1:6]  # Assuming the first column is "Country" and next 5 are timestamps
new_timestamps = [f"Year {i}" for i in range(5)]
impact_data.rename(columns=dict(zip(old_timestamps, new_timestamps)), inplace=True)

# Verify renaming
print("Updated column names:", impact_data.columns)

# Normalize color mapping (set fixed range for better visualization)
vmin, vmax = 35, 70  # Adjust based on expected impact score range

# Create a folder for saving images
output_folder = "impact_timestamp_images"
os.makedirs(output_folder, exist_ok=True)

# Plot maps for different timestamps
timestamps = new_timestamps  # Use the renamed timestamps

for timestamp in timestamps:
    # Merge impact data with world map
    merged = world.merge(impact_data, left_on=country_column, right_on="Country", how="left")

    if timestamp not in merged.columns:
        print(f"Error: '{timestamp}' column not found in merged DataFrame.")
        continue  # Skip if the column is missing
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    merged.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Ensure borders are visible
    merged.plot(column=timestamp, cmap="Blues", linewidth=0.5, edgecolor="black",
                legend=True, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgrey", "label": "No Data"})
    
    # Title and labels
    ax.set_title(f"Impact Score on {timestamp}", fontsize=14)
    ax.set_axis_off()  # Hide axis
    
    # Save figure in the new folder
    image_path = os.path.join(output_folder, f"impact_map_{timestamp}.png")
    plt.savefig(image_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {image_path}")

print("\nAll maps saved successfully.")

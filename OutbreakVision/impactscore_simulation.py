import pandas as pd
import geopandas as gpd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# === Load World Map ===
shapefile_path = "data/countries_py/ne_110m_admin_0_countries.shp"  # Adjust path if necessary
world = gpd.read_file(shapefile_path)

# Correct column name for country names in the shapefile
country_column = "ADMIN"

# === Load Impact Score Data ===
impact_data = pd.read_csv("impact_overtime.csv")

# Rename existing timestamp columns to "Year 0" to "Year 4"
old_timestamps = impact_data.columns[1:6]  # Assuming the first column is "Country" and next 5 are timestamps
new_timestamps = [f"Year {i}" for i in range(5)]
impact_data.rename(columns=dict(zip(old_timestamps, new_timestamps)), inplace=True)

# Print to verify renaming
print("Updated column names:", impact_data.columns)

# Normalize color mapping (adjusted for visibility)
vmin, vmax = impact_data[new_timestamps].min().min(), impact_data[new_timestamps].max().max()

# Define timestamps and number of transition frames
timestamps = new_timestamps  # Use renamed year-based timestamps
num_transition_frames = 30  # Frames between each timestamp
num_start_frames = 60  # Frames to gradually transition from vmin to Year 0

# Merge impact data with the world map
merged = world.merge(impact_data, left_on=country_column, right_on="Country", how="left")

# Video settings
video_filename = "impact_simulation.mp4"
frame_width, frame_height = 1200, 600  # Set desired resolution
fps = 10  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# Generate frames for the video
fig, ax = plt.subplots(figsize=(12, 6))

# === Step 1: Initial Phase (All Countries Start at the Same Color) ===
start_scores = pd.Series(vmin, index=merged.index)  # Start all countries at the lowest impact score
end_scores = merged[timestamps[0]].fillna(vmin)  # Transition to first timestamp (Year 0)

for j in range(num_start_frames):
    alpha = j / num_start_frames  # Interpolation factor (0 to 1)
    interpolated_scores = (1 - alpha) * start_scores + alpha * end_scores

    ax.clear()
    merged.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Ensure borders are visible
    merged.plot(column=interpolated_scores, cmap="plasma", linewidth=0.5, edgecolor="black",
                legend=False, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgrey", "label": "No Data"})
    
    ax.set_title(f"Impact Score Transition (Starting at Same Color → {timestamps[0]})", fontsize=14)
    ax.set_axis_off()

    # Save frame
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = cv2.resize(img, (frame_width, frame_height))
    out.write(img)

# === Step 2: Transition Through Remaining Years ===
for i in range(len(timestamps) - 1):
    start_time = timestamps[i]
    end_time = timestamps[i + 1]

    start_scores = merged[start_time].fillna(vmin)
    end_scores = merged[end_time].fillna(vmin)

    for j in range(num_transition_frames):
        alpha = j / num_transition_frames
        interpolated_scores = (1 - alpha) * start_scores + alpha * end_scores

        ax.clear()
        merged.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Ensure borders are visible
        merged.plot(column=interpolated_scores, cmap="plasma", linewidth=0.5, edgecolor="black",
                    legend=False, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgrey", "label": "No Data"})
        
        ax.set_title(f"Impact Score Transition ({start_time} → {end_time})", fontsize=14)
        ax.set_axis_off()

        # Save frame
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (frame_width, frame_height))
        out.write(img)

# Release the video writer
out.release()
plt.close(fig)

print(f"\nSimulation video saved as '{video_filename}' successfully.")

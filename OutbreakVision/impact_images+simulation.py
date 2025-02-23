import os
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

# Ensure we are renaming only the impact score columns
original_timestamps = impact_data.columns[1:6]  # Assuming first column is "Country", next 5 are timestamps
new_timestamps = [f"Year {i}" for i in range(5)]
impact_data.rename(columns=dict(zip(original_timestamps, new_timestamps)), inplace=True)

# Normalize color mapping (adjusted for visibility)
vmin, vmax = impact_data[new_timestamps].min().min(), impact_data[new_timestamps].max().max()

# Merge impact data with the world map
merged = world.merge(impact_data, left_on=country_column, right_on="Country", how="left")

# === Video Settings ===
video_filename = "impact_simulation.mp4"
frame_width, frame_height = 1200, 600  # Set desired resolution
fps = 10  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

# === Create Folder for Snapshots ===
snapshot_folder = "ImpactScore_snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

# === Generate Frames for Video ===
fig, ax = plt.subplots(figsize=(12, 6))
num_transition_frames = 30  # Frames between each timestamp
num_start_frames = 60  # Frames to transition from vmin to Year 0

# === Step 1: Initial Phase (All Countries Start at the Same Color) ===
start_scores = pd.Series(vmin, index=merged.index)  # Start all countries at the lowest impact score
end_scores = merged["Year 0"].fillna(vmin)  # Transition to first timestamp (Year 0)

for j in range(num_start_frames):
    alpha = j / num_start_frames  # Interpolation factor (0 to 1)
    interpolated_scores = (1 - alpha) * start_scores + alpha * end_scores

    ax.clear()
    merged.boundary.plot(ax=ax, linewidth=0.5, color="black")  # Ensure borders are visible
    merged.plot(column=interpolated_scores, cmap="plasma", linewidth=0.5, edgecolor="black",
                legend=False, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgrey", "label": "No Data"})

    ax.set_title(f"Impact Score Transition", fontsize=14)
    ax.set_axis_off()

    # Save frame to video
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img = cv2.resize(img, (frame_width, frame_height))
    out.write(img)

# === Save First Snapshot for "Year 0" ===
image_path = os.path.join(snapshot_folder, f"ImpactScore_Year_0.png")
plt.savefig(image_path, bbox_inches="tight", dpi=300)
print(f"Saved snapshot: {image_path}")

# === Step 2: Transition Through Remaining Years (Video + Snapshots) ===
for i in range(len(new_timestamps) - 1):
    start_time = new_timestamps[i]
    end_time = new_timestamps[i + 1]

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

        # Save frame to video
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (frame_width, frame_height))
        out.write(img)

    # === Save Snapshot for Current Timestamp ===
    image_path = os.path.join(snapshot_folder, f"ImpactScore_{end_time}.png")
    plt.savefig(image_path, bbox_inches="tight", dpi=300)
    print(f"Saved snapshot: {image_path}")

# === Release Video Writer ===
out.release()
plt.close(fig)

print(f"\nSimulation video saved as '{video_filename}' successfully.")
print("All impact score snapshots have been saved successfully.")

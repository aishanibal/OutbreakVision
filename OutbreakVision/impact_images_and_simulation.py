def generate_impact_video():
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    # Load World Map
    shapefile_path = "data/countries_py/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)

    # Load Impact Score Data
    impact_data = pd.read_csv("impact_overtime.csv")

    # Adjust timestamps
    original_timestamps = impact_data.columns[1:6]
    new_timestamps = [f"Year {i}" for i in range(5)]
    impact_data.rename(columns=dict(zip(original_timestamps, new_timestamps)), inplace=True)

    # Normalize color mapping
    vmin, vmax = impact_data[new_timestamps].min().min(), impact_data[new_timestamps].max().max()

    # Merge impact data with world map
    merged = world.merge(impact_data, left_on="ADMIN", right_on="Country", how="left")

    # Generate Video
    video_filename = "static/impact_simulation.mp4"
    frame_width, frame_height = 1200, 600
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    fig, ax = plt.subplots(figsize=(12, 6))
    num_transition_frames = 30

    for i in range(len(new_timestamps) - 1):
        start_time, end_time = new_timestamps[i], new_timestamps[i + 1]
        start_scores = merged[start_time].fillna(vmin)
        end_scores = merged[end_time].fillna(vmin)

        for j in range(num_transition_frames):
            alpha = j / num_transition_frames
            interpolated_scores = (1 - alpha) * start_scores + alpha * end_scores

            ax.clear()
            merged.boundary.plot(ax=ax, linewidth=0.5, color="black")
            merged.plot(column=interpolated_scores, cmap="plasma", linewidth=0.5, edgecolor="black",
                        legend=False, vmin=vmin, vmax=vmax, ax=ax, missing_kwds={"color": "lightgrey", "label": "No Data"})

            ax.set_title(f"Impact Score Transition ({start_time} → {end_time})", fontsize=14)
            ax.set_axis_off()

            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img = cv2.resize(img, (frame_width, frame_height))
            out.write(img)

    out.release()
    plt.close(fig)

    return video_filename

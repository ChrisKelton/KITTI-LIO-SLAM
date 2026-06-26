import collections
import os
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ColumnNames = [
    "frame",
    "track_id",
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "height",
    "width",
    "length",
    "x",
    "y",
    "z",
    "rotation_y",
]

TypeConv = [
    int,  # frame
    int,  # track_id
    str,  # type
    int,  # truncated
    int,  # occluded
    float,  # alpha
    float,  # bbox_left
    float,  # bbox_top
    float,  # bbox_right
    float,  # bbox_bottom
    float,  # height
    float,  # width
    float,  # length
    float,  # x
    float,  # y
    float,  # z
    float,  # rotation_y
]

DtypeDict = {name: conv for name, conv in zip(ColumnNames, TypeConv) if conv != str}
ConvDict = {name: conv for name, conv in zip(ColumnNames, TypeConv) if conv == str}

def extract_tracking_labels(sequence: Path, labels_to_extract: Optional[str | list[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(
        str(sequence),
        delimiter=" ",
        header=None,
        names=ColumnNames,
        dtype=DtypeDict,
        converters=ConvDict,
    )
    df["type"]= [str_.lower() for str_ in df["type"]]

    if labels_to_extract is not None:
        if not isinstance(labels_to_extract, list):
            labels_to_extract = [labels_to_extract]
        df = df[df["type"].isin(labels_to_extract)].reset_index(drop=True)

    return df


# def get_3d_box_corners(row):
#     """Calculates the 8 corners of a 3D bounding box from a DataFrame row."""
#     h, w, l = row["height"], row["width"], row["length"]
#     tx, ty, tz = row["x"], row["y"], row["z"]
#     yaw = row["rotation_y"]
#
#     # KITTI object frame corners relative to bottom-center
#     x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
#     y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
#     z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
#     corners = np.vstack([x_corners, y_corners, z_corners])
#
#     # Rotation matrix around Y-axis
#     R = np.array(
#         [
#             [np.cos(yaw), 0, np.sin(yaw)],
#             [0, 1, 0],
#             [-np.sin(yaw), 0, np.cos(yaw)],
#         ]
#     )
#
#     corners_3d = np.dot(R, corners)
#     corners_3d[0, :] += tx
#     corners_3d[1, :] += ty
#     corners_3d[2, :] += tz
#
#     return corners_3d.T
#
#
# def animate_kitti_3d(df, interval=200, output_path=None, swap_yz: bool = True):
#     """Creates an interactive 3D plot of KITTI frames with key controls.
#
#     Features:
#       - Color-coded unique tracks.
#       - Historical trajectory paths trailing each moving object.
#
#     Controls:
#       - Spacebar: Pause / Play animation
#       - Right Arrow: Step 1 frame forward (only when paused)
#       - Left Arrow: Step 1 frame backward (only when paused)
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#
#     frames = sorted(df["frame"].unique())
#
#     # Establish stable, fixed axis boundaries
#     ax.set_xlim(df["x"].min() - 2, df["x"].max() + 2)
#     if swap_yz:
#         ax.set_ylim(df["z"].min() - 2, df["z"].max() + 2)
#         ax.set_zlim(df["y"].min() - 2, df["y"].max() + 2)
#     else:
#         ax.set_ylim(df["y"].min() - 2, df["y"].max() + 2)
#         ax.set_zlim(df["z"].min() - 2, df["z"].max() + 2)
#     ax.invert_yaxis()
#
#     ax.set_xlabel("X (Left/Right)")
#     if swap_yz:
#         ax.set_ylabel("Z (Depth)")
#         ax.set_zlabel("Y (Up/Down)")
#     else:
#         ax.set_ylabel("Y (Up/Down)")
#         ax.set_zlabel("Z (Depth)")
#
#     # Color map for track tracking (supports up to 20 unique colors cyclically)
#     cmap = plt.cm.get_cmap("tab20")
#
#     edges = [
#         [1, 2], [2, 3], [3, 0],  # Bottom face
#         [5, 6], [6, 7], [7, 4],  # Top face
#         [1, 5], [2, 6], [3, 7],  # Vertical pillars
#     ]
#
#     current_artists = []
#     anim_running = True
#     current_frame_idx = 0
#
#     # Build historical trajectory storage per Track ID across all frames
#     # Structure: {track_id: [(x1, y1, z1), (x2, y2, z2), ...]}
#     history_paths = collections.defaultdict(list)
#
#     def update(frame_idx):
#         nonlocal current_frame_idx
#         current_frame_idx = frame_idx
#
#         # Clear existing geometry, trajectory lines, and text elements
#         while current_artists:
#             artist = current_artists.pop()
#             artist.remove()
#
#         frame_num = frames[frame_idx]
#         status = (
#             ""
#             if anim_running
#             else " [PAUSED — Use Left/Right Arrow keys to step]"
#         )
#         ax.set_title(
#             f"KITTI 3D Bounding Boxes — Frame: {frame_num}{status}", fontsize=12
#         )
#
#         frame_df = df[df["frame"] == frame_num]
#
#         # First, dynamically update trajectories up to the current frame index
#         # (This logic rebuilds correctly even if the user jumps or steps backward)
#         history_paths.clear()
#         past_frames_df = df[df["frame"] <= frame_num].sort_values("frame")
#         for _, past_row in past_frames_df.iterrows():
#             t_id = int(past_row["track_id"])
#             if swap_yz:
#                 history_paths[t_id].append((past_row["x"], past_row["z"], past_row["y"]))
#             else:
#                 history_paths[t_id].append((past_row["x"], past_row["y"], past_row["z"]))
#
#         # Draw the active objects in the current frame
#         for _, row in frame_df.iterrows():
#             track_id = int(row["track_id"])
#
#             # Map the track ID to a distinct cyclic color index
#             color = cmap(track_id % 20)
#
#             # Draw 3D bounding box edges
#             corners = get_3d_box_corners(row)
#             for edge in edges:
#                 if swap_yz:
#                     (line,) = ax.plot3D(
#                         corners[edge, 0],
#                         corners[edge, 2],
#                         corners[edge, 1],
#                         color=color,
#                         linewidth=2,
#                         alpha=0.9,
#                     )
#                 else:
#                     (line,) = ax.plot3D(
#                         corners[edge, 0],
#                         corners[edge, 1],
#                         corners[edge, 2],
#                         color=color,
#                         linewidth=2,
#                         alpha=0.9,
#                     )
#                 current_artists.append(line)
#
#             # Draw current center point
#             if swap_yz:
#                 scatter = ax.scatter(row["x"], row["z"], row["y"], color=color, s=25)
#             else:
#                 scatter = ax.scatter(row["x"], row["y"], row["z"], color=color, s=25)
#             current_artists.append(scatter)
#
#             # Draw the historical trajectory trail line
#             path_pts = np.array(history_paths[track_id])
#             if len(path_pts) > 1:
#                 (trail,) = ax.plot3D(
#                     path_pts[:, 0],
#                     path_pts[:, 1],
#                     path_pts[:, 2],
#                     color=color,
#                     linestyle="--",
#                     linewidth=1.5,
#                     alpha=0.6,
#                 )
#                 current_artists.append(trail)
#
#             # Add Track ID text overlay above the box
#             if swap_yz:
#                 text_element = ax.text(
#                     row["x"],
#                     row["z"],
#                     row["y"] - row["height"],
#                     s=f"ID: {track_id}",
#                     color="black",
#                     fontsize=8,
#                     fontweight="bold",
#                     bbox=dict(
#                         facecolor="white", alpha=0.7, edgecolor="none", pad=1
#                     ),
#                 )
#             else:
#                 text_element = ax.text(
#                     row["x"],
#                     row["y"] - row["height"],
#                     row["z"],
#                     s=f"ID: {track_id}",
#                     color="black",
#                     fontsize=8,
#                     fontweight="bold",
#                     bbox=dict(
#                         facecolor="white", alpha=0.7, edgecolor="none", pad=1
#                     ),
#                 )
#             current_artists.append(text_element)
#
#         return current_artists
#
#     # Generator function to handle frame routing dynamically
#     def frame_generator():
#         i = 0
#         while i < len(frames):
#             yield i
#             if anim_running:
#                 i = (i + 1) % len(frames)
#             else:
#                 i = current_frame_idx
#
#     # CRITICAL FIX: If saving, do not use the custom infinite frame_generator.
#     # Use a concrete range loop instead so the writer can count frames safely.
#     if output_path:
#         frames_source = range(len(frames))
#         repeat_option = False  # Turn off repeating loops when extracting file
#     else:
#         frames_source = frame_generator
#         repeat_option = True
#
#     ani = animation.FuncAnimation(
#         fig,
#         update,
#         frames=frames_source,
#         interval=interval,
#         cache_frame_data=False,
#         blit=False,
#         repeat=repeat_option,
#     )
#
#     # Keyboard input event handling logic
#     def on_key(event):
#         nonlocal anim_running, current_frame_idx
#         if event.key == " ":
#             anim_running = not anim_running
#             if anim_running:
#                 ani.resume()
#             else:
#                 ani.pause()
#             update(current_frame_idx)
#             fig.canvas.draw_idle()
#
#         elif event.key == "right":
#             if not anim_running:
#                 next_idx = (current_frame_idx + 1) % len(frames)
#                 update(next_idx)
#                 fig.canvas.draw_idle()
#
#         elif event.key == "left":
#             if not anim_running:
#                 prev_idx = (current_frame_idx - 1) % len(frames)
#                 update(prev_idx)
#                 fig.canvas.draw_idle()
#
#     fig.canvas.mpl_connect("key_press_event", on_key)
#
#     # Export to file cleanly before loading interactive plt.show() loop
#     if output_path:
#         print(f"Saving animation to {output_path}...")
#         ani.save(
#             output_path,
#             writer=animation.PillowWriter(fps=int(1000 / interval)),
#         )
#         print("Saving complete!")
#
#     plt.show()
#     return ani


def get_3d_box_corners_in_image(row):
    """Calculates the 8 corners of a 3D bounding box from a DataFrame row."""
    h, w, l = row["height"], row["width"], row["length"]
    tx, ty, tz = row["x"], row["y"], row["z"]
    yaw = row["rotation_y"]

    # KITTI object frame corners relative to bottom-center
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.vstack([x_corners, y_corners, z_corners])

    # Rotation matrix around Y-axis
    R = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ]
    )

    corners_3d = np.dot(R, corners)
    corners_3d[0, :] += tx
    corners_3d[1, :] += ty
    corners_3d[2, :] += tz

    return corners_3d.T


def get_3d_box_corners_in_velo(row):
    h, w, l = row["height"], row["width"], row["length"]
    vx, vy, vz = row["x"], row["y"], row["z"]
    alpha_velo = row["rotation_y"]

    # In Velodyne local frame: X=width, Y=length, Z=height
    # We assume the local origin is at the box's bottom-center (matching KITTI conventions)
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2]
    y_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
    z_corners = [0, 0, 0, 0, h, h, h, h]  # 0 is bottom face, h is top face

    local_corners = np.vstack([x_corners, y_corners, z_corners])

    # Rotation matrix around Velodyne Z-axis (yaw)
    # Note: Ensure alpha_velo is calculated properly for the Velodyne frame orientation
    cos_a = np.cos(alpha_velo)
    sin_a = np.sin(alpha_velo)
    R_velo = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    # Rotate local corners and translate to the Velodyne center
    velo_corners_3d = np.dot(R_velo, local_corners)
    velo_corners_3d[0, :] += vx
    velo_corners_3d[1, :] += vy
    velo_corners_3d[2, :] += vz

    return velo_corners_3d.T


def animate_kitti_3d(
    df,
    interval=200,
    output_path=None,
    image_dir=None,
):
    """Creates an interactive visual player of KITTI tracking labels.

    If image_dir is provided, it overlays 2D bounding boxes and 2D paths onto
    the sequence frames. Otherwise, it projects a full 3D spatial scene.

    Controls:
      - Spacebar: Pause / Play animation
      - Right Arrow: Step 1 frame forward (only when paused)
      - Left Arrow: Step 1 frame backward (only when paused)
    """
    use_image_corners: bool = True if image_dir is not None else False
    if use_image_corners:
        get_3d_box_corners = get_3d_box_corners_in_image
    else:
        get_3d_box_corners = get_3d_box_corners_in_velo

    fig = plt.figure(figsize=(12, 7))
    frames = sorted(df["frame"].unique())

    # Switch display profile layouts based on input arguments
    use_2d_overlay = image_dir is not None

    if use_2d_overlay:
        ax = fig.add_subplot(111)  # Standard 2D coordinate system
    else:
        ax = fig.add_subplot(111, projection="3d")  # Spatial 3D world
        ax.set_xlim(df["x"].min() - 2, df["x"].max() + 2)
        ax.set_ylim(df["y"].min() - 2, df["y"].max() + 2)
        ax.set_zlim(df["z"].min() - 2, df["z"].max() + 2)
        ax.invert_yaxis()
        ax.set_xlabel("X (Left/Right)")
        ax.set_ylabel("Y (Up/Down)")
        ax.set_zlabel("Z (Depth)")

    cmap = plt.colormaps.get_cmap("tab20")
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # Bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # Top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Vertical pillars
    ]

    current_artists = []
    anim_running = True
    current_frame_idx = 0
    history_paths = collections.defaultdict(list)

    def update(frame_idx):
        nonlocal current_frame_idx
        current_frame_idx = frame_idx

        # Flush all visual items from the active view frame container
        while current_artists:
            artist = current_artists.pop()
            artist.remove()

        frame_num = frames[frame_idx]
        status = (
            ""
            if anim_running
            else " [PAUSED — Use Left/Right Arrow keys to step]"
        )
        ax.set_title(
            f"KITTI Tracking Sequence — Frame: {frame_num}{status}", fontsize=12
        )

        frame_df = df[df["frame"] == frame_num]

        # ----------------- LAYER PROFILE A: 2D IMAGE OVERLAY -----------------
        if use_2d_overlay:
            # KITTI tracks sequence frames use a zero-padded 6-digit naming pattern
            img_path = os.path.join(image_dir, f"{int(frame_num):06d}.png")
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                im_art = ax.imshow(img)
                # Ensure the background frame is removed cleanly next cycle
                current_artists.append(im_art)

            # Rebuild historical 2D bounding box center trajectories
            history_paths.clear()
            past_df = df[df["frame"] <= frame_num].sort_values("frame")
            for _, past_row in past_df.iterrows():
                t_id = int(past_row["track_id"])
                # BBox center point coordinate mapping: (X_mid, Y_mid)
                cx = (past_row["bbox_left"] + past_row["bbox_right"]) / 2.0
                cy = (past_row["bbox_top"] + past_row["bbox_bottom"]) / 2.0
                history_paths[t_id].append((cx, cy))

            # Render 2D target labels onto the background layer
            for _, row in frame_df.iterrows():
                track_id = int(row["track_id"])
                color = cmap(track_id % 20)

                x1, y1 = row["bbox_left"], row["bbox_top"]
                w2d = row["bbox_right"] - x1
                h2d = row["bbox_bottom"] - y1

                # Draw 2D bounding box contour borders
                rect = patches.Rectangle(
                    (x1, y1),
                    w2d,
                    h2d,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(rect)
                current_artists.append(rect)

                # Draw historical 2D trail line paths
                path_pts = np.array(history_paths[track_id])
                if len(path_pts) > 1:
                    (trail,) = ax.plot(
                        path_pts[:, 0],
                        path_pts[:, 1],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    current_artists.append(trail)

                # Append textual metadata context tags above the 2D bounding boxes
                text_element = ax.text(
                    x1,
                    y1 - 5,
                    s=f"ID: {track_id} ({row['type']})",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor="none"),
                )
                current_artists.append(text_element)

        # ---------------- LAYER PROFILE B: ORIGINAL 3D WORLD -----------------
        else:
            history_paths.clear()
            past_df = df[df["frame"] <= frame_num].sort_values("frame")
            for _, past_row in past_df.iterrows():
                t_id = int(past_row["track_id"])
                history_paths[t_id].append(
                    (past_row["x"], past_row["y"], past_row["z"])
                )

            for _, row in frame_df.iterrows():
                track_id = int(row["track_id"])
                color = cmap(track_id % 20)

                corners = get_3d_box_corners(row)
                for edge in edges:
                    (line,) = ax.plot3D(
                        corners[edge, 0],
                        corners[edge, 1],
                        corners[edge, 2],
                        color=color,
                        linewidth=2,
                        alpha=0.9,
                    )
                    current_artists.append(line)

                scatter = ax.scatter(
                    row["x"], row["y"], row["z"], color=color, s=25
                )
                current_artists.append(scatter)

                path_pts = np.array(history_paths[track_id])
                if len(path_pts) > 1:
                    (trail,) = ax.plot3D(
                        path_pts[:, 0],
                        path_pts[:, 1],
                        path_pts[:, 2],
                        color=color,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.6,
                    )
                    current_artists.append(trail)

                text_element = ax.text(
                    row["x"],
                    row["y"] - row["height"],
                    row["z"],
                    s=f"ID: {track_id}",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white", alpha=0.7, edgecolor="none", pad=1
                    ),
                )
                current_artists.append(text_element)

        return current_artists

    # Configure active compilation tracking sources
    if output_path:
        frames_source = range(len(frames))
        repeat_option = False
    else:
        def frame_generator():
            i = 0
            while i < len(frames):
                yield i
                if anim_running:
                    i = (i + 1) % len(frames)
                else:
                    i = current_frame_idx

        frames_source = frame_generator
        repeat_option = True

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames_source,
        interval=interval,
        cache_frame_data=False,
        blit=False,
        repeat=repeat_option,
    )

    def on_key(event):
        nonlocal anim_running, current_frame_idx
        if event.key == " ":
            anim_running = not anim_running
            if anim_running:
                ani.resume()
            else:
                ani.pause()
            update(current_frame_idx)
            fig.canvas.draw_idle()
        elif event.key == "right" and not anim_running:
            update((current_frame_idx + 1) % len(frames))
            fig.canvas.draw_idle()
        elif event.key == "left" and not anim_running:
            update((current_frame_idx - 1) % len(frames))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    if output_path:
        print(f"Saving animation sequence to {output_path}...")
        ani.save(
            output_path,
            writer=animation.PillowWriter(fps=int(1000 / interval)),
        )
        print("Saving complete!")

    plt.show()
    return ani


def load_calib_txt_file(txt_file: Path) -> dict[str, np.ndarray]:
    with open(str(txt_file), 'r') as f:
        lines = [line.strip("\n").split(":") for line in f.readlines()]
    calibration_mats: dict[str, np.ndarray] = {}
    for line in lines:
        if len(line) == 2:
            key = line[0]
            mat_vals = [float(val) for val in line[1].split(" ") if val != ""]
        elif len(line) == 1:
            vals = line[0].split(" ")
            key = vals[0]
            mat_vals = [float(val) for val in vals[1:] if val != ""]
        else:
            raise NotImplementedError(f"Unexpected number of split entries in line. Got '{len(line)}' in '{line}'")
        mat = np.asarray(mat_vals)
        if len(mat_vals) == 12:
            mat = mat.reshape(3, 4)
        elif len(mat_vals) == 9:
            mat = mat.reshape(3, 3)
        else:
            raise NotImplementedError(f"Got unsupported number of values: '{len(mat_vals)}'")
        calibration_mats[key] = mat

    return calibration_mats


def main():
    tracking_data_dir: Path = Path("/home/ckelton/data/KITTI/tracking")
    labels_dir: Path = tracking_data_dir / "labels/training/label_02"
    images_dir: Path = tracking_data_dir / "images/training/image_02"
    calib_dir: Path = tracking_data_dir / "calib/training/calib"
    output_dir: Path = labels_dir / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    overwrite_3d_plots: bool = True
    overwrite_2d_plots: bool = False

    label_sequences: list[Path] = sorted(labels_dir.glob("*.txt"))

    for label_sequence in label_sequences:
        sequence = label_sequence.stem
        df = extract_tracking_labels(label_sequence, labels_to_extract=["car", "pedestrian"])

        calib_txt_path = calib_dir / f"{sequence}.txt"
        transformation_mat = None
        if calib_txt_path.exists():
            transformation_mats: dict[str, np.ndarray] = load_calib_txt_file(calib_txt_path)
            T = transformation_mats.get("Tr_velo_cam")
            if T is not None:
                R_rect = transformation_mats.get("R_rect")
                if R_rect is not None:
                    T = np.dot(R_rect, T)
                    R = T[:3, :3]
                    t = T[:3, 3]
                    R_inv = R.T
                    t_inv = R_inv @ -t
                    transformation_mat = np.column_stack([R_inv, t_inv])
            else:
                print(f"No 'Tr_velo_cam' key in transformation matrices loaded from '{calib_txt_path}'. Got '{list(transformation_mats.keys())}'")

        df_: pd.DataFrame = df.copy()
        if transformation_mat is not None:
            if transformation_mat.shape != (3, 4):
                raise RuntimeError(
                    f"Expected transformation_mat to be of shape (3, 4). Got '{transformation_mat.shape}'")
            tmp = (transformation_mat @ np.asarray([df_["x"], df_["y"], df_["z"], np.ones(len(df_))])).T
            df_["x"] = tmp[:, 0]
            df_["y"] = tmp[:, 1]
            df_["z"] = tmp[:, 2]

            def transform_to_velo_alpha(row):
                ry = row["rotation_y"]
                v_cam = np.array([np.sin(ry), 0.0, np.cos(ry), 0.0])
                v_velo = transformation_mat @ v_cam
                alpha_velo = np.arctan2(v_velo[0], v_velo[1])
                alpha_velo = (alpha_velo + np.pi) % (2 * np.pi) - np.pi
                return pd.Series(
                    {
                        "rotation_y": alpha_velo
                    }
                )

            df_["rotation_y"] = df_.apply(transform_to_velo_alpha, axis=1)

        output_3d_plot: Path = output_dir / f"{sequence}-3d.gif"
        if overwrite_3d_plots or not output_3d_plot.exists():
            animate_kitti_3d(
                df_,
                output_path=output_3d_plot,
            )
            plt.close()
        else:
            print("Skipping animating 3D plot.")

        output_2d_plot: Path = output_dir / f"{sequence}-2d.gif"
        if overwrite_2d_plots or not output_2d_plot.exists():
            image_dir = images_dir / sequence
            if image_dir.exists():
                animate_kitti_3d(df, output_path=output_2d_plot, image_dir=image_dir)
            plt.close()


if __name__ == '__main__':
    main()

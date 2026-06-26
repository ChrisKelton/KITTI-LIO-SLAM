from pathlib import Path
from typing import Optional

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


def convert_labels_from_cam_to_velo_coordinates(df: pd.DataFrame, cam_to_velo: np.ndarray) -> pd.DataFrame:
    tmp = (cam_to_velo @ np.asarray([df["x"], df["y"], df["z"], np.ones(len(df))])).T
    df["x"] = tmp[:, 0]
    df["y"] = tmp[:, 1]
    df["z"] = tmp[:, 2]


    def transform_to_velo_alpha(row):
        ry = row["rotation_y"]
        v_cam = np.array([np.sin(ry), 0.0, np.cos(ry), 0.0])
        v_velo = cam_to_velo @ v_cam
        alpha_velo = np.arctan2(v_velo[0], v_velo[1])
        alpha_velo = (alpha_velo + np.pi) % (2 * np.pi) - np.pi
        return pd.Series(
            {
                "rotation_y": alpha_velo
            }
        )


    df["rotation_y"] = df.apply(transform_to_velo_alpha, axis=1)

    return df

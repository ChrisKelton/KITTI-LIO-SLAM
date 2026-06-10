from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import Polygon, LineString
from tqdm import tqdm

DATA_PATH: Path = Path("/home/ckelton/data/KITTI/raw_data/2011_09_26_drive_0022")
CALIB_PATH = DATA_PATH / "calib"
SYNC_PATH = DATA_PATH / "sync"
TRACKLETS_PATH = DATA_PATH / "tracklets"


def save_polygons_func(poly: Polygon) -> Polygon:
    return Polygon(np.column_stack(poly.exterior.coords.xy).dot([[1, 0], [0, -1]]))


def save_lines_func(line: LineString) -> LineString:
    return LineString(np.column_stack(line.xy).dot([[1, 0], [0, -1]]))


def polygons_to_gpkg(polys: list[Polygon], outpath: Path, func: Optional[Callable] = None):
    if func is None:
        func = lambda x: x
    gdf = gpd.GeoDataFrame(
        {'name': [str(idx) for idx in range(len(polys))]},
        geometry=list(map(func, polys)),
        crs=None,
    )
    gdf.to_file(outpath, driver="GPKG")


def lines_to_gpkg(
    lines: list[LineString],
    outpath: Path,
    func: Optional[Callable] = None,
    crs: Optional[str] = None,
):
    if func is None:
        func = lambda x: x
    gdf = gpd.GeoDataFrame(
        {'name': [str(idx) for idx in range(len(lines))]},
        geometry=list(map(func, lines)),
        crs=crs,
    )
    gdf.to_file(outpath, driver="GPKG")


def generate_exterior_linestrings_from_polygon(polygon: Polygon) -> list[LineString]:
    x, y = zip(*polygon.exterior.coords)
    x = np.stack([x[:-1], x[1:]], axis=1)
    y = np.stack([y[:-1], y[1:]], axis=1)

    linestrings: list[LineString] = []
    for coords in zip(x, y):
        linestrings.append(LineString(zip(*coords)))
    return linestrings


@dataclass
class Camera:
    S: np.ndarray  # 1x2 size of image before rectification
    K: np.ndarray  # 3x3 calibration matrix of camera before rectification
    D: np.ndarray  # 1x5 distortion vector of camera before rectification
    R: np.ndarray  # 3x3 rotation matrix of camera (extrinsic)
    T: np.ndarray  # 3x1 translation vector of camera (extrinsic)
    S_rect: np.ndarray  # 1x2 size of image after rectification
    R_rect: np.ndarray  # 3x3 rectifying rotation to make image planes co-planar
    P_rect: np.ndarray  # 3x4 projection matrix after rectification

    @classmethod
    def from_dict(cls, dict_: dict[str, np.ndarray]) -> "Camera":
        for key, val in dict_.items():
            if "s" in key.lower():
                dict_[key] = val.reshape(1, 2)
            elif key.lower() in ["k", "r", "r_rect"]:
                dict_[key] = val.reshape(3, 3)
            elif key.lower() == "d":
                dict_[key] = val.reshape(1, 5)
            elif key.lower() == "t":
                dict_[key] = val.reshape(3, 1)
            elif key.lower() == "p_rect":
                dict_[key] = val.reshape(3, 4)
        return cls(**dict_)

    @classmethod
    def from_txt_file(cls, txt_file: Path) -> list["Camera"]:
        with open(str(txt_file), 'r') as f:
            lines = [line.strip("\n") for line in f.readlines()]
        cameras: list["Camera"] = []
        camera_dicts: dict[str, dict[str, np.ndarray]] = {}
        for line in lines:
            split_ = line.split("_0")
            if len(split_) == 1:
                continue
            camera_key = split_[0]
            camera_val = split_[1]
            camera_val_split = camera_val.split(": ")
            camera_idx = camera_val_split[0]
            camera_arr = np.asarray(list(map(float, camera_val_split[-1].split(" "))))
            camera_dicts.setdefault(camera_idx, {})[camera_key] = camera_arr

        for camera_idx, camera_dict in camera_dicts.items():
            cameras.append(cls.from_dict(camera_dict))
        return cameras


@dataclass
class Transform:
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3x1

    @classmethod
    def from_dict(cls, dict_: dict[str, np.ndarray]) -> "Transform":
        new_dict: dict[str, np.ndarray] = {}
        for key, val in dict_.items():
            if key.lower() == "r":
                new_dict[key] = val.reshape(3, 3)
            elif key.lower() == "t":
                new_dict[key.lower()] = val.reshape(3, 1)
            else:
                raise ValueError(f"Got key not in 'Transform': '{key}'")
        return cls(**new_dict)

    @classmethod
    def from_txt_file(cls, txt_file: Path, skip_other_keys: bool = True) -> "Transform":
        with open(str(txt_file), 'r') as f:
            lines = [line.strip("\n") for line in f.readlines()]
        dict_: dict[str, np.ndarray] = {}
        for line in lines:
            if "calib_time" in line.lower():
                continue
            split_ = line.split(": ")
            if skip_other_keys and split_[0].lower() not in [key.lower() for key in list(cls.__annotations__.keys())]:
                continue
            dict_[split_[0]] = np.asarray(list(map(float, split_[-1].split(" "))))
        return cls.from_dict(dict_)


def ingest_calibration_data(calib_dir: Path) -> tuple[list[Camera], Optional[Transform], Optional[Transform]]:
    velo: Optional[Transform] = None
    imu: Optional[Transform] = None
    cameras: list[Camera] = []
    for txt_file in sorted(calib_dir.glob("**/*.txt")):
        if "cam_to_cam" in txt_file.stem.lower():
            cameras = Camera.from_txt_file(txt_file)
        elif "velo_to_cam" in txt_file.stem.lower():
            velo = Transform.from_txt_file(txt_file)
        elif "imu_to_velo" in txt_file.stem.lower():
            imu = Transform.from_txt_file(txt_file)
        else:
            raise NotImplementedError(f"'{txt_file}' not supported.")
    return cameras, velo, imu


def plot_3_dof_values(
    df: pd.DataFrame,
    column_names: tuple[str, str, str],
    ts: np.ndarray,
    output_path: Path,
    units: str | list[str],
    colors: tuple[str, str, str] = ('r', 'g', 'b'),
    ts_units: str = "(s)",
    y_on_same_axis: bool | list[bool] = True,
):
    if isinstance(units, str):
        units = [units] * 3
    if len(units) != 3:
        raise ValueError(f"Improper number of units provided. Got '{len(units)}' 3.")
    if isinstance(y_on_same_axis, bool):
        y_on_same_axis = [y_on_same_axis] * 3
    if len(y_on_same_axis) != 3:
        raise ValueError(f"Improper number of 'y_on_same_axis' provided. Got '{len(y_on_same_axis)}' expected 3.")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    min_ = 1e12
    max_ = 0.0
    stds_ = []
    for idx, column_name in enumerate(column_names):
        if y_on_same_axis[idx]:
            min_ = min(np.min(df[column_name]), min_)
            max_ = max(np.max(df[column_name]), max_)
            stds_.append(np.std(df[column_name]))
    for idx, column_name in enumerate(column_names):
        axs[idx].plot(ts, df[column_name], colors[idx])
        axs[idx].set_title(column_name)
        axs[idx].set_ylabel(f"{column_name} {units[idx]}")
        if y_on_same_axis[idx]:
            axs[idx].set_ylim(min_ - np.mean(stds_), max_ + np.mean(stds_))
    axs[-1].set_xlabel(f"Time {ts_units}")
    plt.tight_layout()
    fig.savefig(str(output_path))
    plt.close()


NavstatToStringMap: dict[int, str] = {
    0: "invalid",
    1: "fix",
    2: "differential",
    3: "rtk",
    4: "rtk_float",
}
StringToNavsatMap: dict[str, int] = {
    "invalid": 0,
    "fix": 1,
    "differential": 2,
    "rtk": 3,
    "rtk_float": 4,
}


def navstat_to_string(navstat: int) -> str:
    return NavstatToStringMap.get(navstat, "unknown")


def string_to_navstat(navstat: str) -> int:
    return StringToNavsatMap.get(navstat, 5)


def plot_navstat(
    navstats: list[str] | list[int],
    out_path: Path,
):
    if isinstance(navstats[0], str):
        navstats = list(map(string_to_navstat, navstats))
    vals, cnts = np.unique(navstats, return_counts=True)
    cnts_ = np.zeros((len(NavstatToStringMap)))
    for idx, val in enumerate(vals):
        cnts_[val] = cnts[idx]
    plt.bar(list(NavstatToStringMap.keys()), cnts_)
    plt.xticks(list(NavstatToStringMap.keys()), list(NavstatToStringMap.values()), rotation=45)
    plt.xlabel("Navstat")
    plt.ylabel("Count")
    plt.title("Count of navstats")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


@dataclass
class OxtsData:
    lat: float    # latitude of the oxts-unit (deg)
    lon: float    # longitude of the oxts-unit (deg)
    alt: float    # altitude of the oxts-unit (m)
    roll: float   # roll angle (rad),  0 = level, positive = left side up,     range: -pi   .. +pi
    pitch: float  # pitch angle(rad),  0 = level, positive = front down,       range: -pi/2 .. +pi/2
    yaw: float    # pitch angle (rad), 0 = east,  positive = counterclockwise, range: -pi   .. +pi
    vn: float     # velocity towards north (m/s)
    ve: float     # velocity towards east (m/s)
    vf: float     # forward velocity, i.e., parallel to earth-surface (m/s)
    vl: float     # leftward velocity, i.e., parallel to earth-surface (m/s)
    vu: float     # upward velocity, i.e., perpendicular to earth-surface (m/s)
    ax: float     # acceleration in x, i.e., in direction of vehicle front (m/s^2)
    ay: float     # acceleration in y, i.e., in direction of vehicle left (m/s^2)
    az: float     # acceleration in z, i.e., in direction of vehicle top (m/s^2)
    af: float     # forward acceleration (m/s^2)
    al: float     # leftward acceleration (m/s^2)
    au: float     # upward acceleration (m/s^2)
    wx: float     # angular rate around x(rad/s)
    wy: float     # angular rate around y (rad/s)
    wz: float     # angular rate around z (rad/s)
    wf: float     # angular rate around forward axis (rad/s)
    wl: float     # angular rate around leftward axis (rad/s)
    wu: float     # angular rate around upward axis (rad/s)
    pos_accuracy: float  # velocity accuracy (north/east in m)
    vel_accuracy: float  # velocity accuracy (north/east in m/s)
    navstat: int         # navigation status (see navstat_to_string)
    numsats: int         # number of satellites tracked by primary GPS receiver
    posmode: int         # position mode of primary GPS receiver (see gps_mode_to_string)
    velmode: int         # velocity mode of primary GPS receiver (see gps_mode_to_string)
    orimode: int         # orientation mode of primary GPS receiver (see gps_mode_to_string)

    @classmethod
    def get_default_dataframe(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=["ts", *list(cls.__annotations__.keys())])

    def to_array(self) -> np.ndarray:
        return np.asarray(list(self.__dict__.values()))

    def navstat_to_string(self) -> str:
        return navstat_to_string(self.navstat)

    @staticmethod
    def gps_mode_to_string(mode: float | int) -> str:
        if int(mode) == 0:
            return "invalid"
        elif int(mode) == 1:
            return "standalone"
        elif int(mode) == 2:
            return "differential"
        elif int(mode) == 4:
            return "rtk_float"
        elif int(mode) == 5:
            return "rtk_fixed"
        else:
           return "unknown"

    def posmode_to_string(self) -> str:
        return self.gps_mode_to_string(self.posmode)

    def velmode_to_string(self) -> str:
        return self.gps_mode_to_string(self.velmode)

    def orimode_to_string(self) -> str:
        return self.gps_mode_to_string(self.orimode)


@dataclass
class OxtsDataTs:
    ts: str | pd.Timestamp
    data: OxtsData

    @classmethod
    def from_txt_file(cls, txt_file_path: Path, keys: list[str], ts: str | pd.Timestamp) -> "OxtsDataTs":
        with open(str(txt_file_path), 'r') as f:
            lines = [line.strip("\n") for line in f.readlines()]
        if len(lines) > 1:
            raise NotImplementedError(f"Multiple lines of data found in '{txt_file_path}'.")
        data_dict: dict[str, float] = {}
        for idx, val in enumerate(lines[0].split(" ")):
            data_dict[keys[idx]] = float(val)
        return cls(
            ts=ts,
            data=OxtsData(**data_dict),
        )

    def to_pandas_series(self) -> pd.Series:
        return pd.Series(
            index=["ts", *list(self.data.__annotations__.keys())],
            data=[
                float(self.ts) if isinstance(self.ts, str) else self.ts.value,
                *list(self.data.to_array()[:25]),
                self.data.navstat_to_string(),
                int(self.data.numsats),
                self.data.posmode_to_string(),
                self.data.velmode_to_string(),
                self.data.orimode_to_string(),
            ]
        )


def ingest_oxts_data(oxts_dir: Path, out_plots_dir: Optional[Path] = None) -> pd.DataFrame:
    if out_plots_dir is None:
        out_plots_dir = oxts_dir / "plots"

    # Dataformat
    dataformat_txt_path = sorted(oxts_dir.glob("**/dataformat.txt"))
    if len(dataformat_txt_path) == 0:
        raise RuntimeError(f"Could not find 'dataformat.txt' at '{oxts_dir}'.")
    elif len(dataformat_txt_path) > 1:
        raise RuntimeError(f"Found multiple 'dataformat.txt' at '{oxts_dir}'.")
    dataformat_txt_path = dataformat_txt_path[0]
    with open(str(dataformat_txt_path), 'r') as f:
        data_keys = [line.strip("\n").split(":")[0] for line in f.readlines()]

    # Timestamps
    timestamps_txt_path = sorted(oxts_dir.glob("**/timestamps.txt"))
    if len(timestamps_txt_path) == 0:
        raise RuntimeError(f"Could not find 'timestamps.txt' at '{oxts_dir}'.")
    elif len(timestamps_txt_path) > 1:
        raise RuntimeError(f"Found multiple 'timestamps.txt' at '{oxts_dir}'.")
    timestamps_txt_path = timestamps_txt_path[0]

    with open(str(timestamps_txt_path), 'r') as f:
        timestamps = [line.strip("\n") for line in f.readlines()]
    for stamp_idx in range(len(timestamps)):
        dt = datetime.strptime(timestamps[stamp_idx][:26], '%Y-%m-%d %H:%M:%S.%f')
        dt = f"{dt.timestamp()}"
        if timestamps[stamp_idx][25] == "0":
            dt += "0"
        dt += timestamps[stamp_idx][26:]
        timestamps[stamp_idx] = dt

    # Data
    data_dir = sorted(oxts_dir.glob("**/data"))
    if len(data_dir) == 0:
        raise RuntimeError(f"Could not find 'data' at '{oxts_dir}'.")
    elif len(data_dir) > 1:
        raise RuntimeError(f"Found multiple 'data' at '{oxts_dir}'.")
    data_dir = data_dir[0]

    oxts_df: pd.DataFrame = OxtsData.get_default_dataframe()
    for ts, data_txt_file in tqdm(zip(timestamps, sorted(data_dir.glob("**/*.txt"))), desc="Loading OXTS Data", total=len(timestamps)):
        oxts_data = OxtsDataTs.from_txt_file(data_txt_file, data_keys, ts)
        oxts_df.loc[len(oxts_df)] = oxts_data.to_pandas_series()
    oxts_df.sort_values(by=["ts"], inplace=True)
    ts = np.asarray(oxts_df["ts"])
    plot_kwargs: list[dict[str, Any]] = [
        {
            "column_names": ("lat", "lon", "alt"),
            "units": ["deg", "deg", "m"],
            "output_path": out_plots_dir / "latlonalt.png",
        },
        {
            "column_names": ("lat", "lon", "pos_accuracy"),
            "units": ["deg", "deg", "sigma"],
            "y_on_same_axis": False,
            "output_path": out_plots_dir / "latlon-pos_accuracy.png",
        },
        {
            "column_names": ("roll", "pitch", "yaw"),
            "units": "rad",
            "output_path": out_plots_dir / "rollpitchyaw.png",
        },
        {
            "column_names": ("vf", "vl", "vu"),
            "units": "m/s",
            "output_path": out_plots_dir / "velocity_to_earth-surface.png",
        },
        {
            "column_names": ("ax", "ay", "az"),
            "units": "m/s^2",
            "output_path": out_plots_dir / "acceleration_in_dir_of_vehicle.png",
        },
        {
            "column_names": ("af", "al", "au"),
            "units": "m/s^2",
            "output_path": out_plots_dir / "acceleration.png",
        },
        {
            "column_names": ("wx", "wy", "wz"),
            "units": "rad/s",
            "output_path": out_plots_dir / "angular_rate_around_xyz.png",
        },
        {
            "column_names": ("wf", "wl", "wu"),
            "units": "rad/s",
            "output_path": out_plots_dir / "angular_rate_around_axis.png",
        },
    ]
    out_plots_dir.mkdir(exist_ok=True, parents=True)

    for plot_kwargs_ in plot_kwargs:
        plot_3_dof_values(
            df=oxts_df,
            ts=ts,
            **plot_kwargs_,
        )

    # Plot GPS values
    plot_navstat(list(oxts_df["navstat"]), out_plots_dir / "navstats.png")

    linestrings: list[LineString] = []
    for idx in range(len(oxts_df)-1):
        # TODO: Color code linestrings based on certainty of gps data
        lat0, lon0 = oxts_df.iloc[idx]["lat"], oxts_df.iloc[idx]["lon"]
        lat1, lon1 = oxts_df.iloc[idx+1]["lat"], oxts_df.iloc[idx+1]["lon"]
        navstat0, navstat1 = oxts_df.iloc[idx]["navstat"], oxts_df.iloc[idx+1]["navstat"]
        numsats0, numsats1 = oxts_df.iloc[idx]["numsats"], oxts_df.iloc[idx+1]["numsats"]

        # Invalid GPS Data
        if navstat0 == 0 or navstat1 == 0:
            continue

        linestrings.append(LineString([(lat0, lon0), (lat1, lon1)]))
    lines_to_gpkg(linestrings, outpath=out_plots_dir / "latlon.gpkg", func=save_lines_func, crs="EPSG:4326")

    return oxts_df


def main():
    cameras, velo, imu = ingest_calibration_data(CALIB_PATH)
    oxts_data = ingest_oxts_data(sorted(SYNC_PATH.glob("**/oxts"))[0])


if __name__ == '__main__':
    main()

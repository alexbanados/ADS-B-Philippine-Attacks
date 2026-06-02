from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PH_TIMEZONE = ZoneInfo("Asia/Manila")
REQUIRED_COLUMNS = ["timestamp", "altitude_meters", "speed_kmh"]

PHASE_GROUND = 0
PHASE_TAKEOFF = 1
PHASE_INITIAL_CLIMB = 2
PHASE_CLIMB = 3
PHASE_CRUISE = 4
PHASE_DESCENT = 5
PHASE_APPROACH = 6

PHASE_LABELS = {
    PHASE_GROUND: "Ground",
    PHASE_TAKEOFF: "Takeoff",
    PHASE_INITIAL_CLIMB: "Initial Climb",
    PHASE_CLIMB: "Climb",
    PHASE_CRUISE: "Cruise",
    PHASE_DESCENT: "Descent",
    PHASE_APPROACH: "Approach",
}
PHASE_VALUES = {label: value for value, label in PHASE_LABELS.items()}

LEVEL_FLIGHT_PHASES = {PHASE_CLIMB, PHASE_DESCENT}
LEVEL_FLIGHT_MIN_ALTITUDE_METERS = 500
LEVEL_FLIGHT_MIN_DURATION_SECONDS = 10
LEVEL_FLIGHT_BOUNDARY_BUFFER_SECONDS = 60 #This excludes rows within X seconds after takeoff.
LEVEL_FLIGHT_CRUISE_BUFFER_SECONDS = 180 #This excludes rows within X seconds of cruise.
LEVEL_FLIGHT_WINDOW_SECONDS = 10
LEVEL_FLIGHT_WINDOW_STEP_SECONDS = 1
LEVEL_FLIGHT_MIN_WINDOW_DURATION_SECONDS = 6
LEVEL_FLIGHT_CLUSTER_COUNT = 3
LEVEL_FLIGHT_CLUSTER_SCORE_MARGIN = 1.25
LEVEL_FLIGHT_MAX_SELECTED_CLUSTERS = 2
LEVEL_FLIGHT_WINDOW_SCORE_QUANTILE = 0.80


def load_flight(csv_path):
    """Load one processed flight CSV and clean key columns for segmentation."""
    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["Time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["Time"] = df["Time"].dt.tz_convert(PH_TIMEZONE)
    df["Time_local"] = df["Time"].dt.tz_localize(None)
    df["altitude_meters"] = pd.to_numeric(df["altitude_meters"], errors="coerce")
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")

    return df.dropna(subset=["timestamp", "Time_local", "altitude_meters"]).reset_index(
        drop=True
    )


def segment_flight_phases(df, label_phases=False):
    """
    Label rows using the expected altitude pattern:
    Ground -> Takeoff -> Climb -> Cruise -> Descent -> Ground.
    """
    df = df.copy()
    df["phase"] = PHASE_GROUND
    df["is_level"] = 0

    if df.empty:
        return label_phase_values(df) if label_phases else df

    smooth_window = get_smooth_window(len(df))
    altitude = df["altitude_meters"]
    smoothed_altitude = smooth_altitude(altitude, smooth_window)

    max_altitude = smoothed_altitude.max()
    if pd.isna(max_altitude) or max_altitude <= 0:
        return label_phase_values(df) if label_phases else df

    airborne = altitude != 0
    if not airborne.any():
        return label_phase_values(df) if label_phases else df

    first_airborne_idx = airborne.idxmax()
    last_airborne_idx = airborne[::-1].idxmax()

    vertical_speed = get_vertical_speed(df, smoothed_altitude, smooth_window).fillna(0)
    speed = df["speed_kmh"].fillna(0)
    takeoff_start_idx, takeoff_end_idx = final_takeoff_run(
        altitude,
        speed,
        vertical_speed,
        first_airborne_idx,
    )
    if takeoff_start_idx is not None:
        df.loc[takeoff_start_idx : first_airborne_idx - 1, "phase"] = PHASE_TAKEOFF
        if takeoff_end_idx >= first_airborne_idx:
            df.loc[first_airborne_idx:takeoff_end_idx, "phase"] = PHASE_INITIAL_CLIMB

    cruise_band = max(120, max_altitude * 0.01)
    cruise_floor = max_altitude - cruise_band

    cruise_candidates = (
        (smoothed_altitude >= cruise_floor)
        & (vertical_speed.abs() <= 1.0)
        & (df.index >= first_airborne_idx)
        & (df.index <= last_airborne_idx)
    )

    if cruise_candidates.any():
        cruise_start_idx, cruise_end_idx = longest_true_run(cruise_candidates)
    else:
        cruise_start_idx = smoothed_altitude.idxmax()
        cruise_end_idx = cruise_start_idx

    climb_start_idx = first_airborne_idx
    if takeoff_end_idx is not None:
        climb_start_idx = max(climb_start_idx, takeoff_end_idx + 1)

    df.loc[climb_start_idx : cruise_start_idx - 1, "phase"] = PHASE_CLIMB
    df.loc[cruise_start_idx:cruise_end_idx, "phase"] = PHASE_CRUISE
    df.loc[cruise_end_idx + 1 : last_airborne_idx, "phase"] = PHASE_DESCENT
    df = assign_level_flags(df)
    df = assign_approach_phase(df, last_airborne_idx)

    return label_phase_values(df) if label_phases else df


def label_phase_values(df):
    """Return a copy with numeric phase values converted to display labels."""
    df = df.copy()
    df["phase"] = df["phase"].map(PHASE_LABELS).fillna(df["phase"])
    return df


def get_smooth_window(row_count):
    """Return the odd rolling window size used by the phase segmentation logic."""
    smooth_window = min(9, max(3, row_count // 40))
    if smooth_window % 2 == 0:
        smooth_window += 1
    return smooth_window


def smooth_altitude(altitude, smooth_window):
    """Return a smoothed altitude series while preserving row alignment."""
    return altitude.rolling(
        window=smooth_window,
        center=True,
        min_periods=1,
    ).median()


def get_vertical_speed(df, smoothed_altitude, smooth_window):
    """Return a smoothed vertical-speed series in meters per second."""
    if "verticalSpeed_ms" in df.columns:
        vertical_speed = pd.to_numeric(df["verticalSpeed_ms"], errors="coerce")
    else:
        time_delta = df["Time_local"].diff().dt.total_seconds()
        vertical_speed = smoothed_altitude.diff() / time_delta

    return vertical_speed.rolling(
        window=smooth_window,
        center=True,
        min_periods=1,
    ).median()


def assign_level_flags(df):
    """Flag clustered level-flight spans while keeping climb/descent phases."""
    df = df.copy()

    if df.empty:
        return df

    smooth_window = get_smooth_window(len(df))
    altitude = df["altitude_meters"]
    smoothed_altitude = smooth_altitude(altitude, smooth_window)
    vertical_speed = get_vertical_speed(df, smoothed_altitude, smooth_window)

    candidate_rows = (
        df["phase"].isin(LEVEL_FLIGHT_PHASES)
        & (altitude >= LEVEL_FLIGHT_MIN_ALTITUDE_METERS)
        & away_from_takeoff_and_cruise(df)
    )

    window_features = build_level_flight_windows(df, vertical_speed, candidate_rows)
    if window_features.empty:
        return df

    feature_columns = [
        "mean_vertical_speed",
        "altitude_variance",
        "slope",
        "speed_variance",
    ]
    cluster_count = min(LEVEL_FLIGHT_CLUSTER_COUNT, len(window_features))
    window_features["cluster"] = kmeans_clusters(
        window_features[feature_columns],
        cluster_count,
    )
    window_features["level_score"] = level_window_scores(window_features)
    level_clusters = choose_level_clusters(window_features)
    level_score_limit = window_features["level_score"].quantile(
        LEVEL_FLIGHT_WINDOW_SCORE_QUANTILE
    )
    level_windows = window_features[
        window_features["cluster"].isin(level_clusters)
        | (window_features["level_score"] <= level_score_limit)
    ]

    level_rows = pd.Series(False, index=df.index)
    for _, window in level_windows.iterrows():
        level_rows.loc[window["start_idx"] : window["end_idx"]] = True

    level_rows &= candidate_rows
    level_rows = drop_short_level_runs(df, level_rows)

    df.loc[level_rows, "is_level"] = 1

    return df


def assign_approach_phase(df, last_airborne_idx):
    """Label rows after the final level-descent segment until landing as approach."""
    df = df.copy()
    level_descent_rows = df[
        (df["phase"] == PHASE_DESCENT)
        & (df["is_level"] == 1)
    ]
    if level_descent_rows.empty:
        return df

    approach_start_idx = level_descent_rows.index[-1] + 1
    if approach_start_idx <= last_airborne_idx:
        df.loc[approach_start_idx:last_airborne_idx, "phase"] = PHASE_APPROACH

    return df


def build_level_flight_windows(df, vertical_speed, candidate_rows):
    """Return per-window features for rows eligible for level-flight clustering."""
    feature_rows = []

    for start_idx, end_idx, _ in true_runs(candidate_rows):
        group = df.loc[start_idx:end_idx].copy()
        start_position = 0

        while start_position < len(group):
            start_time = group["Time_local"].iloc[start_position]
            target_end_time = start_time + pd.Timedelta(
                seconds=LEVEL_FLIGHT_WINDOW_SECONDS
            )
            window = group[
                (group["Time_local"] >= start_time)
                & (group["Time_local"] <= target_end_time)
            ]

            if len(window) >= 3:
                duration_seconds = (
                    window["Time_local"].iloc[-1] - window["Time_local"].iloc[0]
                ).total_seconds()
                if duration_seconds >= LEVEL_FLIGHT_MIN_WINDOW_DURATION_SECONDS:
                    feature_rows.append(
                        level_flight_window_features(window, vertical_speed)
                    )

            next_start_time = start_time + pd.Timedelta(
                seconds=LEVEL_FLIGHT_WINDOW_STEP_SECONDS
            )
            next_positions = np.flatnonzero(group["Time_local"] >= next_start_time)
            if len(next_positions) == 0:
                break
            next_position = int(next_positions[0])
            start_position = max(next_position, start_position + 1)

    if not feature_rows:
        return pd.DataFrame()

    return pd.DataFrame(feature_rows)


def level_flight_window_features(window, vertical_speed):
    """Compute clustering features for one candidate time window."""
    window_vertical_speed = vertical_speed.loc[window.index]
    timestamp = pd.to_numeric(window["timestamp"], errors="coerce")
    elapsed_seconds = timestamp - timestamp.iloc[0]
    altitude = window["altitude_meters"]
    speed = window["speed_kmh"]

    valid_slope_rows = elapsed_seconds.notna() & altitude.notna()
    if valid_slope_rows.sum() >= 2 and elapsed_seconds[valid_slope_rows].max() > 0:
        slope = np.polyfit(
            elapsed_seconds[valid_slope_rows],
            altitude[valid_slope_rows],
            1,
        )[0]
    else:
        slope = 0

    return {
        "start_idx": window.index[0],
        "end_idx": window.index[-1],
        "mean_vertical_speed": window_vertical_speed.mean(),
        "altitude_variance": altitude.var(ddof=0),
        "slope": slope,
        "speed_variance": speed.var(ddof=0),
    }


def kmeans_clusters(features, cluster_count):
    """Cluster standardized feature rows with deterministic K-means."""
    values = features.astype(float).fillna(0).to_numpy()
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    scaled = (values - means) / stds

    centroids = initial_centroids(scaled, cluster_count)
    labels = np.zeros(len(scaled), dtype=int)

    for _ in range(100):
        distances = ((scaled[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels

        for cluster in range(cluster_count):
            cluster_values = scaled[labels == cluster]
            if len(cluster_values) > 0:
                centroids[cluster] = cluster_values.mean(axis=0)

    return labels


def initial_centroids(values, cluster_count):
    """Choose deterministic starting centroids across the first feature axis."""
    if cluster_count == 1:
        return values[[0]].copy()

    order = np.argsort(values[:, 0])
    positions = np.linspace(0, len(order) - 1, cluster_count).round().astype(int)
    return values[order[positions]].copy()


def choose_level_clusters(window_features):
    """Pick the flattest cluster, plus near-flat clusters when clearly close."""
    cluster_summary = window_features.groupby("cluster").agg(
        mean_vertical_speed=("mean_vertical_speed", lambda values: values.abs().mean()),
        altitude_variance=("altitude_variance", "mean"),
        slope=("slope", lambda values: values.abs().mean()),
        speed_variance=("speed_variance", "mean"),
    )

    score_columns = [
        "mean_vertical_speed",
        "altitude_variance",
        "slope",
        "speed_variance",
    ]
    normalized = cluster_summary[score_columns].copy()
    for column in score_columns:
        column_range = normalized[column].max() - normalized[column].min()
        if column_range == 0:
            normalized[column] = 0
        else:
            normalized[column] = (
                normalized[column] - normalized[column].min()
            ) / column_range

    score = (
        normalized["mean_vertical_speed"]
        + normalized["slope"]
        + normalized["altitude_variance"]
        + (0.5 * normalized["speed_variance"])
    )
    best_score = score.min()
    selected = score[score <= best_score + LEVEL_FLIGHT_CLUSTER_SCORE_MARGIN]
    selected = selected.sort_values().head(LEVEL_FLIGHT_MAX_SELECTED_CLUSTERS)
    return set(selected.index)


def level_window_scores(window_features):
    """Return per-window flatness scores so flat windows in mixed clusters survive."""
    score_columns = [
        "mean_vertical_speed",
        "altitude_variance",
        "slope",
        "speed_variance",
    ]
    scored = pd.DataFrame(index=window_features.index)
    scored["mean_vertical_speed"] = window_features["mean_vertical_speed"].abs()
    scored["altitude_variance"] = window_features["altitude_variance"]
    scored["slope"] = window_features["slope"].abs()
    scored["speed_variance"] = window_features["speed_variance"]

    for column in score_columns:
        column_range = scored[column].max() - scored[column].min()
        if column_range == 0:
            scored[column] = 0
        else:
            scored[column] = (scored[column] - scored[column].min()) / column_range

    return (
        scored["mean_vertical_speed"]
        + scored["slope"]
        + scored["altitude_variance"]
        + (0.5 * scored["speed_variance"])
    )


def drop_short_level_runs(df, level_rows):
    """Remove level-flight runs shorter than the minimum duration."""
    level_rows = level_rows.copy()
    for start_idx, end_idx, _ in true_runs(level_rows):
        start_time = df.loc[start_idx, "Time_local"]
        end_time = df.loc[end_idx, "Time_local"]
        duration_seconds = (end_time - start_time).total_seconds()
        if duration_seconds < LEVEL_FLIGHT_MIN_DURATION_SECONDS:
            level_rows.loc[start_idx:end_idx] = False

    return level_rows


def away_from_takeoff_and_cruise(df):
    """Return rows outside the buffer around takeoff and cruise boundaries."""
    mask = pd.Series(True, index=df.index)

    takeoff_rows = df[df["phase"] == PHASE_TAKEOFF]
    if not takeoff_rows.empty:
        takeoff_end_time = takeoff_rows["Time_local"].iloc[-1]
        seconds_after_takeoff = (df["Time_local"] - takeoff_end_time).dt.total_seconds()
        mask &= seconds_after_takeoff > LEVEL_FLIGHT_BOUNDARY_BUFFER_SECONDS

    cruise_rows = df[df["phase"] == PHASE_CRUISE]
    if not cruise_rows.empty:
        cruise_start_time = cruise_rows["Time_local"].iloc[0]
        cruise_end_time = cruise_rows["Time_local"].iloc[-1]
        seconds_from_cruise_start = (
            (df["Time_local"] - cruise_start_time).dt.total_seconds().abs()
        )
        seconds_from_cruise_end = (
            (df["Time_local"] - cruise_end_time).dt.total_seconds().abs()
        )
        mask &= seconds_from_cruise_start > LEVEL_FLIGHT_CRUISE_BUFFER_SECONDS
        mask &= seconds_from_cruise_end > LEVEL_FLIGHT_CRUISE_BUFFER_SECONDS

    return mask


def final_takeoff_run(altitude, speed, vertical_speed, first_airborne_idx):
    """
    Return the final takeoff segment before climb.

    The segment starts at the final zero-altitude speed-increase run before the
    first airborne row. It can continue after altitude becomes nonzero, stopping
    at the later row before either vertical speed or speed first decreases.
    """
    last_ground_idx = first_airborne_idx - 1
    if last_ground_idx not in altitude.index or altitude.loc[last_ground_idx] != 0:
        return None, None
    if speed.loc[last_ground_idx] <= 0:
        return None, None

    takeoff_start_idx = last_ground_idx

    while takeoff_start_idx - 1 in altitude.index:
        previous_idx = takeoff_start_idx - 1
        if altitude.loc[previous_idx] != 0:
            break
        if speed.loc[previous_idx] <= 0:
            break
        if speed.loc[previous_idx] > speed.loc[takeoff_start_idx]:
            break
        takeoff_start_idx = previous_idx

    vertical_speed_end_idx = row_before_first_decrease(vertical_speed, last_ground_idx)
    speed_end_idx = row_before_first_decrease(speed, last_ground_idx)
    takeoff_end_idx = max(vertical_speed_end_idx, speed_end_idx)

    return takeoff_start_idx, takeoff_end_idx


def row_before_first_decrease(values, start_idx):
    """Return the row before values first decrease after start_idx."""
    end_idx = start_idx
    while end_idx + 1 in values.index:
        next_idx = end_idx + 1
        if values.loc[next_idx] < values.loc[end_idx]:
            break
        end_idx = next_idx

    return end_idx


def longest_true_run(mask):
    """Return the start and end index labels of the longest contiguous True run."""
    best_start = None
    best_end = None
    best_length = 0
    current_start = None
    current_length = 0

    for idx, value in mask.items():
        if value:
            if current_start is None:
                current_start = idx
                current_length = 0
            current_length += 1
            if current_length > best_length:
                best_start = current_start
                best_end = idx
                best_length = current_length
        else:
            current_start = None
            current_length = 0

    return best_start, best_end


def true_runs(mask):
    """Yield start index, end index, and length for each contiguous True run."""
    current_start = None
    current_length = 0
    current_end = None

    for idx, value in mask.items():
        if value:
            if current_start is None:
                current_start = idx
                current_length = 0
            current_end = idx
            current_length += 1
        elif current_start is not None:
            yield current_start, current_end, current_length
            current_start = None
            current_length = 0
            current_end = None

    if current_start is not None:
        yield current_start, current_end, current_length


def phase_summary(df):
    """Return a compact count of how many rows were assigned to each phase."""
    return df["phase"].value_counts(sort=False).to_dict()

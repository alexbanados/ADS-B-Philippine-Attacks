# Derived Flight Features

All distance features are in meters. Rate features use seconds from the
`timestamp` column.

After computing features, the script validates the derived columns and reports
the filename plus CSV row number for:

- missing values
- infinite values
- non-numeric values in required numeric input columns
- duplicate timestamps after the first row
- negative timesteps
- `route_progress` values outside `0` to `1`

## Required Input Columns

- `timestamp`
- `latitude`
- `longitude`
- `altitude_meters`
- `speed_kmh`
- `heading`

## Derived Columns

### `sequence_index`

Zero-based row number after timestamp sorting in `1_preprocess/csv_1preprocessing.py`.

### `t_elapsed_sec`

Seconds elapsed since the first row.

```text
t_elapsed_sec = current timestamp - first timestamp
```

### `dt`

Seconds elapsed since the previous row. The first row is `0`.

```text
dt = current timestamp - previous timestamp
```

### `t_norm`

Elapsed time normalized by total flight duration.

```text
t_norm = t_elapsed_sec / final t_elapsed_sec
```

When total duration is `0`, the result is set to `0`.

### `x_wrt0`

East-west distance from the first row's longitude, in meters.

```text
x_wrt0 = (longitude - start_longitude)
         * cos((latitude + start_latitude) / 2)
         * earth_radius_m
```

Latitudes and longitudes are converted to radians before computation.
Positive values are east of the starting point; negative values are west.

### `y_wrt0`

North-south distance from the first row's latitude, in meters.

```text
y_wrt0 = (latitude - start_latitude) * earth_radius_m
```

Latitudes are converted to radians before computation. Positive values are
north of the starting point; negative values are south.

### `delta_altitude`

Altitude change from the previous row, in meters.

```text
delta_altitude = current altitude_meters - previous altitude_meters
```

The first row is `0`.

### `delta_speed`

Speed change from the previous row, in km/h.

```text
delta_speed = current speed_kmh - previous speed_kmh
```

The first row is `0`.

### `delta_heading`

Heading change from the previous row, in degrees, wrapped to `-180` to `+180`.

```text
delta_heading = ((current heading - previous heading + 180) % 360) - 180
```

The first row is `0`.

### `turn_rate`

Heading change per second, in degrees/second.

```text
dt = current timestamp - previous timestamp
turn_rate = delta_heading / dt
```

When `dt` is `0`, the result is set to `0` to avoid division by zero.

### `acceleration`

Speed change per second, using `speed_kmh` units per second.

```text
dt = current timestamp - previous timestamp
acceleration = delta_speed / dt
```

When `dt` is `0`, the result is set to `0` to avoid division by zero.

### `distance_per_timestep`

Ground distance from the previous latitude/longitude to the current
latitude/longitude, in meters. It uses the Haversine formula:

```text
a = sin(delta_latitude / 2)^2
    + cos(previous_latitude)
      * cos(current_latitude)
      * sin(delta_longitude / 2)^2

distance_per_timestep = 2 * earth_radius_m * asin(sqrt(a))
```

Latitudes and longitudes are converted to radians before computation. The first
row is `0`.

### `route_progress`

Fraction of cumulative traveled distance, from `0.0` at the start to `1.0` at
the end.

```text
cumulative_distance = cumulative sum of distance_per_timestep
total_distance = final cumulative_distance

if total_distance > 0:
    route_progress = cumulative_distance / total_distance
else:
    route_progress = 0
```

If the flight has no movement, all rows are `0`.

ENDPOINT_COLOR = "#111827"

AIRPORT_POINTS = {
    "MNL": (14.513627, 121.014090),
    "CEB": (10.309000, 123.979000),
    "DVO": (7.125633, 125.645990),
    "ILO": (10.832583, 122.493237),
    "MPH": (11.926941, 121.958967),
    "PPS": (9.742127, 118.758210),
}
AIRPORT_COLOR = "#111827"


def mark_ground_endpoints(ax, df, x_column, y_column):
    if df.empty:
        return

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    ax.scatter(
        first_row[x_column],
        first_row[y_column],
        marker="s",
        s=70,
        facecolors="none",
        edgecolors=ENDPOINT_COLOR,
        linewidths=1.8,
        label="First Row",
        zorder=5,
    )
    ax.scatter(
        last_row[x_column],
        last_row[y_column],
        marker="x",
        s=80,
        color=ENDPOINT_COLOR,
        linewidths=2,
        label="Last Row",
        zorder=5,
    )


def mark_airport_points(ax):
    for code, (latitude, longitude) in AIRPORT_POINTS.items():
        ax.scatter(
            longitude,
            latitude,
            marker="o",
            s=46,
            color=AIRPORT_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
        ax.annotate(
            code,
            (longitude, latitude),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=AIRPORT_COLOR,
            zorder=7,
        )

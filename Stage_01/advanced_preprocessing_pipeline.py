"""
Advanced PySpark preprocessing pipeline — Porto Taxi Trajectory dataset.

Implements, end to end, using only native pyspark.sql.functions / Window ops
(no row-at-a-time Python UDFs, so every step stays vectorized and shuffles
only where a transformation genuinely requires one):

  1. Missing data         - GPS technical errors -> drop; structural nulls
                             in ORIGIN_CALL/ORIGIN_STAND -> sentinel category
  2. Outliers              - Winsorize distance/duration/speed at p95
  3. Skew transforms       - log1p / sqrt of the winsorized columns
  4. Categorical encoding  - one-hot CALL_TYPE; frequency-encode TAXI_ID and
                             ORIGIN_STAND (after collapsing rare stands)
  5. DAY_TYPE imbalance    - under-sampling helper + class-weight helper
"""

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType

POINT_SCHEMA = ArrayType(ArrayType(DoubleType()))

# Porto's rough bounding box (lon, lat) - points outside it are GPS receiver glitches
LON_MIN, LON_MAX = -8.75, -8.45
LAT_MIN, LAT_MAX = 41.05, 41.25
MAX_KMH = 200.0                 # implausible inter-point speed -> GPS glitch, not a real trip
WINSOR_PCT = 0.95                # cap the top 5% of distance/duration/speed
RARE_STAND_THRESHOLD = 0.01      # stands used by <1% of trips collapse into "Other_Stand"


# ---------------------------------------------------------------------------
# Load + base geometry (shared groundwork; not one of the 5 numbered steps)
# ---------------------------------------------------------------------------

def load_and_parse(spark: SparkSession, path: str) -> DataFrame:
    """Read the raw CSV and explode POLYLINE (JSON text) into a numeric points array."""
    raw = (spark.read.option("header", True).option("quote", '"').option("escape", '"').csv(path))
    return (raw
            .withColumn("points", F.from_json("POLYLINE", POINT_SCHEMA))
            .withColumn("n_points", F.size("points")))


def add_trip_geometry(df: DataFrame) -> DataFrame:
    """
    distance_km (Haversine over consecutive points), duration_sec (n_points-1)*15s,
    avg_kmh, and max_segment_km (used downstream to flag GPS jumps). All computed with
    Spark SQL higher-order array functions (zip_with/aggregate) - no UDF, no row collection.
    """
    R_KM = 6371.0
    seg_km_expr = f"""
        zip_with(
            slice(points, 1, size(points)-1),
            slice(points, 2, size(points)-1),
            (a, b) -> 2 * {R_KM} * asin(sqrt(
                pow(sin(radians(b[1]-a[1])/2), 2) +
                cos(radians(a[1])) * cos(radians(b[1])) * pow(sin(radians(b[0]-a[0])/2), 2)
            ))
        )
    """
    df = (df
          .withColumn("seg_km", F.expr(seg_km_expr))
          .withColumn("distance_km", F.expr("aggregate(seg_km, cast(0.0 as double), (acc, x) -> acc + x)"))
          .withColumn("max_segment_km", F.expr("array_max(seg_km)"))
          .withColumn("duration_sec", (F.col("n_points") - 1) * 15)
          .withColumn("avg_kmh", F.when(F.col("duration_sec") > 0,
                                        F.col("distance_km") / (F.col("duration_sec") / 3600.0))))
    df = df.drop("seg_km")

    # implausible single-segment speed (>MAX_KMH between two consecutive 15s pings) -> GPS glitch
    max_seg_km_limit = MAX_KMH * 15 / 3600
    return df.filter(F.col("max_segment_km") <= max_seg_km_limit)


# ---------------------------------------------------------------------------
# Step 1 - Missing data
# ---------------------------------------------------------------------------

def filter_gps_errors(df: DataFrame) -> DataFrame:
    """
    Step 1a - technical GPS errors are NOT informative and can't be imputed
    meaningfully, so (and only so for this category) we drop them outright:
      - MISSING_DATA flag set by the data provider
      - degenerate polylines (<2 points -> no measurable trip)
      - any point outside Porto's bounding box (receiver glitch / bad fix)
    The implausible-segment-speed check is applied in `add_trip_geometry`, since
    it needs distance/duration columns that don't exist yet at this point.
    This is filtering on data QUALITY, not on the analysis variables themselves,
    so it does not introduce the bias that listwise deletion of incomplete rows would.
    """
    df = df.filter((F.col("MISSING_DATA") == "False") & (F.col("n_points") >= 2))

    df = (df
          .withColumn("lon_min", F.expr("array_min(transform(points, p -> p[0]))"))
          .withColumn("lon_max", F.expr("array_max(transform(points, p -> p[0]))"))
          .withColumn("lat_min", F.expr("array_min(transform(points, p -> p[1]))"))
          .withColumn("lat_max", F.expr("array_max(transform(points, p -> p[1]))")))
    return df.filter(
        (F.col("lon_min") >= LON_MIN) & (F.col("lon_max") <= LON_MAX) &
        (F.col("lat_min") >= LAT_MIN) & (F.col("lat_max") <= LAT_MAX)
    ).drop("lon_min", "lon_max", "lat_min", "lat_max")


def fill_structural_missing(df: DataFrame) -> DataFrame:
    """
    Step 1b - ORIGIN_CALL / ORIGIN_STAND are missing **by construction**, not at
    random: ORIGIN_CALL is only ever populated when CALL_TYPE='A', ORIGIN_STAND
    only when CALL_TYPE='B' (verified on this dataset: null(ORIGIN_CALL) lines up
    almost exactly with CALL_TYPE != 'A'). Dropping those rows would discard the
    majority of the dataset for no good reason - instead we encode "not
    applicable for this trip" as its own explicit category.
    """
    df = df.withColumn(
        "ORIGIN_CALL_filled",
        F.when(F.col("ORIGIN_CALL").isNull(), "0").otherwise(F.col("ORIGIN_CALL"))
    )
    df = df.withColumn(
        "ORIGIN_STAND_filled",
        F.when(F.col("ORIGIN_STAND").isNull(), "Not_Applicable").otherwise(F.col("ORIGIN_STAND"))
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 - Outliers (Winsorizing)
# ---------------------------------------------------------------------------

def winsorize(df: DataFrame, cols, upper_pct: float = WINSOR_PCT, relative_error: float = 0.001):
    """
    Cap (never delete) legitimate-but-extreme values at the upper_pct percentile.
    `approxQuantile` is one cheap, single-pass driver-side action with a bounded
    relative error; the capping itself is then a fully vectorized `when/otherwise`
    column expression with no shuffle. Returns (df_with_*_wins_columns, thresholds).
    """
    thresholds = {}
    for c in cols:
        q = df.approxQuantile(c, [upper_pct], relative_error)[0]
        thresholds[c] = q
        df = df.withColumn(f"{c}_wins", F.when(F.col(c) > q, F.lit(q)).otherwise(F.col(c)))
    return df, thresholds


# ---------------------------------------------------------------------------
# Step 3 - Skew-reducing transforms
# ---------------------------------------------------------------------------

def add_skew_transforms(df: DataFrame, cols):
    """log1p and sqrt of the winsorized columns, to pull in the right-skewed tail."""
    for c in cols:
        src = f"{c}_wins"
        df = df.withColumn(f"{c}_log", F.log1p(F.col(src)))
        df = df.withColumn(f"{c}_sqrt", F.sqrt(F.col(src)))
    return df


# ---------------------------------------------------------------------------
# Step 4 - Categorical encoding
# ---------------------------------------------------------------------------

def one_hot_call_type(df: DataFrame) -> DataFrame:
    """Step 4a - CALL_TYPE has a fixed, tiny cardinality (A/B/C) -> explicit dummy columns."""
    for v in ["A", "B", "C"]:
        df = df.withColumn(f"CALL_TYPE_{v}", (F.col("CALL_TYPE") == v).cast("int"))
    return df


def frequency_encode(df: DataFrame, col_name: str, out_name: str = None) -> DataFrame:
    """
    Step 4b - frequency/count encoding via a Window function (partition-local count,
    no groupBy+join shuffle). Replaces a high-cardinality id (442 distinct taxis,
    dozens of stands) with its occurrence count, avoiding the dimensionality blow-up
    one-hot encoding would cause on IDs like these.
    """
    out_name = out_name or f"{col_name}_freq"
    w = Window.partitionBy(col_name)
    return df.withColumn(out_name, F.count(F.lit(1)).over(w))


def collapse_rare_stands(df: DataFrame, threshold: float = RARE_STAND_THRESHOLD) -> DataFrame:
    """
    Step 4c - any ORIGIN_STAND used by less than `threshold` share of all trips is
    merged into "Other_Stand", so encoding doesn't carry one near-unique level per
    rarely-used stand. Must run on ORIGIN_STAND_filled (i.e. after step 1b).
    """
    total = df.count()
    w = Window.partitionBy("ORIGIN_STAND_filled")
    df = df.withColumn("_stand_share", F.count(F.lit(1)).over(w) / F.lit(total))
    df = df.withColumn(
        "ORIGIN_STAND_grouped",
        F.when(F.col("_stand_share") < threshold, F.lit("Other_Stand"))
         .otherwise(F.col("ORIGIN_STAND_filled"))
    )
    return df.drop("_stand_share")


# ---------------------------------------------------------------------------
# Step 5 - DAY_TYPE class imbalance
# ---------------------------------------------------------------------------

def undersample_majority(df: DataFrame, label_col: str = "DAY_TYPE", seed: int = 42) -> DataFrame:
    """
    Step 5a (option 1) - random-undersample every class down to the size of the
    smallest one, via `sampleBy` (a single distributed Bernoulli-sampling pass,
    no shuffle / no collect). Use this if DAY_TYPE becomes a model TARGET and you
    want a class-balanced training set, at the cost of throwing away majority rows.
    """
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    if len(counts) < 2:
        return df
    minority_n = min(counts.values())
    fractions = {label: (minority_n / n if n > 0 else 0.0) for label, n in counts.items()}
    return df.sampleBy(label_col, fractions=fractions, seed=seed)


def compute_class_weights(df: DataFrame, label_col: str = "DAY_TYPE") -> dict:
    """
    Step 5b (option 2, data-preserving alternative to undersampling) - per-class
    weight = total / (n_classes * class_count), the standard "balanced" weighting
    used e.g. by sklearn/MLlib weighted losses. Keeps every row; just down-weights
    the majority class's contribution to the loss instead of deleting rows.
    """
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    total, n_classes = sum(counts.values()), len(counts)
    return {label: total / (n_classes * n) for label, n in counts.items()}


def add_class_weight_column(df: DataFrame, weights: dict, label_col: str = "DAY_TYPE") -> DataFrame:
    """Materializes the class-weight dict as a per-row column, e.g. for MLlib's weightCol=...."""
    mapping = F.create_map(*[F.lit(x) for pair in weights.items() for x in pair])
    return df.withColumn("class_weight", mapping.getItem(F.col(label_col)))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(spark: SparkSession, input_path: str) -> DataFrame:
    df = load_and_parse(spark, input_path)
    df = filter_gps_errors(df)
    df = add_trip_geometry(df)
    df = fill_structural_missing(df)

    df, thresholds = winsorize(df, ["distance_km", "duration_sec", "avg_kmh"])
    print("winsorizing thresholds (p95):", thresholds)

    df = add_skew_transforms(df, ["distance_km", "duration_sec"])

    df = one_hot_call_type(df)
    df = frequency_encode(df, "TAXI_ID")
    df = collapse_rare_stands(df)
    df = frequency_encode(df, "ORIGIN_STAND_grouped", out_name="ORIGIN_STAND_freq")

    # DAY_TYPE imbalance: class-weight column is data-preserving, so it's the default
    # wired into the pipeline. `undersample_majority` is left available to call
    # separately if DAY_TYPE is actually used as a supervised target.
    weights = compute_class_weights(df, "DAY_TYPE")
    print("DAY_TYPE class weights:", weights)
    df = add_class_weight_column(df, weights, "DAY_TYPE")

    return df


if __name__ == "__main__":
    spark = (SparkSession.builder.appName("porto-taxi-advanced-prep")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    INPUT_PATH = "sample_train.csv"           # swap for gs://... on DataProc
    OUTPUT_PATH = "out/trips_advanced_prep_sample.parquet"

    result = run_pipeline(spark, INPUT_PATH)

    result.select(
        "TRIP_ID", "CALL_TYPE", "CALL_TYPE_A", "CALL_TYPE_B", "CALL_TYPE_C",
        "ORIGIN_CALL_filled", "ORIGIN_STAND_grouped", "ORIGIN_STAND_freq", "TAXI_ID_freq",
        "distance_km", "distance_km_wins", "distance_km_log",
        "duration_sec", "duration_sec_wins", "duration_sec_log",
        "DAY_TYPE", "class_weight",
    ).show(10, truncate=False)

    result.write.mode("overwrite").parquet(OUTPUT_PATH)
    print("saved:", OUTPUT_PATH, " rows:", result.count())

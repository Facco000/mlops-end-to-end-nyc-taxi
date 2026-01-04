from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feature_repo.data.taxi_trip_source import taxi_trip_source

# ────────────────── Entity ──────────────────
trip = Entity(
    name="trip_id",
    join_keys=["trip_id"],
    description="Unique trip identifier"
)

# ────────────────── Feature View ──────────────────
taxi_trip_features = FeatureView(
    name="taxi_trip_features",
    entities=[trip],
    ttl=None,
    schema=[
        Field(name="haversine_m", dtype=Float32),
        Field(name="trip_distance", dtype=Float32),
        Field(name="fare_amount", dtype=Float32),
        Field(name="passenger_count", dtype=Int64),
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="pickup_latitude", dtype=Float32),
        Field(name="pickup_longitude", dtype=Float32),
        Field(name="dropoff_latitude", dtype=Float32),
        Field(name="dropoff_longitude", dtype=Float32),
    ],
    source=taxi_trip_source,
    online=True,
)

# Ideally, you should run cd feature_repo/taxi_features && feast apply 
# whenever you modify feature_definitions.py.
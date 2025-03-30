import numpy as np
import pandas as pd

def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = (
        data
        .rename(columns=str.lower)
        .rename(columns={
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"
        })
    )

    return data


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    minors_data = data[data["age"] < '18']
    six_star_data = data[data["ratings"] == '6']
    data = (
        data
        .drop(columns='id')
        .drop(index=minors_data.index)
        .drop(index=six_star_data.index)
        .replace('NaN ', np.nan)
        .assign(
            # city column from rider id
            city_name = lambda x: x['rider_id'].str.split('RES').str.get(0).str.strip(),
            # change data type of age
            age = lambda x: x['age'].astype('float'),
            # change data type of ratings
            ratings = lambda x: x['ratings'].astype('float'),
            # absolute value in location based columns
            restaurant_latitude = lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude = lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude = lambda x: x['delivery_latitude'].abs(),
            delivery_longitude = lambda x: x['delivery_longitude'].abs(),
            # date based columns
            order_date = lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
            order_day = lambda x: x['order_date'].dt.day,
            order_month = lambda x: x['order_date'].dt.month,
            order_day_of_week = lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend = lambda x: x['order_day_of_week'].isin(['saturday', 'sunday']).astype('int'),
            # time based columns
            order_time = lambda x: pd.to_datetime(x['order_time'], dayfirst=True),
            order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'], dayfirst=True),
            pickup_time_minutes = lambda x: (x['order_picked_time']-x['order_time']).dt.seconds/60,
            order_time_hour = lambda x: x['order_time'].dt.hour,
            order_time_of_day = lambda x: (x['order_time_hour'].pipe(time_of_day)),
            # categorical columns
            weather = lambda x: (x['weather'].str.replace('conditions ', '').str.lower().replace('nan', np.nan)),
            traffic = lambda x: (x['traffic'].str.strip().str.lower()),
            type_of_order = lambda x: x['type_of_order'].str.strip().str.lower(),
            type_of_vehicle = lambda x: x['type_of_vehicle'].str.strip().str.lower(),
            festival = lambda x: x['festival'].str.strip().str.lower(),
            city_type = lambda x: x['city_type'].str.strip().str.lower(),
            multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),
            time_taken = lambda x: x['time_taken'].str.replace("(min) ", "").astype('int')
        )
        .drop(columns=["order_time","order_picked_time"])
    )

    return data


def clean_lot_long(data: pd.DataFrame) -> pd.DataFrame:
    location_columns = ['restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude']
    data = (
        data
        .assign(**{
            col: (np.where(data[col] < 1, np.nan, data[col].values))
            for col in location_columns
        })
    )

    return data


def extract_date_time_features(ser: pd.Series) -> pd.DataFrame:
    date_col = pd.to_datetime(ser, dayfirst=True)
    return (
        pd.DataFrame({
            "day": date_col.dt.day,
            "month": date_col.dt.month,
            "year": date_col.dt.year,
            "day_of_week": date_col.dt.day_name().str.lower(),
            "is_weekend": date_col.dt.day_name().str.lower().isin(['saturday', 'sunday']).astype('int'),
        })
    )


def time_of_day(ser: pd.Series):
    #hour = pd.to_datetime(ser, dayfirst=True).dt.hour

    return (
        pd.cut(ser, bins=[0, 6, 12, 17, 20, 24], right=True, labels=["after_midnight", "morning", "afternoon", "evening", "night"])
    )


def distance_type(data: pd.DataFrame) -> pd.DataFrame:
    data = (
        data
        .assign(
            distance_type = pd.cut(data['distance'], bins=[0,5,10,15,25], right=False, labels=["short", "medium", "long", "very_long"])
        )
    )

    return data

def calculate_haversine_distance(data: pd.DataFrame):
    loc_columns = ['restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude']
    lat1 = data[loc_columns[0]]
    lon1 = data[loc_columns[1]]
    lat2 = data[loc_columns[2]]
    lon2 = data[loc_columns[3]]

    
    # change into radian
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return(
        data
        .assign(
            distance = distance
        )
    )


def perform_data_cleaning(data: pd.DataFrame) -> None:
    cleaned_data = (
        data
        .pipe(rename_columns)
        .pipe(data_cleaning)
        .pipe(clean_lot_long)
        .pipe(calculate_haversine_distance)
        .pipe(distance_type)
    )

    cleaned_data.to_csv("../data/cleaned_swiggy.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("../data/swiggy.csv")
    perform_data_cleaning(df)
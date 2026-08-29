"""Feature Engineering Module for Freight Rate Prediction."""
import math
from datetime import datetime

FEATURE_NAMES = [
    # Basic signals
    'distance',
    'quote_signal',
    'market_index',
    'weight',
    
    # Interaction terms
    'dist_x_quote',
    'dist_x_market',
    'quote_x_market',
    'quote_div_market',
    'dist_x_quote_x_market',
    
    # Geographic & spatial
    'haversine_distance',
    'distance_ratio',
    'delta_lat',
    'delta_lon',
    'bearing_rad',
    'pickup_lat',
    'pickup_lon',
    'delivery_lat',
    'delivery_lon',
    
    # Equipment encodings & interactions
    'is_dry_van',
    'is_reefer',
    'is_flatbed',
    'eq_rate_boost',
    'dist_qs_reefer',
    'dist_qs_flatbed',
    
    # Weight & density
    'weight_per_mile',
    'weight_x_dist_k',
    'is_heavy_load',
    
    # Temporal & seasonality
    'month',
    'day_of_month',
    'day_of_week',
    'day_of_year',
    'is_weekend',
    'is_month_end',
    'sin_day_of_year',
    'cos_day_of_year',
    'sin_day_of_week',
    'cos_day_of_week',
    'q4_holiday_period',
    
    # Missingness indicators
    'weight_is_missing',
    'market_index_is_missing'
]

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in miles between two coordinate points."""
    r_miles = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(max(0.0, min(1.0, a))), math.sqrt(max(0.0, min(1.0, 1.0 - a))))
    return r_miles * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing angle in radians."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return math.atan2(y, x)

def extract_features(record):
    """Extract full feature dictionary and numeric vector from a single record."""
    dist = float(record['distance'])
    qs = float(record['quote_signal'])
    mi = float(record['market_index'])
    wt = float(record['weight'])
    eq = str(record.get('equipment', 'Dry Van'))
    
    p_lat = float(record['pickup_lat'])
    p_lon = float(record['pickup_lon'])
    d_lat = float(record['delivery_lat'])
    d_lon = float(record['delivery_lon'])
    
    # Date parsing
    date_str = record.get('date', '2025-01-01')
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
    except Exception:
        dt = datetime(2025, 1, 1)
        
    month = dt.month
    day_of_month = dt.day
    day_of_week = dt.weekday()
    day_of_year = dt.timetuple().tm_yday
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    is_month_end = 1.0 if day_of_month >= 27 else 0.0
    q4_holiday_period = 1.0 if month in [11, 12] else 0.0
    
    # Cyclical encoding
    sin_doy = math.sin(2.0 * math.pi * day_of_year / 365.25)
    cos_doy = math.cos(2.0 * math.pi * day_of_year / 365.25)
    sin_dow = math.sin(2.0 * math.pi * day_of_week / 7.0)
    cos_dow = math.cos(2.0 * math.pi * day_of_week / 7.0)
    
    # Geographic
    hav_dist = calculate_haversine(p_lat, p_lon, d_lat, d_lon)
    dist_ratio = dist / (hav_dist + 1.0)
    delta_lat = d_lat - p_lat
    delta_lon = d_lon - p_lon
    bearing = calculate_bearing(p_lat, p_lon, d_lat, d_lon)
    
    # Equipment
    is_reefer = 1.0 if eq == 'Reefer' else 0.0
    is_flatbed = 1.0 if eq == 'Flatbed' else 0.0
    is_dry_van = 1.0 if eq == 'Dry Van' else 0.0
    eq_rate_boost = 1.14 if is_reefer else (1.08 if is_flatbed else 1.0)
    
    # Interactions
    dist_x_qs = dist * qs
    dist_x_mi = dist * mi
    qs_x_mi = qs * mi
    qs_div_mi = qs / (mi + 1e-4)
    dist_x_qs_x_mi = dist * qs * mi
    
    dist_qs_reefer = dist_x_qs * is_reefer
    dist_qs_flatbed = dist_x_qs * is_flatbed
    
    # Weight
    wt_per_mile = wt / (dist + 1.0)
    wt_x_dist_k = (wt * dist) / 1000.0
    is_heavy = 1.0 if wt > 38000.0 else 0.0
    
    feat_dict = {
        'distance': dist,
        'quote_signal': qs,
        'market_index': mi,
        'weight': wt,
        'dist_x_quote': dist_x_qs,
        'dist_x_market': dist_x_mi,
        'quote_x_market': qs_x_mi,
        'quote_div_market': qs_div_mi,
        'dist_x_quote_x_market': dist_x_qs_x_mi,
        'haversine_distance': hav_dist,
        'distance_ratio': dist_ratio,
        'delta_lat': delta_lat,
        'delta_lon': delta_lon,
        'bearing_rad': bearing,
        'pickup_lat': p_lat,
        'pickup_lon': p_lon,
        'delivery_lat': d_lat,
        'delivery_lon': d_lon,
        'is_dry_van': is_dry_van,
        'is_reefer': is_reefer,
        'is_flatbed': is_flatbed,
        'eq_rate_boost': eq_rate_boost,
        'dist_qs_reefer': dist_qs_reefer,
        'dist_qs_flatbed': dist_qs_flatbed,
        'weight_per_mile': wt_per_mile,
        'weight_x_dist_k': wt_x_dist_k,
        'is_heavy_load': is_heavy,
        'month': float(month),
        'day_of_month': float(day_of_month),
        'day_of_week': float(day_of_week),
        'day_of_year': float(day_of_year),
        'is_weekend': is_weekend,
        'is_month_end': is_month_end,
        'sin_day_of_year': sin_doy,
        'cos_day_of_year': cos_doy,
        'sin_day_of_week': sin_dow,
        'cos_day_of_week': cos_dow,
        'q4_holiday_period': q4_holiday_period,
        'weight_is_missing': float(record.get('weight_is_missing', 0.0)),
        'market_index_is_missing': float(record.get('market_index_is_missing', 0.0))
    }
    
    vec = [feat_dict[name] for name in FEATURE_NAMES]
    return feat_dict, vec

def transform_records(records):
    """Transform a list of cleaned records into feature matrix X and target y."""
    X = []
    y = []
    has_target = 'posted_rate' in records[0]
    for r in records:
        _, vec = extract_features(r)
        X.append(vec)
        if has_target:
            y.append(float(r['posted_rate']))
    return X, (y if has_target else None)

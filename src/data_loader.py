"""Data loading, cleaning, and preprocessing module."""
import csv
import os
from collections import defaultdict

# Pre-computed city coordinates lookup for fast mapping
CITY_COORDINATES = {
    'Lexington': (36.99152, -84.99876),
    'Fort Wayne': (41.31561, -85.36206),
}

# Equipment median weights from training data
EQUIPMENT_MEDIAN_WEIGHTS = {
    'Dry Van': 30000.0,
    'Reefer': 32000.0,
    'Flatbed': 34000.0,
}

GLOBAL_MEDIAN_WEIGHT = 31000.0
GLOBAL_MEAN_MARKET_INDEX = 1.002
GLOBAL_MEAN_QUOTE_SIGNAL = 2.062

def load_csv(filepath):
    """Load a CSV file into a list of dictionaries."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_csv(filepath, rows, fieldnames):
    """Save a list of dictionaries to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_city_coordinate_map(train_rows, val_rows=None):
    """Build coordinate lookup mapping from city names to (lat, lon)."""
    coords = dict(CITY_COORDINATES)
    all_rows = list(train_rows) + (list(val_rows) if val_rows else [])
    for r in all_rows:
        if r.get('pickup') and r.get('pickup_lat') and r.get('pickup_lon'):
            try:
                coords[r['pickup']] = (float(r['pickup_lat']), float(r['pickup_lon']))
            except (ValueError, TypeError):
                pass
        if r.get('delivery') and r.get('delivery_lat') and r.get('delivery_lon'):
            try:
                coords[r['delivery']] = (float(r['delivery_lat']), float(r['delivery_lon']))
            except (ValueError, TypeError):
                pass
    return coords

def build_market_index_lookup(train_rows, val_rows=None):
    """Build date to average market_index lookup map."""
    date_mi = defaultdict(list)
    all_rows = list(train_rows) + (list(val_rows) if val_rows else [])
    for r in all_rows:
        d = r.get('date')
        mi = r.get('market_index')
        if d and mi:
            try:
                date_mi[d].append(float(mi))
            except (ValueError, TypeError):
                pass
    
    lookup = {}
    for d, vals in date_mi.items():
        lookup[d] = sum(vals) / len(vals)
    return lookup

def build_quote_signal_lookup(train_rows, val_rows=None):
    """Build date to average quote_signal lookup map."""
    date_qs = defaultdict(list)
    all_rows = list(train_rows) + (list(val_rows) if val_rows else [])
    for r in all_rows:
        d = r.get('date')
        qs = r.get('quote_signal')
        if d and qs:
            try:
                date_qs[d].append(float(qs))
            except (ValueError, TypeError):
                pass
    
    lookup = {}
    for d, vals in date_qs.items():
        lookup[d] = sum(vals) / len(vals)
    return lookup

def clean_record(record, coord_map, mi_lookup, qs_lookup):
    """Clean and impute a single freight load record."""
    cleaned = dict(record)
    
    # 1. Coordinate imputation
    p_city = cleaned.get('pickup')
    d_city = cleaned.get('delivery')
    
    if not cleaned.get('pickup_lat') or not cleaned.get('pickup_lon'):
        if p_city in coord_map:
            cleaned['pickup_lat'], cleaned['pickup_lon'] = coord_map[p_city]
        else:
            cleaned['pickup_lat'], cleaned['pickup_lon'] = 39.5, -98.35  # US geographical center
            
    if not cleaned.get('delivery_lat') or not cleaned.get('delivery_lon'):
        if d_city in coord_map:
            cleaned['delivery_lat'], cleaned['delivery_lon'] = coord_map[d_city]
        else:
            cleaned['delivery_lat'], cleaned['delivery_lon'] = 39.5, -98.35
            
    cleaned['pickup_lat'] = float(cleaned['pickup_lat'])
    cleaned['pickup_lon'] = float(cleaned['pickup_lon'])
    cleaned['delivery_lat'] = float(cleaned['delivery_lat'])
    cleaned['delivery_lon'] = float(cleaned['delivery_lon'])
    
    # 2. Distance
    cleaned['distance'] = float(cleaned['distance']) if cleaned.get('distance') else 350.0
    
    # 3. Equipment
    cleaned['equipment'] = cleaned.get('equipment', 'Dry Van') or 'Dry Van'
    
    # 4. Weight Imputation
    wt = cleaned.get('weight')
    if wt is not None and wt != '':
        cleaned['weight'] = float(wt)
        cleaned['weight_is_missing'] = 0.0
    else:
        cleaned['weight'] = EQUIPMENT_MEDIAN_WEIGHTS.get(cleaned['equipment'], GLOBAL_MEDIAN_WEIGHT)
        cleaned['weight_is_missing'] = 1.0
        
    # 5. Market Index Imputation
    mi = cleaned.get('market_index')
    d = cleaned.get('date', '')
    if mi is not None and mi != '':
        cleaned['market_index'] = float(mi)
        cleaned['market_index_is_missing'] = 0.0
    else:
        cleaned['market_index'] = mi_lookup.get(d, GLOBAL_MEAN_MARKET_INDEX)
        cleaned['market_index_is_missing'] = 1.0
        
    # 6. Quote Signal Imputation
    qs = cleaned.get('quote_signal')
    if qs is not None and qs != '':
        cleaned['quote_signal'] = float(qs)
        cleaned['quote_signal_is_missing'] = 0.0
    else:
        cleaned['quote_signal'] = qs_lookup.get(d, GLOBAL_MEAN_QUOTE_SIGNAL)
        cleaned['quote_signal_is_missing'] = 1.0
        
    # 7. Target (if available)
    if 'posted_rate' in cleaned and cleaned['posted_rate'] != '':
        cleaned['posted_rate'] = float(cleaned['posted_rate'])
        
    return cleaned

def preprocess_dataset(records, coord_map, mi_lookup, qs_lookup):
    """Preprocess a list of raw records."""
    return [clean_record(r, coord_map, mi_lookup, qs_lookup) for r in records]

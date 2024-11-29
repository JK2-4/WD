# Find the nearest weather station for each estate
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r  
    
def find_nearest_station(lat, lon, stations):
    distances = {station['city'] : haversine(lon, lat, station['coordinates'][1], station['coordinates'][0]) for station in stations}
    # Find the station with the smallest distance
    station = min(distances, key=distances.get)
    distance = round(distances[station],2)
     
    return station, distance

'''
implement 


# Find the nearest weather station for each estate, and store in two columns
df['station', 'distance'] = df.apply(lambda x: find_nearest_station(x['latitude'], x['longitude'], LOCATIONS), axis = 1)
# Unwrap as separate columns
df[['station', 'distance']] = pd.DataFrame(df['station', 'distance'].tolist(), index=df.index)

'''

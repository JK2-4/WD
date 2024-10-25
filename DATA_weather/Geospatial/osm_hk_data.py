'''
Osmosis data

21 DISTRICTS; 

NORTH 
CENTRAL_AND_WESTERN
EASTERN
HONG_KONG
ISLANDS
KOWLOON
KOWLOON_CITY
KWAI_TSING
KWUN_TONG
NEW_TERRITORIES
SAI_KUNG
SHAM_SHUI_PO
SHA_TIN
SOUTHERN
TAI_PO
TSUEN_WAN
TUEN_MUN
WAN_CHAI
WONG_TAI_SIN
YAU_TSIM_MONG
YUEN_LONG
OVERALL

'''

!wget https://download.geofabrik.de/asia/china/hong-kong-latest.osm.pbf -O /content/drive/MyDrive/wd/osm5_data/hong-kong-latest5.osm.pbf

!cd /content/drive/MyDrive/wd/osmosis-0.49.2
!chmod +x ./bin/osmosis #ensure executable

import sys
sys.path.append('/content/drive/MyDrive/wd/osmosis-0.49.2') # ensure path 


################# extracting data 

'''
#logic

!./bin/osmosis \
  --read-pbf file=/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/osm_bpf/china-latest.osm.pbf \
  --bounding-polygon file=/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/poly2/OVERALL.poly \
  --write-pbf file=/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf/OVERALL.osm.pbf

'''

import os
import subprocess
from pathlib import Path

# Define your working directory (adjust as needed)
wd = Path('/content/drive/MyDrive/wd/osmosis-0.49.2')
poly_dir = Path('/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/poly2')
output_dir = Path('/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf')

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Path to the osmosis executable
osmosis_executable = "/content/drive/MyDrive/wd/osmosis-0.49.2/bin/osmosis"

# Loop through all .poly files in the poly directory and run Osmosis for each one
for poly_file in poly_dir.glob('*.poly'):
    district_name = poly_file.stem
    output_file = output_dir / f"{district_name}.osm.pbf"

    path_chn_dump = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/osm_bpf/china-latest.osm.pbf'

    # Construct the Osmosis command using the direct path to the executable
    command = [
        osmosis_executable,  # Path to the Osmosis executable
        '--read-pbf', str(path_chn_dump),
        '--bounding-polygon', f"file={poly_file}",
        '--write-pbf', str(output_file)
    ]

    print(f"Processing district: {district_name}")
    try:
        subprocess.run(command, check=True)
        print(f"Successfully created {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {district_name}: {e}")

  
################## convert 

#logic example

!apt-get install osmium-tool
!osmium cat /content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf/NORTH.osm.pbf -o /content/NORTH.osm

##################### convert and save data 

#!pip install osmium pandas
import osmium
import pandas as pd
from pathlib import Path

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.data = []

    def node(self, n):
        tags = n.tags
        relevant_tags = [
            'landuse', 'natural', 'building', 'highway', 'railway', 'public_transport',
            'building:levels', 'building:material', 'height', 'waterway'
        ]
        for tag in relevant_tags:
            if tag in tags:
                self.data.append({
                    'type': 'node',
                    'id': n.id,
                    'lat': n.location.lat,
                    'lon': n.location.lon,
                    'tags': {tag: tags.get(tag)}
                })
                break

    def way(self, w):
        tags = w.tags
        relevant_tags = [
            'landuse', 'natural', 'building', 'highway', 'railway', 'public_transport',
            'building:levels', 'building:material', 'height', 'waterway'
        ]
        for tag in relevant_tags:
            if tag in tags:
                self.data.append({
                    'type': 'way',
                    'id': w.id,
                    'nodes': [n.ref for n in w.nodes],
                    'tags': {tag: tags.get(tag)}
                })
                break

def extract_data(osm_file_path):
    handler = OSMHandler()
    handler.apply_file(osm_file_path)

    df = pd.DataFrame(handler.data)
    return df

def main():
    # List of all districts except NORTH
    districts = [
        "CENTRAL_AND_WESTERN", "EASTERN", "HONG_KONG", "ISLANDS", "KOWLOON",
        "KOWLOON_CITY", "KWAI_TSING", "KWUN_TONG", "NEW_TERRITORIES", "SAI_KUNG",
        "SHAM_SHUI_PO", "SHA_TIN", "SOUTHERN", "TAI_PO", "TSUEN_WAN", "TUEN_MUN",
        "WAN_CHAI", "WONG_TAI_SIN", "YAU_TSIM_MONG", "YUEN_LONG", "OVERALL"
    ]

    # Path to the directory containing PBF files
    pbf_dir = Path("/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf")

    for district in districts:
        osm_file_path = pbf_dir / f"{district}.osm.pbf"
        df = extract_data(osm_file_path)

        # Save extracted data to CSV for further analysis
        output_csv = f"/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/csvs/uhi_risk_data_{district.lower()}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    main()

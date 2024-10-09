import requests
from bs4 import BeautifulSoup
import os
import csv
import re

# ==============================
# Configuration
# ==============================

MAIN_URL = 'https://www.hko.gov.hk/en/abouthko/opendata_intro.htm'  # Main Open Data Page
OUTPUT_FILE = 'ddownload_links.csv'  # CSV file to save links
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}  # Mimic browser
BASE_URL = 'https://data.gov.hk'

# ==============================
#  Functions
# ==============================

def sanitize_filename(name):
    """Sanitizes a string to be used as a directory or file name."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def get_soup(url):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred while fetching {url}: {http_err}")
    except Exception as err:
        print(f"An error occurred while fetching {url}: {err}")
    return None

def get_dataset_links(main_url):
    """Scrapes the main Open Data page to extract dataset names and URLs."""
    soup = get_soup(main_url)
    if not soup:
        return []
    
    dataset_links = []
    # Find all divs with class 'mobile_content_item' (structure containing dataset links)
    dataset_items = soup.find_all('div', class_='mobile_content_item')
    
    print(f"Found {len(dataset_items)} datasets.")
    
    for item in dataset_items:
        title_div = item.find('div', class_='mobile_content_item_title')
        if title_div:
            link_tag = title_div.find('a', href=True)
            if link_tag:
                dataset_url = link_tag['href']
                if not dataset_url.startswith('http'):
                    dataset_url = BASE_URL + dataset_url
                dataset_name = link_tag.get_text(strip=True)
                dataset_links.append({'name': dataset_name, 'url': dataset_url})
    
    return dataset_links

def get_download_links(dataset_page_url):
    """Scrapes a dataset page to extract all CSV download links."""
    soup = get_soup(dataset_page_url)
    if not soup:
        return []
    
    download_links = []
    
    # Find all <a> tags with href attributes
    link_tags = soup.find_all('a', href=True)
    for link in link_tags:
        href = link['href']
        link_text = link.get_text(strip=True).lower()
        
        # Filter for CSV download links
        if 'download' in link_text or href.endswith('.csv'):
            if not href.startswith('http'):
                href = BASE_URL + href  # Ensure full URL
            download_links.append(href)
    
    print(f"Found {len(download_links)} CSV download links on {dataset_page_url}.")
    return download_links

def save_links_to_csv(links, output_file):
    """Saves a list of links to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset Name', 'Download Link'])  # Write header
        for dataset_name, link in links:
            writer.writerow([dataset_name, link])
    print(f"Download links saved to {output_file}")

# ==============================
# Logic
# ==============================

# dataset links 
dataset_links = get_dataset_links(MAIN_URL)

if not dataset_links:
    print("No dataset links found. Exiting.")
    exit()

# Loop through each dataset page for CSV download links
all_download_links = []

for dataset in dataset_links:
    dataset_name = dataset['name']
    dataset_url = dataset['url']
    print(f"Processing dataset: {dataset_name} ({dataset_url})")
    
    # Get all CSV download links from the dataset page
    download_links = get_download_links(dataset_url)
    
    # Append the dataset name and its download links to the list
    for link in download_links:
        all_download_links.append((dataset_name, link))

# save
save_links_to_csv(all_download_links, OUTPUT_FILE)

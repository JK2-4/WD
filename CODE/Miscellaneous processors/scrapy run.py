#'''
#  Install required packages (Uncomment if running in a new environment)
!pip install selenium webdriver-manager
!apt-get update
!apt-get install -y wget unzip
!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
!dpkg -i google-chrome-stable_current_amd64.deb
!apt-get -fy install
#'''

import re
import os
import pandas as pd
import json
import csv
import time
import logging
from urllib.parse import urlparse, parse_qs
from multiprocessing import Pool, current_process, cpu_count

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    TimeoutException,
    StaleElementReferenceException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_unique_key(row):
    """
    Returns the 'identifier' if it's not empty; otherwise, returns the 'href'.
    """
    identifier = str(row['identifier']).strip()
    if identifier and identifier.lower() != 'nan':
        return identifier
    else:
        return row['href']

# logging (Each process will have its own log file)
def setup_logging(estate_name):
    logger = logging.getLogger(estate_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f'centa_taipo_scraping_{estate_name}.log')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def extract_detailcode(url):
    """Extract the 'detailcode' parameter from the given URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    detailcode = query_params.get('detailcode', [None])[0]
    return detailcode

def extract_chartdata_from_script(driver, logger):
    """
    Extracts the chart data from a script tag containing 'chartData' - no static sleep.
    """
    try:
        script_tag = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, '//script[contains(text(), "chartData")]'))
        )
        return script_tag.get_attribute('innerHTML').strip()
    except TimeoutException:
        logger.warning("Script tag containing 'chartData' not found within timeout.")
        return ""


def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument('--headless')
    options.add_argument('window-size=2560,1440')
    options.add_argument('--disable-gpu')  #  Windows OS
    options.add_argument('--no-sandbox')   #  Linux OS
    options.add_argument('--disable-dev-shm-usage') 

    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)

    return driver

import pandas as pd
import re
def process_estate(args):
    """
    Processes a single estate by reading links from an existing CSV file,
    extracting chart data from each link, and saving the data incrementally
    to a CSV file named '{estate_name}_extracted.csv'.

    Parameters:
    - args: Tuple containing (estate_name, input_csv_path, output_csv_path)
    """
    estate_name, input_csv_path, output_csv_path = args
    logger = setup_logging(estate_name)
    logger.info(f"Process {current_process().name} started for estate: {estate_name}")

    try:
        driver = setup_driver()
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver for estate '{estate_name}': {e}")
        return

    try:
        links_df = pd.read_csv(input_csv_path)
        logger.info(f"Successfully loaded links from: {input_csv_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found for estate '{estate_name}': {input_csv_path}")
        driver.quit()
        return
    except Exception as e:
        logger.error(f"Error reading CSV for estate '{estate_name}': {e}")
        driver.quit()
        return

    required_columns = {'identifier', 'href'}
    if not required_columns.issubset(links_df.columns):
        logger.error(f"CSV file for estate '{estate_name}' must contain columns: {required_columns}")
        driver.quit()
        return

    if links_df.empty:
        logger.warning(f"No links found in CSV for estate '{estate_name}'. Skipping processing.")
        driver.quit()
        return

    processed_keys = set()
    if os.path.exists(output_csv_path):
        try:
            existing_df = pd.read_csv(output_csv_path)
            existing_df['unique_key'] = existing_df.apply(lambda row: get_unique_key(row), axis=1)
            processed_keys = set(existing_df['unique_key'].astype(str))
            logger.info(f"Found {len(processed_keys)} already processed links in {output_csv_path}.")
        except Exception as e:
            logger.error(f"Error reading existing output CSV '{output_csv_path}': {e}")

            pass

    write_header = not os.path.exists(output_csv_path)

    try:
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['identifier', 'href', 'chart_data']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()

            for index, row in links_df.iterrows():
                unique_key = get_unique_key(row)
                identifier = str(row['identifier']).strip()
                href = row['href']

                if unique_key in processed_keys:
                    logger.info(f"Skipping already processed link: Identifier='{identifier}', Href='{href}'")
                    continue

                logger.info(f"Processing link {index + 1}/{len(links_df)}: Identifier='{identifier}', Href='{href}'")

                try:
                    driver.get(href)
                    wait = WebDriverWait(driver, 15)  

                    # Wait until the chart container is present ( CSS_SELECTOR ?)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.period-box-item.period-box-item-selected')))
                    logger.info(f"Chart container found for link: Href='{href}'")

                    # 'month' button
                    try:
                        month_button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.period-box-item.period-box-item-selected'))
                        )
                        month_button.click()
                        logger.info(f"Clicked 'month' button for link: Href='{href}'")
                        time.sleep(1)  # Wait for dynamic content to load
                    except TimeoutException:
                        logger.warning(f"'Month' button not found or not clickable for link: Href='{href}'")

                    # 'all' button
                    try:
                        all_button = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.interval-list > div:nth-child(5)'))
                        )
                        all_button.click()
                        logger.info(f"Clicked 'all' button for link: Href='{href}'")
                        time.sleep(1)  # Wait for dynamic content to load
                    except TimeoutException:
                        logger.warning(f"'All' button not found or not clickable for link: Href='{href}'")

                    time.sleep(4)
                    chart_data = extract_chartdata_from_script(driver, logger)

                    writer.writerow({
                        'identifier': identifier if identifier and identifier.lower() != 'nan' else '',
                        'href': href,
                        'chart_data': json.dumps(chart_data)  # Convert to JSON string if needed
                    })

                    logger.info(f"Successfully extracted and saved chart data for link: Href='{href}'")

                except Exception as e:
                    logger.error(f"Error processing link '{href}': {e}")
                    continue  

        logger.info(f"Data extraction complete. Output saved to: {output_csv_path}")
        print(f"Process {current_process().name}: Data extraction complete for estate '{estate_name}'.")

    except Exception as e:
        logger.error(f"Error opening/writing to CSV file '{output_csv_path}': {e}")

    finally:
        driver.quit()
        logger.info(f"WebDriver closed for estate: {estate_name}")

import os
import re

def extract_code_from_filename(filename):
    match = re.search(r'_(\w+)\.csv$', filename)
    return match.group(1) if match else None

def extract_estate_name(file_path):
    match = re.search(r'links_(.+)\.csv$', file_path)
    return match.group(1) if match else None

def process_estate_multi(file):
    nn = pd.read_csv(file)
    estate_name = extract_estate_name(file)
    input_csv_path = f'/content/drive/MyDrive/wd/centaline/jan1725links/links_{estate_name}.csv'
    output_csv_path = f'/content/drive/MyDrive/wd/centaline/jan1725_extracted/extracted_{estate_name}.csv'
    args = estate_name, input_csv_path, output_csv_path
    
    process_estate(args)  # process_estates fn from scrapy.py


#--------------------------------------------------------------------------
# FILTER FILES TO PROCESS

valid_df = pd.read_csv('/content/drive/MyDrive/wd/centaline/valid_df.csv')
valid_df['coded'] = valid_df['Link'].apply(lambda x: re.search(r'code=([^&]+)', x).group(1))

old_directory = '/content/drive/MyDrive/wd/centaline/jan1725'
link_dir = '/content/drive/MyDrive/wd/centaline/jan1725links'
extracted_dir_new = '/content/drive/MyDrive/wd/centaline/jan1725_extracted'

old_files = [f for f in os.listdir(old_directory) if f.endswith('.csv')] #855
old_codes = [extract_code_from_filename(file) for file in old_files]
done_files = [f for f in os.listdir(extracted_dir_new) if f.endswith('.csv')] #315
done_codes = [extract_code_from_filename(file) for file in done_files]
link_files = [f for f in os.listdir(link_dir) if f.endswith('.csv')] #3241
all_codes = [extract_code_from_filename(file) for file in link_files]

matched_files = [] # valid df vs all files
for file in link_files:
    file_code = extract_code_from_filename(file)
    if file_code in valid_df['coded'].values:
        matched_files.append(os.path.join(link_dir, file))
matched_codes = [extract_code_from_filename(file) for file in matched_files]

remaining_codes = [code for code in matched_codes if code not in done_codes] # 922
remaining_valid_df = valid_df[~valid_df['coded'].isin(done_codes)]

remaining_files = []
for file in link_files:
    file_code = extract_code_from_filename(file)
    if file_code in remaining_valid_df['coded'].values:
        remaining_files.append(os.path.join(link_dir, file))

#------------------------------
# multiprocessing Pool
num_processes = min(6, len(remaining_files))
with Pool(processes=num_processes) as pool:
    pool.map(process_estate_multi, remaining_files)

logging.info("Web scraping completed for all estates.")
print("Web scraping completed for all estates.")
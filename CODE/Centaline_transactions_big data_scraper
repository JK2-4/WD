# Install required packages
#!pip install selenium webdriver-manager
#!apt-get update
#!apt-get install -y wget unzip
#!wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
#!dpkg -i google-chrome-stable_current_amd64.deb
#!apt-get -fy install

import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd
from urllib.parse import parse_qs
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    TimeoutException,
    StaleElementReferenceException,
)
import logging
from urllib.parse import urlparse
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure logging
logging.basicConfig(
    filename='centa_taipo_scraping.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def is_valid_url(url):
    """Check if the URL is valid."""
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def extract_detailcode(url):
    """Extract the 'detailcode' parameter from the given URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    detailcode = query_params.get('detailcode', [None])[0]
    return detailcode

def extract_chartdata_from_script(driver):
    """
    Extracts the raw chart data from the third script tag in the body.
    """
    try:
        # Locate the third script tag in the body
        script_tag = driver.find_element(By.XPATH, '//script[contains(text(), "chartData")]')
        script_content = script_tag.get_attribute('innerHTML').strip()
        return script_content
    except NoSuchElementException:
        logging.warning("Script tag '/html/body/script[3]' not found.")
        return ""

def setup_driver():
    """Sets up the Selenium WebDriver with appropriate options."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    # Uncomment the next line to run Chrome in headless mode
    options.add_argument('--headless')
    options.add_argument('window-size=2560,1440')
    options.add_argument('--disable-gpu')  # Applicable to Windows OS
    options.add_argument('--no-sandbox')   # Applicable to Linux OS
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems

    # Initialize the WebDriver using webdriver-manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    return driver

def main():
    # Initialize the WebDriver
    driver = setup_driver()
    wait = WebDriverWait(driver, 10)  # 10 seconds timeout

    # Define the path to your CSV file
    csv_path = '/content/drive/MyDrive/wd/centaline/centa_taipo_output.csv'  # Update this path if necessary

    # Read the existing CSV into a DataFrame
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded CSV from {csv_path}.")
        print(f"Successfully loaded CSV from {csv_path}.")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        print(f"Error reading CSV file: {e}")
        driver.quit()
        return

    # Ensure that the 4th column contains the links
    link_column_index = 3  # 4th column
    if df.shape[1] <= link_column_index:
        logging.error(f"CSV does not have a 4th column for links.")
        print(f"CSV does not have a 4th column for links.")
        driver.quit()
        return

    # Initialize a new column for ChartData
    df['ChartData'] = ""

    output_json_path = '/content/drive/MyDrive/wd/centaline/centa_final.json'

    try:
        # Iterate through each row to process the links
        for index, row in df.iterrows():         # df.iloc[1092:] if interrupted 
            try:
                estate_link = row.iloc[link_column_index]
                if not is_valid_url(estate_link):
                    logging.warning(f"Row {index}: Invalid URL '{estate_link}'. Skipping.")
                    continue

                # Extract 'detailcode' from the URL
                detailcode = extract_detailcode(estate_link)
                if not detailcode:
                    logging.warning(f"Row {index}: 'detailcode' not found in URL '{estate_link}'. Skipping.")
                    continue

                logging.info(f"Row {index}: Extracted detailcode '{detailcode}'.")

                # Construct the chart link
                chart_link = f"https://hkdata.centanet.com/BigData/Chart/Index?code={detailcode}&type=building&pv=edu_ter"

                # Navigate to the chart link
                driver.get(chart_link)
                # Allow the page to load
                wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[1]')))
                logging.info(f"Row {index}: Navigated to chart link '{chart_link}'.")

                # Click the 'month' button
                try:
                    month_button = wait.until(
                        EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[2]/div[2]/div[1]'))
                    )
                    month_button.click()
                    logging.info(f"Row {index}: Clicked 'month' button.")
                except TimeoutException:
                    logging.warning(f"Row {index}: 'month' button not found or not clickable. Skipping.")
                    continue

                # Click the 'all' button
                try:
                    all_button = wait.until(
                        EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[3]/div[2]/div[5]'))
                    )
                    all_button.click()
                    logging.info(f"Row {index}: Clicked 'all' button.")
                except TimeoutException:
                    logging.warning(f"Row {index}: 'all' button not found or not clickable. Skipping.")
                    continue

                # Allow some time for the chart to update
                time.sleep(10)

                # Extract chart data from the script tag
                chart_data = extract_chartdata_from_script(driver)
                if chart_data:
                    df.at[index, 'ChartData'] = chart_data
                    logging.info(f"Row {index}: Extracted chart data.")
                else:
                    logging.warning(f"Row {index}: No chart data extracted.")

                # Save the DataFrame as JSON after processing each row
                df.iloc[[index]].to_json(output_json_path, orient='records', lines=True, force_ascii=False, mode='a')
                logging.info(f"Row {index}: Updated JSON saved after extracting chart data for this row.")

            except Exception as e:
                logging.error(f"Row {index}: Unexpected error: {e}")
                continue

    finally:
        # Close the WebDriver
        driver.quit()

if __name__ == "__main__":
    main()




# CSV - Save the DataFrame after processing all rows
#output_csv_path = '/content/drive/MyDrive/wd/centaline/centa_final.csv'
#df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

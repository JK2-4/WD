#!pip install beautifulsoup4
import csv
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    filename='html_parsing.log',
    filemode='w',  # Overwrite the log file each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_html_to_csv(html_file_path, csv_file_path):
    """
    Parses the given HTML file to extract 'field' and 'item-name' from each 'div.group-item'
    and writes the data into a CSV file.
    
    :param html_file_path: Path to the input HTML file.
    :param csv_file_path: Path to the output CSV file.
    """
    try:
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
        logging.info(f"Successfully loaded HTML file: {html_file_path}")
    except Exception as e:
        logging.error(f"Error reading HTML file: {e}")
        return

    # Find all divs with class 'group-item'
    group_items = soup.find_all('div', class_='group-item')
    logging.info(f"Found {len(group_items)} 'div.group-item' elements.")

    # Prepare data for CSV
    data_rows = []
    for idx, div in enumerate(group_items, start=1):
        # Extract the 'field' attribute
        field_value = div.get('field', '').strip()
        if not field_value:
            logging.warning(f"Div {idx}: Missing 'field' attribute.")
        
        # Extract the text from 'span.item-name'
        item_name_span = div.find('span', class_='item-name')
        item_name = item_name_span.get_text(strip=True) if item_name_span else ''
        if not item_name:
            logging.warning(f"Div {idx}: Missing 'span.item-name' text.")
        
        # Append the extracted data to the list
        data_rows.append({
            'Field': field_value,
            'Item Name': item_name
        })
        logging.info(f"Div {idx}: Extracted Field='{field_value}', Item Name='{item_name}'.")

    # Write data to CSV
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Field', 'Item Name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data_rows:
                writer.writerow(row)
        logging.info(f"Successfully wrote data to CSV file: {csv_file_path}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    # Specify the paths to your HTML input file and desired CSV output file
    html_input_path = 'CW0145_source.html'        # Replace with your actual HTML file path
    csv_output_path = 'htmloutput.csv'       # Desired name for the output CSV file

    parse_html_to_csv(html_input_path, csv_output_path)

    print(f"Data extraction complete. Check '{csv_output_path}' for results and 'html_parsing.log' for logs.")

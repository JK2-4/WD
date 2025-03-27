library(jsonlite)
library(stringr)
library(readr)
library(stringr)

extract_code_from_href <- function(link) {
  match <- regmatches(link, regexpr("code=([^&]+)", link))
  if (length(match) > 0) {
    return(sub("code=", "", match))
  } else {
    return(NA)
  }
}

extract_estate_name <- function(filename) {
  estate <- sub("_extracted\\.csv$", "", basename(filename))
  return(estate)
}

#-------------- try : t<- process_csv(csv_files[1])

process_csv <- function(csv_file) {
  print(paste("\nProcessing file:", csv_file))
  
  # Try reading the CSV file
  data <- tryCatch(read_csv(csv_file), error = function(e) {
    print(paste("Error reading", csv_file, ":", e))
    return(data.frame())
  })
  
  # Skip if the file is empty
  if (nrow(data) == 0) {
    print(paste("Skipping empty file:", csv_file))
    return(data.frame())
  }
  
  # Check for required columns
  required_cols <- c("Field", "chart_data", "href")
  if (!all(required_cols %in% colnames(data))) {
    print(paste("Skipping file", csv_file, ": Required column(s) not found."))
    return(data.frame())
  }
  
  # Extract file-level identifiers
  href_series <- na.omit(data$href)
  file_code <- if (length(href_series) > 0) extract_code_from_href(href_series[1]) else NA
  estate_name <- extract_estate_name(csv_file)
  
  if (file_code %in% existing_file_codes) {
    print(paste("Skipping file", csv_file, ": FileCode already exists."))
    return(data.frame())
  }
  
  extracted_rows <- list()
  
  # Process each row in the CSV
  for (i in 1:nrow(data)) {
    row <- data[i, ]
    field_value <- row$Field
    param_key <- field_value
    js_text <- row$chart_data
    
    # Clean the JSON data
    js_text <- gsub('\\"', '"', js_text)
    js_text <- gsub("\\\\", "", js_text)  # Remove backslashes

    # Extract the JSON array using regex
    match <- str_match(js_text, "chartData\\s*:\\s*(\\[\\{.*?\\}\\])\\s*(?:,|;)")
    if (!is.na(match[2])) {
      chart_data_json_str <- match[2]
      chart_data <- tryCatch(fromJSON(chart_data_json_str), error = function(e) {
        print(paste("Error decoding JSON in file", csv_file, "row", i, ":", e))
        return(NULL)
      })
      if (!is.null(chart_data)) {
        for (j in 1:nrow(chart_data)) {
          json_row  <- chart_data[j, ]
          
          extracted_rows <- append(extracted_rows, list(data.frame(
            FileCode = file_code,
            EstateName = estate_name,
            Field = field_value,
            ParamValue = json_row[[param_key]],
            Type = json_row$Type,
            DateValue = json_row$DateValue,
            Period = json_row$Period
          )))
        }
      }
    } else {
      print(paste("No matching chartData pattern found in file", csv_file, "row", i))
    }
  }
  
  return(do.call(rbind, extracted_rows))
}

#------------------- get zips

zip_directory <- "C:\\Users\\jkwatra\\Downloads\\zipped"
zip_files <- list.files(zip_directory, pattern = "\\.zip$", full.names = TRUE)
temp_dir <- tempdir()

collect_csv_files <- function(zip_file, temp_dir) {
  unzip(zip_file, exdir = temp_dir)
  csv_files <- list.files(temp_dir, pattern = "\\.csv$", full.names = TRUE,  recursive = TRUE)
  return(csv_files)
}
csv_files <- unlist(lapply(zip_files, collect_csv_files, temp_dir = temp_dir))
unique_csv_files <- unique(csv_files)

#-------------------------- main process json 
library(dplyr)
library(tidyr)
library(arrow)

output_parquet <- "Big_data_estates.parquet"

if (file.exists(output_parquet)) {
  existing_data <- read_parquet(output_parquet)
  existing_file_codes <- unique(existing_data$FileCode)
} else {
  existing_file_codes <- character(0)
}

all_extracted <- list()
for (csv_file in unique_csv_files) {
  df_extracted <- process_csv(csv_file)
  if (nrow(df_extracted) > 0) {
    all_extracted <- append(all_extracted, list(df_extracted))
  } else {
    print(paste("No data extracted from", csv_file))
  }
  
  if (length(all_extracted) > 0) {
    combined_df <- do.call(rbind, all_extracted)
    
    pivot_df <- combined_df %>%
      pivot_wider(names_from = c(Period, DateValue), values_from = ParamValue) %>%
      ungroup()
    
    # Merge EstateName into the final output using FileCode as the key
    estate_mapping <- combined_df %>%
      select(FileCode, EstateName) %>%
      distinct()
    final_df <- left_join(pivot_df, estate_mapping, by = "FileCode")
    write_parquet(final_df, output_parquet)
    print(paste("Saved progress after processing", csv_file))
  }
}

print("Finished processing all files.")

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
  
  data <- read_csv(csv_file)
  
  if (nrow(data) == 0) {
    print(paste("Skipping empty file:", csv_file))
    return(data.frame())
  }
  
  required_cols <- c("identifier", "chart_data", "href")
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
    href_value <- row$href
    extracted_field <- str_match(href_value, "pv=([^&]+)")[, 2]
    field_value <- if (!is.na(extracted_field)) extracted_field else row$identifier
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
          
          extracted_rows <- append(extracted_rows, list(
            tryCatch({
              data.frame(
                FileCode = ifelse(is.na(file_code), NA, file_code),
                EstateName = ifelse(is.na(estate_name), NA, estate_name),
                Field = ifelse(is.na(field_value), NA, field_value),
                ParamValue = json_row[[param_key]],
                Type = json_row$Type,
                DateValue = json_row$DateValue,
                Period = json_row$Period
              )
            }, error = function(e) {
              print(paste("Error creating dataframe for file", csv_file, "row", i, ":", e))
              return(NULL)
            })
          ))
        }
      }
    } else {
      print(paste("No matching chartData pattern found in file", csv_file, "row", i))
    }
  }
  
  return(do.call(rbind, extracted_rows))
}

#------------------- get zips
#csv_file <- "C:\\Users\\jahnv\\Downloads\\extracted_(Belvedere) Greenview Cou..._TW0102.csv"

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
library(tidyverse) 

existing_file_codes <- unique(big_df$FileCode)
qsave(big_df, "big_df1.qs")

#csv_file <- "C:\\Users\\jahnv\\Downloads\\extracted_(Au Tau) Long Shin Estate..._YL0210.csv"

big_df <- tibble() 

for (csv_file in unique_csv_files) {
  df_extracted <- process_csv(csv_file)
  
  if (nrow(df_extracted) > 0) {
    new_cols <- setdiff(names(df_extracted), names(big_df))
    
    if (nrow(big_df) == 0) {
      big_df <- df_extracted
    } else {
      # Ensure column types match before binding
      df_extracted <- mutate(df_extracted, across(names(big_df), as.character))
      big_df <- mutate(big_df, across(names(df_extracted), as.character))
      
      big_df <- bind_rows(big_df, df_extracted)
    }
  } else {
    print(paste("No data extracted from", csv_file))
  }
}


pivot_df <- big_df %>%
  pivot_wider(names_from = Field, values_from = ParamValue) %>%
  ungroup()

library(haven) 

output_dta <- "13may_final.dta"
write_dta(pivot_df, output_dta)

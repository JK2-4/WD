library(devtools)
devtools::install_github("tnagler/VC2copula")
library(VC2copula)

# Define required packages
required_packages <- c("copula", "ggplot2", 
  "dplyr", "lubridate", "tidyr", 
  "networkD3", "igraph", "httr"
)

# Identify packages that are not yet installed
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

# Install missing packages
if(length(new_packages)) install.packages(new_packages)

# Install devtools if not already installed
if (!require(devtools)) {
  install.packages("devtools")
}

library(copula)
library(ggplot2)
library(dplyr)
library(lubridate)
library(tidyr)
library(igraph)
library(httr)

################################


data_file <- "hk_weather_data.csv"
weather_data <- read.csv(data_file)

# Convert 'time' column to Date type
weather_data$time <- as.Date(weather_data$time)

# Inspect the first few rows
head(weather_data)

# Check for missing values
sum(is.na(weather_data))

# Verify no missing values remain
if (any(is.na(weather_data))) {
  # Fill missing values using forward and backward fill
  weather_data <- weather_data %>%
    arrange(time) %>%
    tidyr::fill(tavg, tmin, tmax, prcp, wspd_anomaly, wdir_sin, wdir_cos, pres, temp_anomaly, wspd_anomaly, .direction = "downup")
  
  # If any NAs still exist, fill with column means
  weather_data <- weather_data %>%
    mutate(across(c(tavg, tmin, tmax, prcp, wspd_anomaly, wdir_sin, wdir_cos, pres, temp_anomaly, wspd_anomaly), 
                  ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
}

# Confirm no missing values
stopifnot(!any(is.na(weather_data)))

# variables 
# Select relevant variables
vars <- weather_data %>%
  select(tavg, prcp, wspd_anomaly, wdir_sin, wdir_cos, pres)

# Display summary statistics
summary(vars)

#uniform marginals
# Function to transform data to uniform margins using ranks
to_uniform <- function(x) {
  rank(x, ties.method = "average") / (length(x) + 1)
}
u <- as.data.frame(lapply(vars, to_uniform))
head(u)

data(vars, package = "VineCopula")
pairs(vars[, 1:4])

vine <- vineCopula(as.integer(4))
fit <- fitCopula(vine, vars[, 1:4])
pairs(rCopula(500, fit@copula))


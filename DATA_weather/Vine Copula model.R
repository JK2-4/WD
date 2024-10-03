###################################################################### lib
library(devtools)
devtools::install_github("tnagler/VC2copula")
library(VC2copula)

# Define required packages
required_packages <- c("VineCopula","copula", "ggplot2", 
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

library(VineCopula)
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

######################################################################  Results 
# Fit a vine copula model (R-vine)
# familyset = 1:10 allows for various bivariate copula families
vine_model <- RVineStructureSelect(u, familyset = 1:10, type = "RVine")
print(vine_model)

variable_names <- c("tavg", "prcp", "wspd_anomaly", "wdir_sin", "wdir_cos", "pres")
# Number of variables
d <- length(variable_names)

# Initialize matrices
Matrix <- matrix(0, nrow = d, ncol = d)
family <- matrix(0, nrow = d, ncol = d)
par <- matrix(NA, nrow = d, ncol = d)
par2 <- matrix(NA, nrow = d, ncol = d)

# Assign row and column names for clarity
rownames(Matrix) <- variable_names
colnames(Matrix) <- variable_names
rownames(family) <- variable_names
colnames(family) <- variable_names
rownames(par) <- variable_names
colnames(par) <- variable_names
rownames(par2) <- variable_names
colnames(par2) <- variable_names

names <- variable_names

# Create the RVineMatrix object
RVM <- RVineMatrix(
  Matrix = vine_model$Matrix,
  family = vine_model$family,
  par = vine_model$par,
  par2 = vine_model$par2,
  names = names,
  check.pars = TRUE
)

# View the RVineMatrix object
print(RVM)

#simulate data based on copula 
# Simulate 1000 observations from the fitted vine copula
simulated_data <- RVineSim(1000, vine_model)
head(simulated_data)

# Extract the vine structure matrix
vine_structure <- vine_model$Matrix
print(vine_structure)

# Extract the copula families matrix
copula_families <- vine_model$family
print(copula_families)

# Extract the first parameter matrix
copula_par1 <- vine_model$par
copula_par2 <- vine_model$par2
print(copula_par1)
print(copula_par2)

###################################################################### test log likelihood - tbd
#RVineLogLik(u, RVM, par = RVM$par, par2 = RVM$par2,separate = FALSE, verbose = TRUE, check.pars = TRUE, calculate.V = TRUE)


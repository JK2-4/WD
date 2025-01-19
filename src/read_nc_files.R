# READ NC

# List of required packages
required_packages <- c(
  "lubridate", "dplyr", "xtable", "zoo", 
  "googledrive", "progress", "parallel", "ncdf4"
)

# Install any missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Define directories
out_dir <- "C://Users//jahnv//Downloads//tryout"
output_dir <- "C://Users//jahnv//Downloads//"

# Create directories if they don't exist
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

library(googledrive)
drive_auth()

#LINKS - ---------------------------------------------------------------------------------------------------------------------------------
#https://drive.google.com/file/d/132NI1X0mHbc5QV-vXZmvBft9qm0kGfdV/view?usp=sharing
#https://drive.google.com/file/d/1mKWsSPsvLxEBIsymSnJNZv94AWW-ATnr/view?usp=sharing
#https://drive.google.com/file/d/12_5mCNQEqQIl0mlDmwUohudR13uKMSIH/view?usp=sharing

file_id <- "12_5mCNQEqQIl0mlDmwUohudR13uKMSIH"
temp_nc_file <- tempfile(fileext = ".nc")
drive_download(
  as_id(file_id),
  path = temp_nc_file,
  overwrite = TRUE
)
if (file.exists(temp_nc_file)) {
  message("NetCDF file downloaded successfully to a temporary location.")
} else {
  stop("Failed to download the NetCDF file.")
}
library(ncdf4)

nc_data <- nc_open(temp_nc_file)
print(nc_data)
variables <- names(nc_data$var)
print("Variables in the NetCDF file:")
print(variables)

#variable eg: tasmax ---------------------------------------------------------------------------------------------------------------

tasmax <- ncvar_get(nc_data, "sftlf")
print(dim(tasmax))  # Expected: [512, 256, 366]
summary(tasmax)

time <- ncvar_get(nc_data, "time_bnds")
time_units <- ncatt_get(nc_data, "time", "units")$value
calendar <- ncatt_get(nc_data, "time", "calendar")$value
library(lubridate)
start_date <- ymd_hms(strsplit(time_units, " since ")[[1]][2])
dates <- start_date + days(time - 1.5)
head(dates)

nc_close(nc_data)

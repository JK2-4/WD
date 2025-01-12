if(require(Rmarkdown) == FALSE) 
  install.packages("Rmarkdown")
if(require(here) == FALSE) 
  install.packages("here")
if(require(downloader) == FALSE) 
  install.packages("downloader")
if(require(readr) == FALSE) 
  install.packages("readr")
if(require(reshape2) == FALSE) 
  install.packages("reshape2")
if(require(ggplot2) == FALSE) 
  install.packages("ggplot2")
if(require(dplyr) == FALSE) 
  install.packages("dplyr")
if(require(sn) == FALSE) 
  install.packages("sn")
if(require(geoR) == FALSE) 
  install.packages("geoR")
if(require(cowplot) == FALSE) 
  install.packages("cowplot")

install.packages('RGENERATEPREC')
library(RGENERATEPREC)

cw <- read.csv("C://Users//jkwatra//Downloads//CW_rain_6100_6016_9913.csv", header=TRUE)
# Most % of observations are zeros indicating the dry events
cw$CW_best_rain
dim(cw)
cw[1:5,]

cw$obstime <- as.POSIXct(cw$obstime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
cw$year <- as.numeric(format(cw$obstime, "%Y"))
cw$month <- as.numeric(format(cw$obstime, "%m"))
year_min <- min(cw$year, na.rm = TRUE)
year_max <- max(cw$year, na.rm = TRUE)
cat("Year Range:", year_min, "to", year_max, "\n")
origin <- paste(year_min,1,1,sep="-")

PRECIPITATION <- cw[, c("CW_best_rain", "CW_nearest_rain", "HKO_rain")]
TEMPERATURE_MAX <- cw[, c("CW_best_tmax", "CW_nearest_tmax", "HKO_tmax")]
TEMPERATURE_MIN <- cw[, c("CW_best_tmin", "CW_nearest_tmin", "HKO_tmin")]

period_prec <- cw$year >= year_min & cw$year <= year_max
period_temp <- cw$year >= year_min & cw$year <= year_max

# Subset the data
prec_mes <- PRECIPITATION
Tx_mes <- TEMPERATURE_MAX
Tn_mes <- TEMPERATURE_MIN

valmin <- 1.0
prec_occurrence_mes <- prec_mes>=valmin

exogen <- Tx_mes - Tn_mes
head(exogen)

model_multisite <- PrecipitationOccurrenceMultiSiteModel(x=PRECIPITATION,exogen=exogen,
                                                         origin=origin,multisite_type="wilks")

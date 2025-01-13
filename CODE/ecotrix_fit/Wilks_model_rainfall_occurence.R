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

########################## DATA PREP
cw <- read.csv("C://Users//jkwatra//Downloads//CW_rain_6100_6016_9913.csv", header=TRUE)
# of observations are zeros indicating the dry events
cw$CW_best_rain
dim(cw)
cw[1:5,]

cw$obstime <- as.POSIXct(cw$obstime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
cw$year <- as.numeric(format(cw$obstime, "%Y"))
cw$month <- as.numeric(format(cw$obstime, "%m"))
cw$day <- as.numeric(format(cw$obstime, "%d"))
cw$hr <- as.numeric(format(cw$obstime, "%H"))

year_min <- min(cw$year, na.rm = TRUE)
year_max <- max(cw$year, na.rm = TRUE)
cat("Year Range:", year_min, "to", year_max, "\n")
origin <- paste(year_min,1,1,sep="-")
end <- paste(year_max,12,31,sep="-")

PRECIPITATION <- cw[, c("CW_best_rain", "CW_nearest_rain", "HKO_rain", "day", "month", "year", "hr")]
TEMPERATURE_MAX <- cw[, c("CW_best_tmax", "CW_nearest_tmax", "HKO_tmax","day", "month", "year", "hr")]
TEMPERATURE_MIN <- cw[, c("CW_best_tmin", "CW_nearest_tmin", "HKO_tmin","day", "month", "year", "hr")]

period_prec <- cw$year >= year_min & cw$year <= year_max
period_temp <- cw$year >= year_min & cw$year <= year_max

# Subset the data
prec_mes <- PRECIPITATION
Tx_mes <- TEMPERATURE_MAX
Tn_mes <- TEMPERATURE_MIN

if(sum(is.na(Tx_mes)) != 0)  # NA values? nope - preprocessed in py
  print("ERROR - NA values")
if(sum(is.na(Tn_mes)) != 0) 
  print("ERROR - NA values")
if(sum(is.na(prec_mes)) != 0) 
  print("ERROR - NA values")

nwetdays <- nwetdays(prec_mes,origin)

valmin <- 1
prec_occurrence_mes <- prec_mes>=valmin
station <- names(prec_mes)[!(names(prec_mes) %in% c("day","month","year","hr"))]

########################## MULTISITE Model

station <- station[1:3]
exogen <- Tx_mes[,1:3] - Tn_mes[,1:3]
head(exogen)
months <- factor(prec_mes$month)

model_multisite <- PrecipitationOccurrenceMultiSiteModel(x=prec_mes[,1:3],exogen=exogen,origin=origin,multisite_type="wilks")

########################### LOGIT-type Model

model_multisite_logit <- PrecipitationOccurrenceMultiSiteModel(x=prec_mes, exogen=exogen,
                                                               origin=origin, multisite_type="logit", station=station)
obs_multisite <- prec_mes >= valmin
gen_multisite <- generate(model_multisite, exogen=exogen, origin=origin, end=end)
gen_multisite_logit <- generate(model_multisite_logit, exogen=exogen, origin=origin, end=end)

                                                        
###################### SIMULATE DRY/WET DAYS - 

stations <- c("CW_best_rain", "CW_nearest_rain", "HKO_rain")

precamount <- PrecipitationAmountModel(prec_mes,station=stations[1],origin=origin)
val <- predict(precamount)
prec_gen <- generate(precamount)

non_zero_count <- sum(prec_gen != 0) # non 0 values (wet)
non_zero_count
non_zero <- prec_gen[prec_gen != 0,]
library(ggplot2)  
# non 0 frequency 
ggplot(data.frame(prec_gen = non_zero), aes(x = prec_gen)) +  
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +  
  labs(title = "Histogram of Non-Zero prec_gen", x = "prec_gen Values", y = "Frequency") +  
  theme_minimal() 

###################### single station - compare multi vs single site model

precamount_single <- PrecipitationAmountModel(prec_mes,station=station[1],origin=origin)
val_single <- predict(precamount_single)
prec_gen_single <- generate(precamount_single)
month <- adddate(as.data.frame(residuals(precamount_single[[station[1]]])),origin=origin)$month
plot(factor(month),residuals(precamount_single[[station[1]]]))
### Comparison (Q-Q plot) between multi and single sites.
qqplot(prec_mes[,1],prec_gen[,1],col=1,xlab = "prec measurements",ylab ="model generated")
abline(0,1)
points(sort(prec_mes[,1]),sort(prec_gen_single[,1]),pch=2,col=2)
legend("bottomright",pch=c(1,2),col=c(1,2),legend=c("Multi Sites","Single Site"))
abline(0,1)


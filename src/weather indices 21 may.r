Sys.setlocale("LC_ALL", "English_United States.1252")

# r is a list of estates based data 
install.packages('qs')

library(ncdf4)
library(tools)
library(qs)
install.packages("climdex.pcic")
library(climdex.pcic)
install.packages('PCICt')
library(PCICt)
install.packages('https://pacificclimate.org/R/climdex.pcic_1.1-11.tar.gz')
install.packages(install.packages("C:\\Users\\jkwatra\\Downloads\\RClimDex_2.0.tar.gz", repos=NULL, type="source"))
install.packages("ClimInd")
library(ClimInd)
install.packages('ClimInd', repos = c('https://fergusreiggracia.r-universe.dev', 'https://cloud.r-project.org'))

zip_dir <- "C:\\Users\\jkwatra\\Downloads\\zzipped"
nc_files <- list.files(zip_dir, full.names = TRUE, recursive = TRUE)
all_files<- nc_files

r <- qread('C:\\Users\\jkwatra\\Downloads\\all3k_result_list_try.qs')
for (nm in names(r)) {
  r[[nm]]$indices <- NULL  # Delete 'indices'
  r[[nm]]$data_all_new <- NULL  # Delete 'data_all_new'
}
library(doParallel)
library(foreach)
library(ncdf4)

cl <- makeCluster(5)
registerDoParallel(cl)

results <- foreach(nm = names(r),
                   .packages = c("ncdf4", "tools", "ClimInd"),
                   .export   = c("all_files", "r")) %dopar% {
                     
                     estate <- r[[nm]]
                     we     <- estate$west_east
                     ns     <- estate$north_south
                     lat    <- estate$Latitude
                     
                     data_all_new <- list()
                     for (file in all_files) {
                       fn <- tools::file_path_sans_ext(basename(file))
                       nc <- nc_open(file)
                       vars <- names(nc$var)
                       vname <- if (startsWith(fn, "dom_wdir")) "wind_direction" else vars[4]
                       vec <- as.vector(ncvar_get(nc, vname)[we, ns, ])
                       tu    <- ncatt_get(nc, "Time", "units")$value
                       start <- as.Date(sub("days since ", "", tu))
                       seqd  <- seq.Date(start, by="day", length.out=length(vec))
                       names(vec) <- format(seqd, "%m/%d/%y")
                       data_all_new[[fn]] <- vec
                       nc_close(nc)
                     }
                     
  # flatten periods…
  tmean      <- unlist(data_all_new[c("t2_0110_hk_dailymean", "t2_1120_hk_dailymean", "t2_3039_hk_dailymean", "t2_4049_hk_dailymean")])
  tmax       <- unlist(data_all_new[c("t2_0110_hk_dailymax",  "t2_1120_hk_dailymax",  "t2_3039_hk_dailymax",  "t2_4049_hk_dailymax")])
  tmin       <- unlist(data_all_new[c("t2_0110_hk_dailymin",  "t2_1120_hk_dailymin",  "t2_3039_hk_dailymin",  "t2_4049_hk_dailymin")])
  humid_mean <- unlist(data_all_new[c("rh2_0110_hk_dailymean","rh2_1120_hk_dailymean","rh2_3039_hk_dailymean","rh2_4049_hk_dailymean")])
  precip     <- unlist(data_all_new[c("dp_0110_hk",           "dp_1120_hk",           "dp_3039_hk",           "dp_4049_hk")])
  solar_rad  <- unlist(data_all_new[c("sr_0110_hk_dailymean","sr_1120_hk_dailymean","sr_3039_hk_dailymean","sr_4049_hk_dailymean")])
  wind_speed <- unlist(data_all_new[c("wsp_0110_hk_dailymean","wsp_1120_hk_dailymean","wsp_3039_hk_dailymean","wsp_4049_hk_dailymean")])
  wind_dir   <- unlist(data_all_new[c("dom_wdir_0110_hk",     "dom_wdir_1120_hk",     "dom_wdir_3039_hk",     "dom_wdir_4049_hk")])
  names(tmean) <- sub("^[^.]*\\.", "", names(tmean))
  names(tmax) <- sub("^[^.]*\\.", "", names(tmax))
  names(tmin) <- sub("^[^.]*\\.", "", names(tmin))
  names(humid_mean) <- sub("^[^.]*\\.", "", names(humid_mean))
  names(precip) <- sub("^[^.]*\\.", "", names(precip))
  names(solar_rad) <- sub("^[^.]*\\.", "", names(solar_rad))
  names(wind_speed) <- sub("^[^.]*\\.", "", names(wind_speed))
  names(wind_dir) <- sub("^[^.]*\\.", "", names(wind_dir))
  
  
  
  safe <- function(expr) {
    tryCatch(expr, error = function(e) { warning(e$message); NA })
  }
  
  indices <- list(
    # — Bio/climdex indices —
    bio5   = safe(bio5(tmean, tmax,  data_names = NULL, na.rm = FALSE)),
    bio11  = safe(bio11(tmean, na.rm = FALSE)),
    bio10  = safe(bio10(tmean, na.rm = FALSE)),
    id     = safe(id(tmean, na.rm = FALSE)),
    jci    = safe(jci(tmean, value = lat, na.rm = FALSE)),
    moi    = safe(moi(tmean, lat = lat, time.scale = "month", na.rm = FALSE)),
    ntg    = safe(ntg(tmean, data_names = names(tmean), time.scale = "month", na.rm = FALSE)),
    ogs10  = safe(ogs10(tmean, data_names = names(tmean))),
    wki    = safe(wki(tmean, data_names = names(tmean), time.scale = "month")),
    ws     = safe(ws(tmean, data_names = names(tmean), time.scale = "month")),
    wsdi   = safe(wsdi(tmean, data_names = names(tmean), time.scale = "month")),
    xtg    = safe(xtg(tmean, data_names = names(tmean), time.scale = "month")),
    hi     = safe(hi(taverage = tmean, rh = humid_mean, data_names = names(tmean), time.scale = "month", na.rm = FALSE)),
    hd17   = safe(hd17(tmean, data_names = names(tmean), time.scale = "month")),
    tn90p  = safe(tn90p(tmin, data_names = names(tmin), time.scale = "month")),
    tr     = safe(tr(tmin, data_names = names(tmin), time.scale = "month")),
    tx10p  = safe(tx10p(tmax, data_names = names(tmax), time.scale = "month")),
    vwd    = safe(vwd(tmax, data_names = names(tmax), time.scale = "month")),
    
    # — Precipitation extremes & drought —
    prcptot = safe(prcptot(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    cdd     = safe(cdd(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    cwd     = safe(cwd(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    r10mm   = safe(r10mm(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    r20mm   = safe(r20mm(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    rx1day  = safe(rx1day(precip, data_names = names(precip), time.scale = "year", na.rm = FALSE)),
    rx5d    = safe(rx5d(precip, data_names = names(precip), time.scale = "year", na.rm = FALSE)),
    sdii    = safe(sdii(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    r95tot  = safe(r95tot(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    r99tot  = safe(r99tot(precip, data_names = names(precip), time.scale = "month", na.rm = FALSE)),
    
    # — Temperature‑based indices —
    fd      = safe(fd(tmin, data_names = names(tmin), time.scale = "year", na.rm = FALSE)),
    su      = safe(su(tmax, data_names = names(tmax), time.scale = "year", na.rm = FALSE)),
    csd     = safe(csd(tmax, data_names = names(tmax), time.scale = "month", na.rm = FALSE)),
    stx32   = safe(stx32(tmax, data_names = names(tmax), time.scale = "year", na.rm = FALSE)),
    
    # — Humidity index —
    mi      = safe(mi(taverage = tmean, rh = humid_mean, data_names = names(tmean), time.scale = "month", na.rm = FALSE)),
    
    # — Wind‑based indices —
    fg      = safe(fg(wind_speed, data_names = names(wind_speed), time.scale = "month", na.rm = FALSE)),
    fgcalm  = safe(fgcalm(wind_speed, data_names = names(wind_speed), time.scale = "month", na.rm = FALSE)),
    fg6bft  = safe(fg6bft(wind_speed, data_names = names(wind_speed), time.scale = "month", na.rm = FALSE)),
    
    # — Aridity/continentality —
    mai     = safe(mai(pr = precip, taverage = tmean, time.scale = "year", na.rm = FALSE)),
    mfi     = safe(mfi(precip, data_names = names(precip), time.scale = "year", na.rm = FALSE)),
    
    # — Solar/radiation indices —
    ssd     = safe(ssd(solar_rad, data_names = names(solar_rad), time.scale = "month", na.rm = FALSE)),
    ssp     = safe(ssp(solar_rad, data_names = names(solar_rad), time.scale = "month", na.rm = FALSE)),
    snd     = safe(snd(solar_rad, data_names = names(solar_rad), time.scale = "month", na.rm = FALSE)),
    
    # — Thermal comfort indices —
    wci     = safe(wci(taverage = tmean, w = wind_speed, na.rm = FALSE)),
    utci    = safe(utci(taverage = tmean, rh = humid_mean, w = wind_speed, tmrt = solar_rad, time.scale = "month", na.rm = FALSE))
  )
  list(data_all_new = data_all_new, indices = indices)
}
stopCluster(cl)
  
for (i in seq_along(names(r))) {
  nm <- names(r)[i]
  r[[nm]]$data_all_new <- results[[i]]$data_all_new
  r[[nm]]$indices      <- results[[i]]$indices
}
qsave(result_list, "all3k_.qs")
fwrite(result_list, "all3k_.qs", nThread=5)
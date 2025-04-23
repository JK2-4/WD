# r is a list of estates based data 


for (nm in names(r)) {
  estate <- r[[nm]]
  we     <- estate$west_east
  ns     <- estate$north_south
  lat    <- estate$Latitude
  
  # 1) Read each NetCDF into a named list of daily series
  data_all_new <- list()
  for (file in all_files) {
    fn <- file_path_sans_ext(basename(file))
    nc <- nc_open(file)
    
    # pick variable name
    vars     <- names(nc$var)
    if (startsWith(fn, "dom_wdir")) {
      vname <- "wind_direction"
    } else {
      data_vars <- setdiff(vars, c("lon","lat","time","Time","XLAT","XLONG","XTIME"))
      vname     <- if (length(data_vars)==1) data_vars else data_vars[1]
    }
    
    # extract & name by ISO dates
    vec <- ncvar_get(nc, vname)[we, ns, ]
    tu  <- ncatt_get(nc, "Time", "units")$value
    start <- as.Date(sub("days since ", "", tu))
    dates <- seq.Date(start, by="day", length.out = length(vec))
    names(vec) <- format(dates, "%Y-%m-%d")
    
    data_all_new[[fn]] <- vec
    nc_close(nc)
  }
  
  # 2) Concatenate each variable across your four periods WITHOUT name‐prefixes
  tmean      <- do.call(c, data_all_new[c(
                   "t2_0110_hk_dailymean","t2_1120_hk_dailymean",
                   "t2_3039_hk_dailymean","t2_4049_hk_dailymean")])
  tmax       <- do.call(c, data_all_new[c(
                   "t2_0110_hk_dailymax", "t2_1120_hk_dailymax",
                   "t2_3039_hk_dailymax", "t2_4049_hk_dailymax")])
  tmin       <- do.call(c, data_all_new[c(
                   "t2_0110_hk_dailymin", "t2_1120_hk_dailymin",
                   "t2_3039_hk_dailymin", "t2_4049_hk_dailymin")])
  humid_mean <- do.call(c, data_all_new[c(
                   "rh2_0110_hk_dailymean","rh2_1120_hk_dailymean",
                   "rh2_3039_hk_dailymean","rh2_4049_hk_dailymean")])
  precip     <- do.call(c, data_all_new[c("dp_0110_hk","dp_1120_hk","dp_3039_hk","dp_4049_hk")])
  solar_rad  <- do.call(c, data_all_new[c(
                   "sr_0110_hk_dailymean","sr_1120_hk_dailymean",
                   "sr_3039_hk_dailymean","sr_4049_hk_dailymean")])
  wind_speed <- do.call(c, data_all_new[c(
                   "wsp_0110_hk_dailymean","wsp_1120_hk_dailymean",
                   "wsp_3039_hk_dailymean","wsp_4049_hk_dailymean")])
  wind_dir   <- do.call(c, data_all_new[c(
                   "dom_wdir_0110_hk","dom_wdir_1120_hk",
                   "dom_wdir_3039_hk","dom_wdir_4049_hk")])
  
  # build a real Date vector once
  dates_vec <- as.Date(names(tmean))
  
  # 3) Safely compute every index
  safe <- function(expr) {
    tryCatch(expr, error = function(e) { warning(e$message); NA })
  }
  
  indices <- list(
    # Bio/climdex
    bio5   = safe(bio5(tmean, tmax,          data_names = dates_vec, na.rm=FALSE)),
    bio11  = safe(bio11(tmean,               data_names = dates_vec, na.rm=FALSE)),
    bio10  = safe(bio10(tmean,               data_names = dates_vec, na.rm=FALSE)),
    id     = safe(id(tmean,                  data_names = dates_vec, na.rm=FALSE)),
    jci    = safe(jci(tmean, value=lat,      data_names = dates_vec, na.rm=FALSE)),
    moi    = safe(moi(tmean, lat=lat, time.scale="month",
                      data_names=dates_vec, na.rm=FALSE)),
    ntg    = safe(ntg(tmean, time.scale="month",
                      data_names=dates_vec, na.rm=FALSE)),
    ogs10  = safe(ogs10(tmean,                data_names = dates_vec)),
    wki    = safe(wki(tmean, time.scale="month",
                      data_names = dates_vec)),
    ws     = safe(ws(tmean, time.scale="month",
                     data_names = dates_vec)),
    wsdi   = safe(wsdi(tmean, time.scale="month",
                       data_names = dates_vec)),
    xtg    = safe(xtg(tmean, time.scale="month",
                      data_names = dates_vec)),
    hi     = safe(hi(taverage = tmean, rh = humid_mean,
                     time.scale="month",
                     data_names = dates_vec, na.rm=FALSE)),
    hd17   = safe(hd17(tmean, time.scale="month",
                       data_names = dates_vec)),
    tn90p  = safe(tn90p(tmin, time.scale="month",
                        data_names = as.Date(names(tmin)))),
    tr     = safe(tr(tmin, time.scale="month",
                     data_names = as.Date(names(tmin)))),
    tx10p  = safe(tx10p(tmax, time.scale="month",
                        data_names = as.Date(names(tmax)))),
    tx90p  = safe(tx90p(tmax, time.scale="month",
                        data_names = as.Date(names(tmax)))),
    vwd    = safe(vwd(tmax, time.scale="month",
                      data_names = as.Date(names(tmax)))),
    
    # Precipitation / drought
    prcptot = safe(prcptot(precip, time.scale="month",
                           data_names = dates_vec, na.rm=FALSE)),
    cdd     = safe(cdd(precip, threshold=1, time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    cwd     = safe(cwd(precip, threshold=1, time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    r10mm   = safe(r10mm(precip, time.scale="month",
                         data_names = dates_vec, na.rm=FALSE)),
    r20mm   = safe(r20mm(precip, time.scale="month",
                         data_names = dates_vec, na.rm=FALSE)),
    rx1day  = safe(rx1day(precip, time.scale="year",
                          data_names = dates_vec, na.rm=FALSE)),
    rx5d    = safe(rx5d(precip, time.scale="year",
                        data_names = dates_vec, na.rm=FALSE)),
    sdii    = safe(sdii(precip, time.scale="month",
                        data_names = dates_vec, na.rm=FALSE)),
    r95tot  = safe(r95tot(precip, time.scale="month",
                          data_names = dates_vec, na.rm=FALSE)),
    r99tot  = safe(r99tot(precip, time.scale="month",
                          data_names = dates_vec, na.rm=FALSE)),
    
    # Temperature
    fd      = safe(fd(tmin, time.scale="year",
                      data_names = as.Date(names(tmin)), na.rm=FALSE)),
    id0     = safe(id0(tmin, time.scale="year",
                       data_names = as.Date(names(tmin)), na.rm=FALSE)),
    su      = safe(su(tmax, time.scale="year",
                      data_names = as.Date(names(tmax)), na.rm=FALSE)),
    csd     = safe(csd(tmax, time.scale="month",
                       data_names = as.Date(names(tmax)), na.rm=FALSE)),
    stx32   = safe(stx32(tmax, time.scale="year",
                         data_names = as.Date(names(tmax)), na.rm=FALSE)),
    
    # Humidity
    mi      = safe(mi(taverage=tmean, rh=humid_mean,
                      time.scale="month",
                      data_names = dates_vec, na.rm=FALSE)),
    
    # Wind
    fg      = safe(fg(wind_speed, time.scale="month",
                      data_names = as.Date(names(wind_speed)), na.rm=FALSE)),
    fgcalm  = safe(fgcalm(wind_speed, time.scale="month",
                          data_names = as.Date(names(wind_speed)), na.rm=FALSE)),
    fg6bft  = safe(fg6bft(wind_speed, time.scale="month",
                          data_names = as.Date(names(wind_speed)), na.rm=FALSE)),
    
    # Aridity/continentality
    mai     = safe(mai(pr=precip, taverage=tmean,
                       time.scale="year",
                       data_names = dates_vec, na.rm=FALSE)),
    mfi     = safe(mfi(precip, time.scale="year",
                       data_names = dates_vec, na.rm=FALSE)),
    
    # Solar/radiation
    ssd     = safe(ssd(solar_rad, time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    ssp     = safe(ssp(solar_rad, time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    snd     = safe(snd(solar_rad, time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    
    # Thermal comfort
    wci     = safe(wci(taverage=tmean, rh=humid_mean,
                       w=wind_speed,
                       temperature.metric="celsius",
                       time.scale="month",
                       data_names = dates_vec, na.rm=FALSE)),
    utci    = safe(utci(taverage=tmean, rh=humid_mean,
                        w=wind_speed, tmrt=solar_rad,
                        temperature.metric="celsius",
                        time.scale="month",
                        data_names = dates_vec, na.rm=FALSE))
  )
  
  # 4) attach results
  r[[nm]]$data_all_new <- data_all_new
  r[[nm]]$indices      <- indices
  result_list[[nm]]    <- r[[nm]]
}

# save your full output set
qsave(result_list, "all3k_.qs")
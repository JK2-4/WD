long_df1 <- s2%>%
  pivot_longer(
    cols = matches("^(Year|HalfYear|Quarter|Month|month|quarter|halfyear)_\\d+"),
    names_to = "period",
    values_to = "ParamValue"
  )

long_df1 <- long_df1 %>%
  mutate(
    p = str_to_lower(period),
    
    year = str_extract(p, "\\d{4}"),
    
    month = if_else(
      str_starts(p, "month_"),
      str_extract(p, "(?<=month_\\d{4})\\d{2}"),
      NA_character_
    ),
    
    half = if_else(
      str_starts(p, "halfyear_"),
      str_extract(p, "(?<=halfyear_\\d{4}_)[12]"),
      NA_character_
    ),
    
    quarter = if_else(
      str_starts(p, "quarter_"),
      str_extract(p, "(?<=quarter_\\d{4}q)[1234]"),
      NA_character_
    ),
    
    date = case_when(
      str_starts(p, "year_")     ~ ymd(paste0(year, "-01-01")),
      str_starts(p, "halfyear_") & half=="1" ~ ymd(paste0(year, "-06-30")),
      str_starts(p, "halfyear_") & half=="2" ~ ymd(paste0(year, "-12-31")),
      str_starts(p, "quarter_")  & quarter=="1" ~ ymd(paste0(year, "-01-01")),
      str_starts(p, "quarter_")  & quarter=="2" ~ ymd(paste0(year, "-04-01")),
      str_starts(p, "quarter_")  & quarter=="3" ~ ymd(paste0(year, "-07-01")),
      str_starts(p, "quarter_")  & quarter=="4" ~ ymd(paste0(year, "-10-01")),
      str_starts(p, "month_")    ~ ymd(paste0(year, "-", month, "-01")),
      TRUE                       ~ as.Date(NA)
    )
  )



long_df1 <- long_df1 %>%
  rename_with(~ gsub("\\.", "", .x))
sapply(long_df, is.list)

long_df1_clean <- long_df1 %>%
  mutate(across(where(is.list), ~ map_chr(., ~ paste(unlist(.), collapse = ", "))))

output_dta <- "15may_s2.dta"
write_dta(long_df1_clean, output_dta)

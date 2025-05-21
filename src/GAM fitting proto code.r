library(multidplyr)
library(dplyr)
library(parallel)
library(nycflights13)
library(lubridate)
library(mgcv)

numCores <- detectCores()
cluster <- create_cluster(numCores)

by_dest <- flights %>%
  count(dest) %>%
  filter(n >= 365) %>%
  semi_join(flights, by = "dest") %>%
  mutate(yday = yday(ISOdate(year, month, day))) %>%
  partition(dest, cluster = cluster)

cluster_library(by_dest, "mgcv")

models <- by_dest %>%
  do(mod = gam(dep_delay ~ s(yday) + s(dep_time), data = .))

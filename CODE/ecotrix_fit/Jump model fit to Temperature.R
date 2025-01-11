library(lubridate)   # For date-time parsing
library(tseries)     # For GARCH modeling

rm(list=ls(all=TRUE))
#source("Path//find_jumps.R")
#source("Path//find_arch.R")
#source("Path//Arch_Sim.R")
#source("Path//Jump_Sim.R")
#source("Path//Mean_Rev.R")
#source("Path//find_beta.R")

Path <- "C://Users//jahnv//Downloads//6016.csv"

Temperature_Data = read.csv(Path)
Temperature_Data$obstime <- ymd_hms(Temperature_Data$obstime)
Temperature_Data <- Temperature_Data[order(Temperature_Data$obstime), ]
Temperature <- Temperature_Data$Air.Temperature.in.degree.C
Timestamps <- Temperature_Data$obstime

initial_years <- 5
final_years <- as.numeric(difftime(max(Timestamps), min(Timestamps), units = "weeks")) / 52.1775  # Approximate total years
total_iterations <- floor(final_years - initial_years) + 1
Results <- array(0, c(total_iterations, 18))

overall_start <- min(Timestamps)
overall_end <- max(Timestamps)


# Main loop 
# Loop through each window, starting with initial_years and expanding by 1 year each iteration
for (kk in 1:total_iterations) {
  
  # Calculate the current window's end date
  current_years <- initial_years + (kk - 1)  # Increment year by 1 each iteration
  window_end <- overall_end
  window_start <- window_end - years(current_years)
  
  # Extract data within the current window
  window_indices <- which(Timestamps >= window_start & Timestamps <= window_end)
  Dump_Temp <- Temperature[window_indices]
  Size <- length(Dump_Temp)  # Number of hourly observations in the window
  print(Size)
  time <- 1:Size
  
  # Handle missing data by removing NAs
  valid_indices <- which(!is.na(Dump_Temp))
  Dump_Temp <- Dump_Temp[valid_indices]
  time <- time[valid_indices]
  Size <- length(Dump_Temp)
  print(Size)
  
  # Proceed only if sufficient data is available
  #if (Size < (current_years * 365 * 24) * 0.8) {  # Require at least 80% of expected data
  #  warning(paste("Iteration", kk, ": Insufficient data. Skipping this window."))
  #  next
  #}
  
  # Linear model with seasonal components
  temperature.lm <- lm(Dump_Temp ~ time + sin(2 * pi * time / 24) + cos(2 * pi * time / 24))
  Parameters <- coef(temperature.lm)
  print(Parameters)
  
  A <- Parameters[1]
  B <- Parameters[2]
  C <- sqrt(Parameters[3]^2 + Parameters[4]^2)
  phase <- atan(Parameters[3] / Parameters[4]) - pi
  
  Resid <- residuals(temperature.lm)
  Fit <- fitted.values(temperature.lm)
  
  # Calculate parameters using sourced functions
  Jump_Parameters <- find_jumps(Size, Dump_Temp)
  Arch_Parameters <- find_arch(Dump_Temp)
  b <- Mean_Rev(Dump_Temp, Fit, Size)
  TA <- Arch_Sim(Dump_Temp[1] - Fit[1], Arch_Parameters, Size)
  alpha <- 1
  TY <- Jump_Sim(Jump_Parameters[1:3], Size, alpha)
  
  # Adjust 'beta' calculation based on available Jump_Parameters
  # Ensure that indexing does not exceed bounds
  if ((6 + Size) <= length(Jump_Parameters)) {
    beta <- find_beta(Jump_Parameters[7:(Size + 6)], Size)
  } else {
    beta <- find_beta(Jump_Parameters[7:length(Jump_Parameters)], Size)
  }
  
  TZ <- Jump_Sim(Jump_Parameters[4:6], Size, beta)
  
  # Initialize temperature array and HDD/CDD
  T <- numeric(Size)
  T[1] <- Dump_Temp[1]
  HDD <- 0
  CDD <- 0
  
  if (T[1] > 65) {
    CDD <- T[1] - 65
  }
  if (T[1] < 65) {
    HDD <- 65 - T[1]
  }
  
  # Simulate temperature over time
  for (j in 2:Size) {
    T[j] <- T[j - 1] + Fit[j] - Fit[j - 1] + 
      b * (Fit[j - 1] - T[j - 1]) + 
      TA[j] + TY[j] + TZ[j]
    
    if (T[j] > 65) {
      CDD <- CDD + (T[j] - 65)
    }
    if (T[j] < 65) {
      HDD <- HDD + (65 - T[j])
    }
  }
  
  # Calculate exact HDD and CDD
  HDD_exact <- sum(ifelse(Dump_Temp < 65, 65 - Dump_Temp, 0), na.rm = TRUE)
  CDD_exact <- sum(ifelse(Dump_Temp > 65, Dump_Temp - 65, 0), na.rm = TRUE)
  
  # Calculate difference
  Difference <- sum((T - Dump_Temp)^2, na.rm = TRUE)
  
  # Store results
  Results[kk, 1] <- A
  Results[kk, 2] <- B
  Results[kk, 3] <- C
  Results[kk, 4] <- phase
  Results[kk, 5] <- Arch_Parameters[1]
  Results[kk, 6] <- Arch_Parameters[2]
  Results[kk, 7] <- Jump_Parameters[1]
  Results[kk, 8] <- Jump_Parameters[2]
  Results[kk, 9] <- Jump_Parameters[3]
  Results[kk, 10] <- Jump_Parameters[4]
  Results[kk, 11] <- Jump_Parameters[5]
  Results[kk, 12] <- Jump_Parameters[6]
  Results[kk, 13] <- beta
  Results[kk, 14] <- HDD
  Results[kk, 15] <- HDD_exact
  Results[kk, 16] <- CDD
  Results[kk, 17] <- CDD_exact
  Results[kk, 18] <- b
  
  # Print progress
  print(paste("Iteration", kk, "completed. Window Size:", current_years, "years"))
}

# Save results to a file
output_path <- "C://Users//jahnv//Downloads//Results_6016.csv"  # Replace with your desired output path
write.table(Results, output_path, row.names = FALSE, col.names = FALSE, sep = ",")

# -----------------------------------------------------------------------------
# Function Definitions (Sourced Scripts)
# -----------------------------------------------------------------------------

# Function to find jumps
find_jumps <- function(S, DT) {
  jump_array <- numeric(S)
  jump_array_slow <- numeric(S)
  jump_array_fast <- numeric(S)
  
  SD <- sd(DT, na.rm = TRUE)
  MEAN <- mean(DT, na.rm = TRUE)
  DT_centered <- DT - MEAN
  
  for (i in 1:S) {
    if (DT_centered[i] >= 0) {
      if (DT_centered[i] > 2 * SD) {
        jump_array[i] <- DT_centered[i] - 2 * SD
      } else {
        jump_array[i] <- 0
      }
    } else {
      DT1 <- -DT_centered[i]
      if (DT1 > 2 * SD) {
        jump_array[i] <- DT_centered[i] + 2 * SD
      } else {
        jump_array[i] <- 0
      }
    }
  }
  
  # Initialize first element
  if (jump_array[1] != 0) {
    if (jump_array[2] == 0) {
      jump_array_fast[1] <- jump_array[1]
    } else {
      jump_array_slow[1] <- jump_array[1]
    }
  }
  
  # Initialize last element
  if (jump_array[S] != 0) {
    if (jump_array[S - 1] == 0) {
      jump_array_fast[S] <- jump_array[S]
    } else {
      jump_array_slow[S] <- jump_array[S]
    }
  }
  
  # Process middle elements
  for (j in 2:(S - 1)) {
    if (jump_array[j] != 0) {
      if (jump_array[j - 1] == 0 && jump_array[j + 1] == 0) {
        jump_array_fast[j] <- jump_array[j]
      } else {
        jump_array_slow[j] <- jump_array[j]
      }
    } else {
      jump_array_fast[j] <- 0
      jump_array_slow[j] <- 0
    }
  }
  
  # Extract non-zero jumps
  Dump_Fast <- jump_array_fast[jump_array_fast != 0]
  Dump_Slow <- jump_array_slow[jump_array_slow != 0]
  
  Mean_Fast <- if(length(Dump_Fast) > 0) mean(Dump_Fast) else 0
  SD_Fast <- if(length(Dump_Fast) > 0) sd(Dump_Fast) else 0
  Mean_Slow <- if(length(Dump_Slow) > 0) mean(Dump_Slow) else 0
  SD_Slow <- if(length(Dump_Slow) > 0) sd(Dump_Slow) else 0
  
  # Count slow jumps
  Slow_Counter <- 0
  for (i in 1:(S - 1)) {
    if (jump_array_slow[i] != 0 && jump_array_slow[i + 1] == 0) {
      Slow_Counter <- Slow_Counter + 1
    }
  }
  
  Fast_Intensity <- (S - sum(jump_array_fast == 0)) / S
  Slow_Intensity <- Slow_Counter / S
  
  if ((S - sum(jump_array_fast == 0)) < 2) {
    Fast_Intensity <- 0
    Mean_Fast <- 0
    SD_Fast <- 0
  }
  
  if (Slow_Counter < 2) {  # Adjusted condition
    Slow_Intensity <- 0
    Mean_Slow <- 0
    SD_Slow <- 0
  }
  
  return(c(Fast_Intensity, Mean_Fast, SD_Fast,
           Slow_Intensity, Mean_Slow, SD_Slow,
           jump_array_slow, jump_array_fast))
}

# Function to find ARCH parameters
find_arch <- function(DT) {
  Coef_Arch <- coef(garch(DT, order = c(0, 1), grad = "numerical", trace = FALSE))
  return(Coef_Arch)
}

# Function to find beta
find_beta <- function(J, S) {
  Sum1 <- 0
  Sum2 <- 0
  for (i in 2:S) {
    Sum1 <- Sum1 + (J[i] * J[i - 1])
    Sum2 <- Sum2 + (J[i - 1]^2)
  }
  beta <- (-1) * log(Sum1 / Sum2)
  if (!is.finite(beta)) beta <- 0
  return(beta)
}

# Function to find mean reversion parameter
Mean_Rev <- function(T, F, S) {
  Sum1 <- numeric(S)
  Sum2 <- numeric(S)
  
  for (i in 2:S) {
    current_sd <- sd(T[1:i], na.rm = TRUE)
    if (current_sd != 0) {
      Sum1[i] <- ((T[i - 1] - F[i - 1]) * (T[i] - F[i])) / current_sd
      Sum2[i] <- (T[i] - F[i])^2 / current_sd
    } else {
      Sum1[i] <- 0
      Sum2[i] <- 0
    }
  }
  
  Sum1[!is.finite(Sum1)] <- 0
  Sum2[!is.finite(Sum2)] <- 0
  b <- (-1) * log(sum(Sum1, na.rm = TRUE) / sum(Sum2, na.rm = TRUE))
  
  return(b)
}

# Function for ARCH-based simulation
Arch_Sim <- function(Init, AP, S) {
  S1 <- 10000
  AS_Dump <- matrix(0, nrow = S1, ncol = S)
  AS_Dump[, 1] <- Init
  AS <- numeric(S)
  
  for (i in 1:S1) {
    for (j in 2:S) {
      variance <- AP[1] + AP[2] * AS_Dump[i, j - 1]
      AS_Dump[i, j] <- sqrt(variance) * rnorm(1)
      if (!is.finite(AS_Dump[i, j])) {
        AS_Dump[i, j] <- 0
      }
    }
  }
  
  AS <- colMeans(AS_Dump, na.rm = TRUE)
  return(AS)
}

# Function for jump simulation
Jump_Sim <- function(JP, S, P) {
  Sim_Size <- 10000
  Y <- matrix(0, nrow = Sim_Size, ncol = S)
  J <- numeric(S)
  
  for (k in 1:Sim_Size) {
    t <- 0
    t_count <- c(0)
    lambda <- JP[1]
    
    for (i in 1:S) {
      u <- runif(1)
      t <- t + (-log(u) / lambda)
      if (t > 0 && t <= S) {
        t_count <- c(t_count, t)
      } else {
        break
      }
    }
    
    t_count <- round(t_count)
    Q <- numeric(S)
    
    for (i in 2:length(t_count)) {
      q <- rnorm(1, JP[2], JP[3])
      x <- t_count[i]
      if (!is.na(x) && x >= 1 && x <= S) {  # Added !is.na(x) check
        Q[x] <- Q[x] + q
      }
      # Optionally, handle cases where x is NA or out of bounds
      else {
         next  # Skip to the next iteration
       }
    }
    
    Y[k, 1] <- 0
    for (i in 2:S) {
      Y[k, i] <- Y[k, i - 1] - P * Y[k, i - 1] + Q[i]
    }
  }
  
  J <- colMeans(Y, na.rm = TRUE)
  return(J)
}

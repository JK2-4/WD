# Load necessary libraries
library(dplyr)
library(ggplot2)
library(gridExtra)

# loaded data 
data <- read.csv("C:\\Users\\jkwatra\\Downloads\\marginals_data_hk std id 45007_HK intl airport 2020 Jan-2023 Dec daily.csv")

# Define the number of days in each month for a non-leap year
days_in_month <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


# Define a function to create the empirical vs theoretical plot for a given column
create_plot <- function(column_name) {
  # Calculate Delta for the specified column
  data[[paste0("Delta_", column_name)]] <- c(NA, diff(data[[column_name]]))
  
  # Calculate mean and standard deviation of Delta
  mu_ <- mean(data[[paste0("Delta_", column_name)]], na.rm = TRUE)
  sigma_ <- sd(data[[paste0("Delta_", column_name)]], na.rm = TRUE)
  
  # Define grid for normal distribution
  grid <- seq(min(data[[paste0("Delta_", column_name)]], na.rm = TRUE), 
              max(data[[paste0("Delta_", column_name)]], na.rm = TRUE), 
              length.out = nrow(data))
  
  # Generate plot
  p <- ggplot(data) +
    geom_density(aes(x = .data[[paste0("Delta_", column_name)]], ..density..)) +
    geom_line(aes(x = grid, y = dnorm(grid, mu_, sigma_)), color = "red") +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 0, face = "bold", size = 7), 
      axis.text.y = element_text(face = "bold"), 
      axis.title = element_text(face = "bold"), 
      axis.title.x = element_text(face = "bold", size = 10),
      axis.title.y = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(face = "italic"),
      plot.caption = element_text(face = "italic"),
      panel.grid.major.x = element_line(colour="grey60", linetype="dotted"),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(colour="grey60", linetype="dotted"),
      legend.text = element_text(face = "italic", size = 10),
      legend.title = element_text(face = "bold"),
      legend.position = "top"
    ) +
    labs(
      title = paste("Empiric vs Theoric distribution for", column_name),
      subtitle = "Empiric (Black), Theoric normal (Red)",
      y = "Frequency",
      x = paste("First differences of", column_name)
    )
  return(p)
}

# Create plots for each variable
plot_tavg <- create_plot("tavg")
plot_prcp <- create_plot("prcp")
plot_wspd <- create_plot("wspd_anomaly")
plot_pres <- create_plot("pres")

# Arrange plots in a 2x2 grid
grid.arrange(plot_tavg, plot_prcp, plot_wspd, plot_pres, ncol = 2, top = "Empiric vs Theoric Distribution for Weather Variables")


###################################################################################### model coeff regn 
# Load necessary libraries
library(dplyr)
library(broom)
library(knitr)
library(kableExtra)

df_model <- data 

df_model$DayOfYear <- (1:nrow(df_model)) %% 365
df_model$Month <- cut(df_model$DayOfYear, breaks = cumsum(c(0, days_in_month)), labels = 1:12, include.lowest = TRUE)

# Convert Month to integer
df_model$Month <- as.integer(df_model$Month)

# Add time index
df_model$t <- 1:nrow(df_model)

# Define Omega (annual seasonal frequency for daily data)
df_model$Omega <- 2 * pi / 365

# Rename the temperature variable to match the code's expectations
df_model$Temp <- df_model$tavg

# Fit the seasonal model
seasonal_model <- lm(Temp ~ t + sin(Omega * t) + cos(Omega * t), data = df_model)

# Extract the estimated OLS parameters
a1 <- coef(seasonal_model)[1] 
a2 <- coef(seasonal_model)[2] 
a3 <- coef(seasonal_model)[3] 
a4 <- coef(seasonal_model)[4]

# Compile the results into a table
tibble(
  Model = "$T_t^{m}$ (OLS)", 
  a1 = a1,
  a2 = a2,
  a3 = a3,
  a4 = a4,
  r.squared = glance(seasonal_model)$r.squared,
  sigma = glance(seasonal_model)$sigma
) %>%
  kable(caption = "Estimated parameters for the seasonal model", 
        escape = FALSE) %>%
  kable_classic() %>%
  kable_styling(latex_options = "hold_position")


# Rearrenge the coefficients 
A <- a1
B <- a2
C <- sqrt(a3^2 + a4^2)
Phi <- atan(a4/a3) - base::pi
# Fitted seasonal mean 
df_model$Temp_m <- A + B*df_model$t + C*sin(df_model$Omega*df_model$t + Phi)

# Function for the seasonal drift 
SeasonalDrift <- function(t){
  omega <- 2*base::pi/365
  B + omega * C * cos(omega * t  + Phi)
}
# Function for the seasonal function 
SeasonalFunction <- function(t){
  omega <- 2*base::pi/365
  A + (B * t) + C * sin(omega * t  + Phi)
}

dplyr::tibble(
  Model = "$T_m$", 
  A = A,
  B = B,
  C = C, 
  Phi = Phi) %>%
  knitr::kable(caption = "Coefficients of the regression of Tm", escape = FALSE) %>%
  kableExtra::kable_classic() %>%
  kable_styling(latex_options = "hold_position")

# plot 


g1 <- ggplot(df_model[1:1461,])+
  geom_line(aes(t, Temp), alpha = 0.7)+
  geom_line(aes(t, Temp_m), color = "red")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 0, face = "bold", size = 7), 
        axis.text.y = element_text(face = "bold"), 
        axis.title  = element_text(face = "bold"), 
        axis.title.x = element_text(face = "bold",size = 10),
        axis.title.y = element_text(face = "bold", size = 10),
        plot.title  = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text(face = "italic"),
        panel.grid.major.x = element_line(colour="grey60", linetype="dotted"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dotted"),
        legend.text = element_text(face = "italic", size = 10),
        legend.title = element_text(face = "bold"),
        legend.position = "top" ) +
  labs(
    title = "Alaton's Model: Seasonal Function",
    subtitle = "Empiric mean temperature (Black), Seasonal mean temperature (Red)",
    y = "Temperature (°C)",
    x = NULL,
    caption = "tavg seasonal function caliberated Simulation"
  )


############################ SDE calibration mean estimator
library(purrr)


# Moving Standard Deviation
df_model$Sigma2 <- map_dbl(1:nrow(df_model), ~var(df_model$Temp[1:.x]))
df_model$L1_Sigma2 <- lag(df_model$Sigma2, 1)
# Lag of the Temperature Mean
df_model$L1_Temp_m <- lag(df_model$Temp_m, 1)
# Lag of the Temperature 
df_model$L1_Temp <- lag(df_model$Temp, 1)
# Lag of dependent variable 
df_model$L1_Y <- (df_model$L1_Temp_m - df_model$L1_Temp)/(df_model$L1_Sigma2)
# Remove first two rows to avoid NAs
df_model <- df_model[-c(1,2,3),]
# Mean Reversion Parameter 
a <- -log(sum(df_model$L1_Y * (df_model$Temp - df_model$Temp_m))/sum(df_model$L1_Y * (df_model$L1_Temp- df_model$L1_Temp_m)))


############################## SDE calibration sigma estimator

seasonal_sigma = tibble(Month = 1:12, Nu = 0, Sigma = 0)

for(i in 1:nrow(seasonal_sigma)){
  
  df_month <- dplyr::filter(df_model, Month == i)
  
  Xj <- a * df_month$L1_Temp_m + (1 - a) * df_month$L1_Temp
  Xj_star <- df_month$Temp - (df_month$Temp_m - df_month$L1_Temp_m)
  seasonal_sigma$Nu[i] <- nrow(df_month)
  
  seasonal_sigma$Sigma[i] <- sum((Xj_star - Xj)^2) / (nrow(df_month))
  seasonal_sigma$Sigma[i] <- sqrt(seasonal_sigma$Sigma[i])
}


df_model = left_join(df_model, select(seasonal_sigma, Month, Sigma), by = "Month")

# Create the Seasonal Function for variance
SeasonalSigma <- function(t){
  df_model$Sigma[max(t, 1)]
}

SimulateTemperature <- function(X0 = 1, dt = 1, N = 100, seed = 1){
  
  a = 0.1776014
  dt = 1
  set.seed(seed)
  
  dWt <- rnorm(N, mean = 0, sd = sqrt(dt))
  
  Differences = c(0)
  Deterministic = c(0)
  Stochastic = c(0)
  Variance = c(0)
  Path = c(X0) 
  for(i in 2:N) {
    Deterministic[i] <- (SeasonalDrift(i) + a *( SeasonalFunction(i) - Path[i-1]))*dt
    Variance[i] <- (1 - exp(-2 * a)) / (2 * a)
    Stochastic[i] <- SeasonalSigma(i) * sqrt(Variance[i]) * dWt[i]
    Differences[i] <- Deterministic[i] + Stochastic[i]
    Path[i] <- Path[i-1] + Differences[i]
  }
  
  tibble(t = 1:N, 
         Path = Path, 
         Differences = Differences, 
         Deterministic = Deterministic, 
         Stochastic = Stochastic, 
         Variance = Variance)
}

df_model$Sim1 <- SimulateTemperature(X0 = df_model$Temp[1], N = nrow(df_model), seed = 1)$Path
df_model$Sim2 <- SimulateTemperature(X0 = df_model$Temp[1], N = nrow(df_model), seed = 2)$Path

g2 <- ggplot(df_model[1:1461,])+
  geom_line(aes(t, Sim1), color = "red", alpha = 1)+
  geom_line(aes(t, Temp), alpha = 0.5)+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 0, face = "bold", size = 7), 
        axis.text.y = element_text(face = "bold"), 
        axis.title  = element_text(face = "bold"), 
        axis.title.x = element_text(face = "bold",size = 10),
        axis.title.y = element_text(face = "bold", size = 10),
        plot.title  = element_text(face = "bold"),
        plot.subtitle = element_text(face = "italic"),
        plot.caption = element_text(face = "italic"),
        panel.grid.major.x = element_line(colour="grey60", linetype="dotted"),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour="grey60", linetype="dotted"),
        legend.text = element_text(face = "italic", size = 10),
        legend.title = element_text(face = "bold"),
        legend.position = "top" ) +
  labs(
    title = "Alaton's Model: Simulated vs Empiric",
    subtitle = "Empiric (Black), Simulated (Red)",
    y = "Temperature (°C)",
    x = NULL,
    caption = "tavg SDE caliberated Simulation"
  )

if (!require("gridExtra")) install.packages("gridExtra")
library(gridExtra)

# Combine the two ggplots side by side
grid.arrange(g1, g2, ncol = 2)

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(gridExtra)

# loaded data data <- read.csv(data_file)

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

# Load the data
df_model <- data

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

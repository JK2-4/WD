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



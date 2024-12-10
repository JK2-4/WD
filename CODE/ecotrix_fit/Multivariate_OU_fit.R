# install.packages(c("xtable", "kableExtra"))
library(xtable)
library(kableExtra)
library(xtable)
library(kableExtra)
library(yaml)

######################### Main
# Read the .csv file
df <- read.csv(
  "C://Users//jkwatra//Documents//6001_try.csv",
  colClasses=c("character", rep("numeric", 12))
)

# Find the first line with missing data and remove the data after it
blanks <- which(is.na(df), arr.ind=TRUE)[,"row"]
T <- if(length(blanks) < 1) dim(df)[1] else min(blanks) - 1
df <- df[1:T,]

# Parse the date format to determine the time increments (in seconds)
df$obstime <- as.POSIXct(df$obstime, tz="GMT", format="%Y-%m-%dT%H:%M:%S")
dt <- as.numeric(difftime(df$obstime[2], df$obstime[1], units = "hours"))

# Take a logarithmic transformation of the speed and ti
df$log_speed <- log(df$wind_spd)

# Convert the input data to a matrix
dm <- data.matrix(df[,c("wind_spd", "wind_dir_deg")])

# Run the estimation
fit <- FitOrnsteinUhlenbeck(dm,dt)

# Save the results
library(yaml)
write_yaml(
  list(
    names=rownames(fit$Mu),
    mean=fit$Mu,
    drift=fit$Theta,
    diffusion=chol(fit$Sigma)
  ),
  '..wind_MVOU_process.yml'
)

Mu <- fit$Mu        # Long-term mean vector
Theta <- fit$Theta  # Drift matrix
Sigma <- fit$Sigma  # Diffusion matrix

# Extract parameters
Mu <- fit$Mu
Theta <- fit$Theta
Sigma <- fit$Sigma

# Generate LaTeX tables
mu_latex <- convert_to_latex(Mu, "Long-Term Mean Vector ($\\mu$)")
Theta_latex <- convert_to_latex(Theta, "Drift Matrix ($\\Theta$)")
Sigma_latex <- convert_to_latex(Sigma, "Diffusion Matrix ($\\Sigma$)")

# Combine all LaTeX tables into a single string
all_tables_latex <- paste(mu_latex, Theta_latex, Sigma_latex, sep = "\n\n")

# Write to a .tex file
writeLines(all_tables_latex, "OU_Model_Parameters.tex")



############################################################## Functions (run before main)
FitOrnsteinUhlenbeck = function( Y, tau )
{
  library(pracma)
  T = nrow(Y);
  N = ncol(Y);
  
  X    = Y[ -1,  ];
  F    = cbind( matrix( 1, T-1, 1 ), Y[ -nrow(Y), ] );
  E_XF = t(X) %*% F / T;
  E_FF = t(F) %*% F / T;
  B    = E_XF %*% solve( E_FF );
  if( length( B[ , -1 ] ) != 1 )
  {
    Th = -logm( B[ , -1 ] ) / tau;
    
  }else
  {
    Th = -log( B[ , -1 ] ) / tau;
  }
  
  Mu = solve( diag( 1, N ) - B[ , -1 ] ) %*%  B[ , 1 ] ;
  
  U  = F %*% t(B) - X;
  
  Sig_tau = cov(U);
  
  N = length(Mu);
  TsT = kron( Th, diag( 1, N ) ) + kron( diag( 1, N ), Th );
  
  VecSig_tau = matrix(Sig_tau, N^2, 1);
  VecSig = ( solve( diag( 1, N^2 ) - expm( -TsT * tau ) ) %*% TsT ) %*% VecSig_tau;
  Sig = matrix( VecSig, N, N );
  
  return( list( Mu = Mu, Theta = Th, Sigma = Sig ) )
}


# Function to convert a matrix or vector to a LaTeX table string
convert_to_latex <- function(data, caption) {
  if(is.matrix(data) || is.data.frame(data)) {
    data_df <- as.data.frame(data)
    data_df <- cbind(Variable = rownames(data_df), data_df)
    rownames(data_df) <- NULL
    latex_table <- kable(data_df, format = "latex", booktabs = TRUE, caption = caption) %>%
      kable_styling(latex_options = c("hold_position"))
  } else {
    data_df <- data.frame(Variable = names(data), Value = round(data, 4))
    latex_table <- xtable(data_df, caption = caption)
    latex_table <- print(latex_table, type = "latex", include.rownames = FALSE)
  }
  return(latex_table)
}


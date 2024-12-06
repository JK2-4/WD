
# Set seed for reproducibility
library(Sim.DiffProc)
library(knitr)
knitr::opts_chunk$set(comment="",prompt=TRUE, fig.show='hold', warning=FALSE, message=FALSE)
options(prompt="R> ",scipen=16,digits=5,warning=FALSE, message=FALSE,
        width = 70)

set.seed(1234, kind = "L'Ecuyer-CMRG")

# Initial conditions
y0 = 0.572             # Initial value for S(t)
x0 = 0.12              # Initial value for Y(t)
r = 0.01               # Risk-free rate

# Ornstein-Uhlenbeck (OU) process parameters
gamma = 0.5           
rho = 0.3             
nu = 2.0           
theta = 0.5            
xi = 0.5               # Vol of vol in Y

# S(t) process parameters
a = 0.6               
b = 0.5                
sigma_S = 0.2          

# Option parameters
K = 0.1               
cp = 1                 # 1 for Call, 0 for Put
L = 0.1                # Lower barrier
U = 135.0              # Upper barrier

# Constraints for Y(t)
y_min = 1e-8           # Minimum Y(t) to avoid log(0)
y_max = 150.0        

# ---------------------------- #
#   Simulation of 2D SDEs      #
# ---------------------------- #
fx <- expression(nu*(theta-x),((a*x)+b)*y)
cov_matrix <- matrix(c(xi^2, rho * xi * sigma_S,
                       rho * xi * sigma_S, sigma_S^2),
                     nrow = 2, byrow = TRUE)

# Cholesky decomposition to obtain the diffusion matrix
diffusion_matrix <- t(chol(cov_matrix))

gx <- expression(xi,sigma_S*x)

# Create the SDE object
sde_model <- snssde2d(
  drift = fx,
  diffusion = gx,
  Dt=0.01,
  M=1000,
  x0 = c(x0,y0),
  method = "smilstein", # Using Euler-Maruyama method
  bounds = list(Y = c(y_min, y_max), S = c(0, Inf)) # Enforce Y(t) bounds
)
sde_model
summary(sde_model, at = 10)
plot(sde_model)
plot2d(sde_model,type="n")
points2d(sde_model,col=rgb(0,100,0,50,maxColorValue=255), pch=16)

## the marginal density
denM <- dsde2d(sde_model,pdf="M",at =10)
plot(denM, main="Marginal Density")
## the Joint density
denJ <- dsde2d(sde_model, pdf="J", n=100,at =10)
plot(denJ,display="contour",main="Bivariate Transition Density at time t=10")

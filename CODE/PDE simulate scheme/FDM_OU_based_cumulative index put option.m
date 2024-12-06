% tempelhotput.m 
%        This program determines the price of the weather put option
%        Several parameters are first estimated using the random
%        variable of the precipitation and assuming it follows a
%        Ornestein-Uhlenbeck process. Then the price of the option is
%        determined using a PDE.
%        x is the precipitation and y is the cumulative index
%        C. Nhamgumbe & E. Sousa, April 2023
%
clear;     % clear the variables from memory
clc;       % clear the matlab window from previous tasks
%
x0=0;      % initial value of x
xf=300;    % final value of x
xref=48;   % the mean of the variable x 
y0=0;      % initial value of y
T0 =0;     % initial time of the contract
T=1;       % maturity of the contract
yf=300;    % final value of y 
Tf = T0+T; % final time of the contract
%
%% Parameters necessary to solve the PDE problem
% 
K=33;      % strike price 
r=0.05;    % free risk rate
lambda=0;  % market price of risk
tick=8.1;  % currency converter
%
%% Parameters estimated from the Ornestein-Uhlenbeck process, that is, 
%  sigma, kk, theta(t) 
%
sigma=39.65436; % estimated volatility 
kk=3.5;         % estimated rate of the mean reversion
%
% Estimated parameters to fit function theta(t)
mm=48.2;             % the mean alpha_0
alpha4=-10.0698;
alpha5=9.0541;
alpha6=0.8482;
alpha7=2.8509;
alpha8=-1.4301;
%
theta5=-1.1792;      % nu the reversion parameter (t-nu)
%
%% Steps of the mesh of the numerical method
%
dx=(xf-x0)/(2^8);     % step in the direction x
dy=dx;                % step in the direction y
dt=10^-5;             % step in the time direction
                      % dt should obey to the stability condition
%
%% Griding of the domain
%
Nx=fix((xf-x0)/dx);
Ny=fix((yf-y0)/dy);
Nt=fix((Tf-T0)/dt);
%
x=x0:dx:xf;
y=y0:dy:yf;
t =T0:dt:Tf;
%
[Xi,Yi] = meshgrid(x(1:end),y(1:end));
%
%% Functions needed for PDE
%
% Function f(x)
fJ=zeros(Ny+1,Nx+1);
fJ= xref-Xi; 
%
% Initial solution V(x,y,0)
Uinit=zeros(Ny+1,Nx+1);
Uinit(1:Ny+1,1:Nx+1) = tick.*max(0,K-Yi); % initial solution
%% Uncomment if you want to look at the initial solution
%U=Uinit(2:Ny,2:Nx);
%u = reshape(U',(Ny-1)*(Nx-1),1);
%surf(x,y,Uinit)
%
% Boundary conditions
  Uinit(1:Ny,1)=tick.*max(0,K-Yi(1:Ny,1));           % V(x_max,y,0)
  Uinit(Ny+1,2:Nx+1)=tick.*max(0,K-Yi(Ny+1,2:Nx+1)); % V(x,y_max,0)
%
%% Time iteration of the numerical method   
tic
%
k=1:Ny;
j=2:Nx;
for n=2:Nt+1
  % Definition of the function gamma(x,t) on the PDE
    gama1=zeros(Ny+1,Nx+1);
    gama2=zeros(Ny+1,Nx+1);
    gamaN=zeros(Ny+1,Nx+1);
  % Estimated for monthly unit of time 
    gama1=mm + alpha4*sin(2*pi*(T+T0-t(n-1)-theta5)/12) + alpha5*sin(2*2*pi*(T+T0-t(n-1)-theta5)/12) + alpha6*sin(2*3*pi*(T+T0-t(n-1)-theta5)/12) + alpha7*sin(2*4*pi*(T+T0-t(n-1)-theta5)/12) + alpha8*sin(2*5*pi*(T+T0-t(n-1)-theta5)/12);
    gama2=alpha4*(2*1*pi/12)*cos(2*pi*(T+T0-t(n-1)-theta5)/12) + alpha5*(2*2*pi/12)*cos(2*2*pi*(T+T0-t(n-1)-theta5)/12) + alpha6*(2*3*pi/12)*cos(2*3*pi*(T+T0-t(n-1)-theta5)/12) + alpha7*(2*4*pi/12)*cos(2*4*pi*(T+T0-t(n-1)-theta5)/12) + alpha8*(2*5*pi/12)*cos(2*5*pi*(T+T0-t(n-1)-theta5)/12);
    gamaN=kk*(gama1-Xi)-gama2-lambda*sigma;
    %
  % Coefficients of the numerical method
    %
    FL1 = (-dt)/(2*dx).*gamaN + (sigma^2*dt)/(2*dx^2) + ((dt^2)/(2*dx^2)).*gamaN.*gamaN;
    FL3 = (dt)/(2*dx).*gamaN + (sigma^2*dt)/(2*dx^2) + ((dt^2)/(2*dx^2)).*gamaN.*gamaN;
    Flambda4=(dt)/(dy).*fJ.*max(0,sign(fJ));
    FL6 = 1-r*dt - (dt)/(dy).*fJ.*max(0,sign(fJ)) - (sigma^2*dt)/(dx^2) - ((dt^2)/(dx^2)).*gamaN.*gamaN;
  %
    un=Uinit;
  % Computing all the interior nodes except the last one
    Uinit(k,j)= FL1(k,j).*un(k,j-1) +  FL6(k,j).*un(k,j) + FL3(k,j).*un(k,j+1) + Flambda4(k,j).*un(k+1,j);
  % Computing last node in x separately  because of the Neumann condition
    unNx2=un(k,Nx);
    Uinit(k,Nx+1)= FL1(k,Nx+1).*un(k,Nx) +  FL6(k,Nx+1).*un(k,Nx+1) + FL3(k,Nx+1).*unNx2 + Flambda4(k,Nx+1).*un(k+1,Nx+1);  
  % Boundary conditions
    Uinit(1:Ny,1)=tick.*max(0,K-Yi(1:Ny,1)-xref*(t(n)-T0));
    Uinit(Ny+1,2:Nx+1)=zeros(1,Nx);     
end
timeElapsed=toc
%
%% Plot of the functions 
%
figure(1) % plot of the surface
xx=x(1:Nx/2+1);
yy=y(1:Ny/2+1);
UinitU=Uinit(1:Nx/2+1,1:Ny/2+1);
surf(xx,yy,UinitU,'EdgeColor','none'); 
set(gca,'FontName','Times','FontSize',20);
xlabel('x','FontSize',20);
ylabel('y','FontSize',20);
zlabel('V','FontSize',20);
% 
figure(2) % plot of the level lines for a fixed y
plot(x,Uinit(17,:),'linewidth',3);  
xlabel('x','FontSize',20);
ylabel('V','FontSize',20);
xlim([0 150]); 
%
%
figure(3) % plot of the level lines for a fixed x
plot(y,Uinit(:,10),'linewidth',3);
set(gca,'FontName','Times','FontSize',20);
xlabel('y','FontSize',20);
ylabel('V','FontSize',20);
xlim([0 150]); 

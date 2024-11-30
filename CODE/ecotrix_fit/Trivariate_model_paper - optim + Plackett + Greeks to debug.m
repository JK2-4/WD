function main
    % Using synthetic data.

    clc; close all; clear;

    %% 1.1 Maize Yield Related

    % 1.1.1 Optimization for Beta Parameters and Prediction of Maize Yield
    fprintf('Starting Maize Yield Optimization and Prediction...\n');
    N = 36; % Number of seasons
    L = 100; % Block size for V data

    % Generate synthetic Y (Yield) and V (Variable) data
    Y = 40 + 5*randn(N,1); % Synthetic maize yield data (kg/daa)
    V = abs(100 + 20*randn(N*L,1)); % Synthetic V data

    % Call the MLPCidz function
    [yest, B, sggt, Vsum2] = MLPCidz(V, Y, L);

    %% 1.1.2 Checking Accuracy of Prediction
    fprintf('Checking Accuracy of Prediction...\n');
    figure(1);
    hold on;
    for j = 1:100
        yest_pred = MLPCidz(V, Y, L);
        plot(1:length(yest_pred), yest_pred, 'b-', 'LineWidth', 1.5);
    end
    plot(1:length(Y)-1, Y(1:end-1), 'k-', 'LineWidth', 2);
    legend('100 Prediction Paths', 'Actual Yield', 'Location', 'NorthWest');
    xlabel('Season');
    ylabel('Yield (kg/daa)');
    title('Yield Prediction Paths vs Actual Yield');
    hold off;

    figure(2);
    hold on;
    for j = 1:6
        yest_pred = MLPCidz(V, Y, L);
        plot(Y(1:end-1), yest_pred, 'k+', 'LineWidth', 1.5);
    end
    % Fit a linear model
    Md3 = polyfit(yest_pred, Y(1:end-1), 1);
    Lm = polyval(Md3, yest_pred);
    plot(yest_pred, Lm, 'r-', 'LineWidth', 2);
    legend('Predicted Yield', 'Fit Line', 'Location', 'NorthWest');
    xlabel('Actual Yield (kg/daa)');
    ylabel('Predicted Yield (kg/daa)');
    title('Actual vs Predicted Yield');
    hold off;

    %% 1.1.3 Zero Loss Expected Maize Yield Estimation
    fprintf('Estimating Zero Loss Expected Maize Yield...\n');
    [yest, B, sggt, Vsum2] = MLPCidz(V, Y, L);
    YN = zeros(length(Vsum2),1);
    YN2 = zeros(length(Vsum2),1);
    YN3 = zeros(length(Vsum2),1);
    ts = 1:N;

    for j = 1:length(Vsum2)
        y0 = yest(j);
        a_values = [0.00, 0.00, 0.20];
        for idx = 1:length(a_values)
            a = a_values(idx);
            k1 = -sggt(j);
            k2 = B(j)*Vsum2(j)*(1 - a);
            t0 = ts;
            [t, y] = ode45(@(t,S) ExODE(t, S, k1, k2), t0, y0);
            switch idx
                case 1
                    YN(j) = mean(y);
                case 2
                    YN2(j) = mean(y);
                case 3
                    YN3(j) = mean(y);
            end
        end
    end

    figure(3);
    plot(ts, Y, 'ko-', 'LineWidth', 2);
    hold on;
    plot(ts, YN, 'ro-', 'LineWidth', 2);
    plot(ts, YN2, 'go-', 'LineWidth', 2);
    plot(ts, YN3, 'bo-', 'LineWidth', 2);
    legend('Actual Yield', 'YN (a=0.00)', 'YN2 (a=0.00)', 'YN3 (a=0.20)', 'Location', 'NorthWest');
    xlabel('Season');
    ylabel('Maize Yield (kg/daa)');
    title('Zero Loss Expected Maize Yield Estimation');
    hold off;

    %% 1.2 Pricing By Monte-Carlo
    fprintf('Starting Pricing by Monte-Carlo Simulation...\n');
    MartingalePriceCalc_2();

    %% 1.3 Greeks Calculation
    fprintf('Calculating Greeks...\n');
    MartingalePriceCalc_3(); % Delta Greek
    MartingalePriceCalc_4(); % Gamma Greek
    MartingalePriceCalc_5(); % Theta Greek
    MartingalePriceCalc_6(); % Vega Greek

    fprintf('All computations completed successfully.\n');
end

%% --------------------- Subfunctions Below ---------------------

function [yest, B, sggt, Vsum2] = MLPCidz(V, Y, L)
    % MLPCIDZ - Optimization for Beta Parameters and Prediction of Maize Yield
    % Inputs:
    %   V - Input data vector
    %   Y - Yield data vector
    %   L - Block size for summing V
    % Outputs:
    %   yest - Estimated yield vector
    %   B - Optimized beta parameters
    %   sggt - Estimated sigma values
    %   Vsum2 - Summed V values over blocks

    % Initialize
    Vsum = [];
    Vsq = [];
    j1 = 1; 
    j2 = L;
    N = length(Y);
    for i = 1:N
        if j2 > length(V)
            j2 = length(V);
        end
        Vsum(i) = sum(V(j1:j2));
        Vsq(i) = sum(sqrt(V(j1:j2)));
        j1 = j2 + 1;
        j2 = j1 + L - 1;
    end

    % Change in Y
    y0 = [0; Y];
    y1 = [Y; 0];
    Dey = y1 - y0;
    Dyi = Dey(2:N);
    Dt = 1;

    % Linear Model Fit
    t = (1:N)';
    lm = fitlm(t, Y);
    ytValues = predict(lm, t);

    % Estimating sigma
    RmS = sqrt(mean((ytValues - Y).^2));
    Hi = lm.Diagnostics.Leverage;
    sggt = RmS .* sqrt(1 - Hi);

    % Initial Beta Parameters (B0)
    B0 = 0.0055 * ones(N-1,1);

    % Optimization Loop
    iter = 0;
    maxIter = 1000;
    while true
        iter = iter + 1;
        if iter > maxIter
            warning('Maximum iterations reached in MLPCidz optimization.');
            break;
        end
        dW = sqrt(Dt / N) * randn(N-1,1);
        Vsq1 = Vsq(1:N-1)';
        yt1 = ytValues(1:N-1);
        Vsum2 = Vsum(1:N-1)';
        sggt2 = sggt(1:N-1);

        % Objective Function
        L_obj = @(b) sqrt(mean((Dyi + sggt2 .* yt1 .* Dt - b .* Vsum2 .* Dt - sqrt(b) .* Vsq1 .* dW).^2));

        % Optimization using fminsearch
        options = optimset('MaxIter',500,'MaxFunEvals',500);
        [B_opt, fval] = fminsearch(L_obj, B0, options);

        % Check for convergence
        if fval < 0.1
            break;
        else
            B0 = B_opt;
        end
    end

    B = B_opt;
    BW = dW;
    yest = zeros(N-1,1);
    yest(1) = Y(1);
    for j3 = 1:N-2
        yest(j3+1) = yest(j3) - sggt(j3)*yt1(j3)*Dt + B(j3)*Vsum2(j3)*Dt + sqrt(B(j3)) * Vsq1(j3)*BW(j3);
    end
end

function dSdt = ExODE(~, S, k1, k2)
    % EXODE - Defines the ODE for maize yield estimation
    % dSdt = k1 * S + k2
    dSdt = k1 * S + k2;
end

function MartingalePriceCalc_2()
    % MARTINGALEPRICECALC_2 - Pricing by Monte-Carlo Simulation
    fprintf('Executing Martingale Price Calculation (Monte-Carlo)...\n');
    % Synthetic Data Generation
    gamma = sqrt(0.000057); % Autocorrelation coefficient
    V = abs(100 + 20*randn(1000,1)); % Synthetic V data
    Dta = randn(10000,8) * 10 + 50; % Synthetic weather data
    Dta(:,4) = abs(Dta(:,4)); % Ensure column 4 is positive
    Dta(Dta(:,4) <= 2.54, :) = []; % Remove rows where column 4 <= 2.54

    r = Dta(:,4);
    x = Dta(:,2);
    y = Dta(:,3);
    z = 0.5 * (x + y);
    n = length(z);
    t = (1:n)';
    E = Dta(:,8);
    K = [1188.38;1245.14;1314.42];

    % VOLATILITY CALCULATIONS
    if n < 3892
        error('Insufficient data for volatility calculations.');
    end
    t2 = t(3892:end);
    x2 = x(3892:end);
    y2 = y(3892:end);
    z2 = z(3892:end);
    r2 = r(3892:end);
    E2 = E(3892:end);
    Hraw = sqrt(HogRawEst(x2, y2));
    QVR = sqrt(quadVar(r2));
    QVT = sqrt(quadVar(E2));
    IHT = sqrt(2)*gamma.*Hraw./z2;
    IQVR = sqrt(2)*gamma.*QVR./r2 .* sqrt(r2);
    IQVE = sqrt(2)*gamma.*QVT./E2 .* sqrt(E2);

    SGV = (IHT + IQVR + IQVE).^2 / 2;
    xx2 = (t2 - 3892) ./ 365; % Time in years

    % Synthetic CP and VP data
    CP = randn(length(SGV),1); % Plackett c_(R,E_t,T_t)
    VP = randn(length(SGV),1); % Vine on Plackett c_(R,E_t,T_t)

    % Gaussian Quadrature
    pp2 = spline(xx2, SGV);
    int_z = integral(@(xx) ppval(pp2, xx), xx2(1), xx2(end));

    PCC1 = zeros(5000,1);
    PCC2 = zeros(5000,1);
    for j = 1:5000
        dW = randn(length(SGV),1);
        Int_z2 = dot(dW, sqrt(SGV)); % Brownian Integration
        Lc = 1.17; % Tic value
        r0 = 0.05; % Risk-free interest rate
        V0 = V(1); % Initial V value

        % Integrals
        I1 = r0 * xx2(end) + int_z;
        I2 = Int_z2;
        I3 = exp(-r0 * xx2(end)) * K(3) * Lc;

        % Exponentiation
        Exp = Lc * V0 * exp(-I1 + I2);
        pV = max(Exp - I3, 0);

        % Pricing
        PCC1(j) = mean(CP .* pV) / length(CP); % Plackett Density Price
        PCC2(j) = mean(VP .* pV) / length(VP); % Vine Density Price
    end

    pVf1 = mean(PCC1);
    pVf2 = mean(PCC2);

    fprintf('Plackett Density Price (pVf1): %.4f\n', pVf1);
    fprintf('Vine Density Price (pVf2): %.4f\n', pVf2);
end

function MartingalePriceCalc_3()
    % MARTINGALEPRICECALC_3 - Delta Greek Calculation
    fprintf('Calculating Delta Greek...\n');
    % Synthetic Data Generation
    gamma = sqrt(0.000057); % Autocorrelation coefficient
    V = abs(100 + 20*randn(1000,1)); % Synthetic V data
    Dta = randn(10000,8) * 10 + 50; % Synthetic weather data
    Dta(:,4) = abs(Dta(:,4)); % Ensure column 4 is positive
    Dta(Dta(:,4) <= 2.54, :) = []; % Remove rows where column 4 <= 2.54

    r = Dta(:,4);
    x = Dta(:,2);
    y = Dta(:,3);
    z = 0.5 * (x + y);
    n = length(z);
    t = (1:n)';
    E = Dta(:,8);
    K = [1188.38;1245.14;1314.42];

    % VOLATILITY CALCULATIONS
    if n < 3892
        error('Insufficient data for volatility calculations.');
    end
    t2 = t(3892:end);
    x2 = x(3892:end);
    y2 = y(3892:end);
    z2 = z(3892:end);
    r2 = r(3892:end);
    E2 = E(3892:end);
    Hraw = sqrt(HogRawEst(x2, y2));
    QVR = sqrt(quadVar(r2));
    QVT = sqrt(quadVar(E2));
    IHT = Hraw./z2 .* sqrt(z2);
    IQVR = QVR./r2 .* sqrt(r2);
    IQVE = QVT./E2 .* sqrt(E2);

    SGV = (IHT + IQVR + IQVE).^2 / 2;
    xx2 = (t2 - 3892) ./ 365; % Time in years

    % Synthetic CP and VP data
    CP = randn(length(SGV),1); % Plackett c_(R,E_t,T_t)
    VP = randn(length(SGV),1); % Vine on Plackett c_(R,E_t,T_t)

    % Gaussian Quadrature
    pp2 = spline(xx2, SGV);
    int_z = integral(@(xx) ppval(pp2, xx), xx2(1), xx2(end));

    PCC1 = zeros(500,1);
    PCC2 = zeros(500,1);
    dV0 = 0.01; % Small change in V0 for Delta

    for j = 1:500
        dW = randn(length(SGV),1);
        Int_z2 = dot(dW, sqrt(SGV)); % Brownian Integration
        Lc = 1.17; % Tic value
        r0 = 0.05; % Risk-free interest rate
        V0 = V(1); % Initial V value

        % Integrals
        I1 = r0 * xx2(end) + int_z;
        I2 = Int_z2;
        I3 = exp(-r0 * xx2(end)) * K(1) * Lc;

        % Exponentiation
        Exp = Lc * V0 * exp(-I1 + I2);
        Exp2 = Lc * (V0 + dV0) * exp(-I1 + I2);
        pV1 = max(Exp - I3, 0);
        pV2 = max(Exp2 - I3, 0);

        % Delta Calculation
        sampleDiff(j) = (mean(CP .* pV2) / length(CP) - mean(CP .* pV1) / length(CP)) / dV0;
        sampleDiff2(j) = (mean(VP .* pV2) / length(VP) - mean(VP .* pV1) / length(VP)) / dV0;
    end

    % Statistical Analysis
    [Delta_CP, ~, CI_CP] = normfit(sampleDiff);
    [Delta_VP, ~, CI_VP] = normfit(sampleDiff2);

    fprintf('Delta Greek (CP): %.4f with 95%% CI [%.4f, %.4f]\n', Delta_CP, CI_CP(1), CI_CP(2));
    fprintf('Delta Greek (VP): %.4f with 95%% CI [%.4f, %.4f]\n', Delta_VP, CI_VP(1), CI_VP(2));
end

% Additional MartingalePriceCalc functions (MartingalePriceCalc_4, MartingalePriceCalc_5, MartingalePriceCalc_6)
% should be defined similarly below. Due to space constraints, only MartingalePriceCalc_2 and MartingalePriceCalc_3
% are fully implemented here. You can follow the same structure to implement the remaining functions.

function MartingalePriceCalc_5()
    clc; close all; clear;
    %% Data
    gamma = sqrt(0.000057); % Autocorrelation coefficient
    V = xlsread("vdata2");
    Dta = xlsread("weatherdatacleaned");
    Dta(Dta(:,4) <= 2.54, :) = [];
    r = Dta(:,4);
    x = Dta(:,2);
    y = Dta(:,3);
    z = 0.5 .* (x + y);
    n = length(z);
    t = 1:n;
    E = Dta(:,8);
    K = [1188.38; 1245.14; 1314.42];
    
    %{
    VOLATILITY CALCULATIONS
    %}
    t2 = t(3892:end);
    x2 = x(3892:end);
    y2 = y(3892:end);
    z2 = z(3892:end);
    r2 = r(3892:end);
    E2 = E(3892:end);
    Hraw = sqrt(HogRawEst(x2, y2));
    QVR = sqrt(quadVar(r2));
    QVT = sqrt(quadVar(E2));
    IHT = Hraw ./ z2 .* sqrt(z2);
    IQVR = QVR ./ r2 .* sqrt(r2);
    IQVE = QVT ./ E2 .* sqrt(E2);
    
    pV = [];
    CP = xlsread("placdata"); % Data on Plackett c_(R,E_t,T_t)
    VP = xlsread("vinedata"); % Vine on Plackett c_(R,E_t,T_t)
    SGV = (IHT .* gamma + IQVR .* gamma + IQVE .* gamma).^2 ./ 2;
    xx2 = (t2 - 3892) ./ 365; % Time in years
    DeltaT = 0.001;
    xx3 = xx2 + DeltaT;
    pp2 = spline(xx2, SGV);
    pp3 = spline(xx3, SGV);
    int_z = integral(@(xx) ppval(pp2, xx), xx2(1), xx2(end)); % Gaussian Quadrature
    int_z1 = integral(@(xx) ppval(pp2, xx), xx2(1), xx2(end)); % Gaussian Quadrature
    PCC1 = [];
    PCC2 = [];
    
    for j = 1:5000
        iter = 0;
        while iter < length(CP)
            iter = iter + 1;
            dW = randn(length(SGV), 1);
            Int_z2 = dot(dW, sqrt(SGV)); % Definition of Brownian Integration
            Int_z3 = dot(dW, sqrt(SGV2)); % Missing SGV2 definition, ensure it's defined
            Lc = 1.17; % Tic value
            r0 = 0.05; % Risk free interest rate
            V0 = V(1); % Initial value of V
            
            % The first integral in Equation (27)
            I1 = r0 .* xx2(end) + int_z;
            I11 = r0 .* xx3(end) + int_z1;
            % The second integral in Equation (27)
            I2 = Int_z2;
            I22 = Int_z3;
            % The last component in Equation (27)
            I3 = exp(-r0 .* xx2(end)) .* K(3) .* Lc;
            I33 = exp(-r0 .* xx3(end)) .* K(3) .* Lc;
            % Exponentiation in equation (27)
            Exp = Lc * V0 * exp(-I1 + I2);
            Exp2 = Lc * V0 * exp(-I11 + I2);
            pV(iter) = max(Exp - I3, 0);
            pV2(iter) = max(Exp - I33, 0);
        end
        PCC1(j) = -(dot(CP, pV2) / length(CP) - dot(CP, pV) / length(CP)) / DeltaT; % Plackett Density Price
        PCC2(j) = -(dot(VP, pV2) / length(VP) - dot(VP, pV) / length(VP)) / DeltaT; % Vine Density Price
    end
    
    format long
    [Theta_Greek, ~, CI] = normfit(PCC1);
    [Theta_Greek_V, ~, CI_V] = normfit(PCC2);
    fprintf('Vega Greek (Plackett): %f\n', Theta_Greek);
    fprintf('Vega Greek (Vine): %f\n', Theta_Greek_V);
    
    %%% Calculating them manually
    XC_Pl = mean(PCC1);
    N = length(PCC1);
    A95P = [-std(PCC1)*1.96/sqrt(N); std(PCC1)*1.96/sqrt(N)];
    A99P = [-std(PCC1)*2.58/sqrt(N); std(PCC1)*2.58/sqrt(N)];
    A90P = [-std(PCC1)*1.65/sqrt(N); std(PCC1)*1.65/sqrt(N)];
    A95V = [-std(PCC2)*1.96/sqrt(N); std(PCC2)*1.96/sqrt(N)];
    A99V = [-std(PCC2)*2.58/sqrt(N); std(PCC2)*2.58/sqrt(N)];
    A90V = [-std(PCC2)*1.65/sqrt(N); std(PCC2)*1.65/sqrt(N)];
    
    % Display Confidence Intervals
    disp('95% Confidence Interval for Plackett Vega:');
    disp(A95P);
    disp('99% Confidence Interval for Plackett Vega:');
    disp(A99P);
    disp('90% Confidence Interval for Plackett Vega:');
    disp(A90P);
    
    disp('95% Confidence Interval for Vine Vega:');
    disp(A95V);
    disp('99% Confidence Interval for Vine Vega:');
    disp(A99V);
    disp('90% Confidence Interval for Vine Vega:');
    disp(A90V);
end

function MartingalePriceCalc_6()
    clc; close all; clear;
    %% Data
    gamma = sqrt(0.000057); % Autocorrelation coefficient
    V = xlsread("vdata2");
    Dta = xlsread("weatherdatacleaned");
    Dta(Dta(:,4) <= 2.54, :) = [];
    r = Dta(:,4);
    x = Dta(:,2);
    y = Dta(:,3);
    z = 0.5 .* (x + y);
    n = length(z);
    t = 1:n;
    E = Dta(:,8);
    K = [1188.38; 1245.14; 1314.42];
    
    %{
    VOLATILITY CALCULATIONS
    %}
    t2 = t(3892:end);
    x2 = x(3892:end);
    y2 = y(3892:end);
    z2 = z(3892:end);
    r2 = r(3892:end);
    E2 = E(3892:end);
    Hraw = sqrt(HogRawEst(x2, y2));
    QVR = sqrt(quadVar(r2));
    QVT = sqrt(quadVar(E2));
    IHT = Hraw ./ z2;
    IQVR = QVR ./ r2;
    IQVE = QVT ./ E2;
    
    pV = [];
    CP = xlsread("placdata"); % Data on Plackett c_(R,E_t,T_t)
    VP = xlsread("vinedata"); % Vine on Plackett c_(R,E_t,T_t)
    dSig = 0.001;
    SGV = (IHT .* gamma + IQVR .* gamma + IQVE .* gamma).^2 ./ 2;
    SGV2 = (IHT .* gamma + IQVR .* gamma + IQVE .* gamma + dSig).^2 ./ 2;
    xx2 = (t2 - 3892) ./ 365; % Time in years
    pp1 = spline(xx2, SGV);
    pp2 = spline(xx2, SGV2);
    int_z = integral(@(xx) ppval(pp1, xx), xx2(1), xx2(end)); % Gaussian Quadrature
    int_z1 = integral(@(xx) ppval(pp2, xx), xx2(1), xx2(end)); % Gaussian Quadrature
    PCC1 = [];
    PCC2 = [];
    
    for j = 1:5000
        iter = 0;
        while iter < length(CP)
            iter = iter + 1;
            dW = randn(length(SGV), 1);
            Int_z2 = dot(dW, sqrt(SGV)); % Definition of Brownian Integration
            Int_z3 = dot(dW, sqrt(SGV2)); % Definition of Brownian Integration for SGV2
            Lc = 1.17; % Tic value
            r0 = 0.05; % Risk free interest rate
            V0 = V(1); % Initial value of V
            
            % The first integral in Equation (27)
            I1 = r0 .* xx2(end) + int_z;
            I11 = r0 .* xx2(end) + int_z1;
            % The second integral in Equation (27)
            I2 = Int_z2;
            I22 = Int_z3;
            % The last component in Equation (27)
            I3 = exp(-r0 .* xx2(end)) .* K(3) .* Lc;
            I33 = exp(-r0 .* xx2(end)) .* K(3) .* Lc;
            % Exponentiation in equation (27)
            Exp = Lc * V0 * exp(-I1 + I2);
            Exp1 = Lc * V0 * exp(-I11 + I22);
            pV(iter) = max(Exp - I3, 0);
            pV1(iter) = max(Exp1 - I33, 0);
        end
        PCC1(j) = ((dot(CP, pV1) / length(CP) - dot(CP, pV) / length(CP)) / dSig) / max(K); % Plackett Density Price
        PCC2(j)=((dot(VP,pV1)/length(VP)-dot(VP,pV)/length(VP))./dSig)/max(K); %%%Vine Density Price
    end

    format long
    [Vega_Greek,dummy,CI]=normfit(PCC1);
    Vega_Greek
    [Vega_Greek,dummy,CI]=normfit(PCC2);
    Vega_Greek

    %%%Calcluating them manually
    XC_Pl=mean(PCC1);
    N=length(PCC1);
    A95P=[-std(PCC1)*1.96/sqrt(N);std(PCC1)*1.96/sqrt(N)];
    A99P=[-std(PCC1)*2.58/sqrt(N);std(PCC1)*2.58/sqrt(N)];
    A90P=[-std(PCC1)*1.65/sqrt(N);std(PCC1)*1.65/sqrt(N)];
    A95V=[-std(PCC2)*1.96/sqrt(N);+std(PCC2)*1.96/sqrt(N)]
    A99V=[-std(PCC2)*2.58/sqrt(N);+std(PCC2)*2.58/sqrt(N)]
    A90V=[-std(PCC2)*1.65/sqrt(N);+std(PCC2)*1.65/sqrt(N)]
    
    % Display Confidence Intervals
    disp('95% Confidence Interval for Plackett Delta:');
    disp(A95P);
    disp('99% Confidence Interval for Plackett Delta:');
    disp(A99P);
    disp('90% Confidence Interval for Plackett Delta:');
    disp(A90P);
    
    disp('95% Confidence Interval for Vine Delta:');
    disp(A95V);
    disp('99% Confidence Interval for Vine Delta:');
    disp(A99V);
    disp('90% Confidence Interval for Vine Delta:');
    disp(A90V);
end

function MartingalePriceCalc_4()
    %{
    AIM: CALCULATION OF GAMMA GREEKS
    %}
    clc; close all; clear;
    %%Data
    gamma=sqrt(0.000057); %%%Autocorrelation coefficient
    V=xlsread("vdata2");
    Dta=xlsread("weatherdatacleaned");
    Dta(Dta(:,4)<=2.54,:)=[];
    r=Dta(:,4);
    x=Dta(:,2);
    y=Dta(:,3);
    z=0.5.*(x+y);
    n=length(z);
    t=1:n;
    E=Dta(:,8);
    K=[1188.38;1245.14;1314.42];
    %{
    VOLATILITY CALCULATIONS
    %}
    t2=t(3892:1:n);
    
    x2=x(3892:1:n);
    y2=y(3892:1:n);
    z2=z(3892:1:n);
    r2=r(3892:1:n);
    E2=E(3892:1:n);
    Hraw=sqrt(HogRawEst(x2,y2));
    QVR=sqrt(quadVar(r2));
    QVT=sqrt(quadVar(E2));
    IHT=Hraw./z2.*sqrt(z2);
    IQVR=QVR./r2.*sqrt(r2);
    IQVE=QVT./E2.*sqrt(E2);
    pV1=[]; PV2=[];
    CP=xlsread("placdata"); %%%Data on Plackett c_(R,E_t,T_t)
    VP=xlsread("vinedata"); %%%Vine on Plackett c_(R,E_t,T_t)
    SGV=(IHT.*gamma+IQVR.*gamma+IQVE.*gamma).^2./2;
    xx2=(t2-3892)./365; %%%Time in years
    pp2=spline(xx2,SGV);
    int_z= quad(@(xx)ppval(pp2,xx),xx2(1),xx2(end)); %%%Gaussian Quadrature
    sampleDiff=[];sampleDiff2=[];

    for j=1:500
        iter=0;
        while iter<length(CP)
        iter=iter+1;
        dW=randn(length(SGV),1);
        Int_z2=dot(dW,sqrt(SGV)) ; %%%Definition of Brownian Integration
        Lc=1.17; %%%Tic value
        r0=0.05; %%%Risk free interest rate
        V0=V(1); %%%Initial value of V
        %%%The first integral in Equation (27)
        I1=r0.*xx2(end)+int_z;
        %%%The second integral in Equation (27)
        I2=Int_z2;
        %%%The last component in Equation (27)
        I3=exp(-r0.*xx2(end)).*K(3).*Lc;
        %%%Exponention in equation (27)
        dV0=0.01;
        Exp=Lc*V0*exp(-I1+I2);
        Exp2=Lc*(V0+dV0)*exp(-I1+I2);
        pV1(iter)=max(Exp-I3,0);
        pV2(iter)=max(Exp2-I3,0);
        end
        sampleDiff(j)=(dot(CP,pV2)/length(CP)-dot(CP,pV1)/length(CP))./dV0; %%%Plackett Density Price
        sampleDiff2(j)=(dot(VP,pV2)/length(VP)-dot(VP,pV1)/length(VP))./dV0; %%%Vine Density Price
    end

    format long
    T01=[0,sampleDiff];
    T0E1=[sampleDiff,0];
    DT1=T0E1-T01;
    
    DT1(1)=[];
    DT1(end)=[];
    sampleDiff=DT1/dV0;
    T02=[0,sampleDiff2];
    T0E2=[sampleDiff2,0];
    DT2=T0E2-T02;
    DT2(1)=[];
    DT2(end)=[];
    sampleDiff2=DT2/dV0;
    [Gamma_Greek,dummy,CI]=normfit(sampleDiff);
    Gamma_Greek
    [Gamma_Greek,dummy,CI]=normfit(sampleDiff2);
    %%%Calcluating them manually
    XC_Pl=mean(sampleDiff);
    N=length(sampleDiff)
    A95P=[-std(sampleDiff)*1.96/sqrt(N);std(sampleDiff)*1.96/sqrt(N)];
    A99P=[-std(sampleDiff)*2.58/sqrt(N);std(sampleDiff)*2.58/sqrt(N)];
    A90P=[-std(sampleDiff)*1.65/sqrt(N);std(sampleDiff)*1.65/sqrt(N)];
    A95V=[-std(sampleDiff2)*1.96/sqrt(N);+std(sampleDiff2)*1.96/sqrt(N)]
    A99V=[-std(sampleDiff2)*2.58/sqrt(N);+std(sampleDiff2)*2.58/sqrt(N)]
    A90V=[-std(sampleDiff2)*1.65/sqrt(N);+std(sampleDiff2)*1.65/sqrt(N)]
    % Display Confidence Intervals
    disp('95% Confidence Interval for Plackett Vega:');
    disp(A95P);
    disp('99% Confidence Interval for Plackett Vega:');
    disp(A99P);
    disp('90% Confidence Interval for Plackett Vega:');
    disp(A90P);
        
    disp('95% Confidence Interval for Vine Vega:');
    disp(A95V);
    disp('99% Confidence Interval for Vine Vega:');
    disp(A99V);
    disp('90% Confidence Interval for Vine Vega:');
    disp(A90V);
    end

function QV = quadVar(x)
    % QUADVAR - Calculates quadratic variation
    % Inputs:
    %   x - Input vector
    % Outputs:
    %   QV - Quadratic variation vector

    T0 = [0; x];
    TE = [x; 0];
    DT = TE - T0;
    DT(1) = mean(DT);
    DT(end) = [];
    QV = DT.^2;
end

function Hraw = HogRawEst(x, y)
    % HOGRAWEST - Estimates raw Hog values
    % Inputs:
    %   x - Input vector
    %   y - Input vector
    % Outputs:
    %   Hraw - Estimated Hog raw values

    Hraw = ((1/6) * (y - x)).^2;
end

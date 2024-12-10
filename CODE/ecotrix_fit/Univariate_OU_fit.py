import pandas as pd
import numpy as np
import scipy.stats as ss
import yaml
from pathlib import Path

# Define output paths
output_csv_path = Path("/content/drive/MyDrive/wFTproject/MLE_OU/aligned_results.csv")
output_yaml_path = Path("/content/drive/MyDrive/wFTproject/MLE_OU/aligned_results.yaml")

#import pickle
#with open('/content/drive/MyDrive/wFTproject/aligned_tables.pkl', 'rb') as f:
#    aligned_table = pickle.load(f)

# Initialize results container
all_results = []

# Iterate through all stations
for station_id, aligned_table in aligned_tables.items():
    try:
        # Extract wind speed and observation time
        wind = aligned_table[['table_9_Wind Speed in km/hr', 'obstime']].rename(columns={'table_9_Wind Speed in km/hr': 'wind_spd'}).dropna()

        # Ensure there is enough data
        if len(wind) < 2:
            print(f"Insufficient data for station {station_id}. Skipping.")
            continue

        # Prepare data for OLS and MLE
        XX = wind.wind_spd[:-1]
        YY = wind.wind_spd[1:]
        N = len(XX) + 1
        T = 10 * 12 * 30  # daily
        T_vec, dt = np.linspace(0, T, N, retstep=True)

        # OLS
        beta, alpha, _, _, _ = ss.linregress(XX, YY)
        kappa_ols = -np.log(beta) / dt
        theta_ols = alpha / (1 - beta)
        res = YY - beta * XX - alpha
        std_resid = np.std(res, ddof=2)
        sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta**2))

        # MLE
        Sx = np.sum(XX)
        Sy = np.sum(YY)
        Sxx = XX @ XX
        Sxy = XX.values @ YY.values
        Syy = YY @ YY

        theta_mle = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
        kappa_mle = -(1 / dt) * np.log(
            (Sxy - theta_mle * Sx - theta_mle * Sy + N * theta_mle**2) / (Sxx - 2 * theta_mle * Sx + N * theta_mle**2)
        )
        sigma2_hat = (
            Syy
            - 2 * np.exp(-kappa_mle * dt) * Sxy
            + np.exp(-2 * kappa_mle * dt) * Sxx
            - 2 * theta_mle * (1 - np.exp(-kappa_mle * dt)) * (Sy - np.exp(-kappa_mle * dt) * Sx)
            + N * theta_mle**2 * (1 - np.exp(-kappa_mle * dt)) ** 2
        ) / N
        sigma_mle = np.sqrt(sigma2_hat * 2 * kappa_mle / (1 - np.exp(-2 * kappa_mle * dt)))

        t_data = T_vec
        x_data = np.array(wind.wind_spd)
        wind_re1 = [[t_data, x_data]]
        estimator = OrnsteinUhlenbeckEstimator(wind_re1, n_it=1)
        mle_2_results = {
            "mu": estimator.mu,
            "eta": estimator.eta,
            "sigma_sq": estimator.sigma_sq()
        }

        # Append results
        result = {
            "station_id": station_id,
            "theta_ols": theta_ols,
            "kappa_ols": kappa_ols,
            "sigma_ols": sig_ols,
            "theta_mle": theta_mle,
            "kappa_mle": kappa_mle,
            "sigma_mle": sigma_mle,
            "mle_2_mu": mle_2_results["mu"],
            "mle_2_eta": mle_2_results["eta"],
            "mle_2_sigma_sq": mle_2_results["sigma_sq"]
        }
        all_results.append(result)

    except Exception as e:
        print(f"Error processing station {station_id}: {e}")

# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")

# Save results to YAML
with open(output_yaml_path, 'w') as yaml_file:
    yaml.dump(all_results, yaml_file, default_flow_style=False)
print(f"Results saved to {output_yaml_path}")

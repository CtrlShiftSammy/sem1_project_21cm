import math
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# -----------------------
# 1. Generate noisy Gaussian data
# -----------------------
np.random.seed(69)

# True parameters
A_true = 50.0
mu_true = 0.0
sigma_true = 1.0
C_true = 1.0

# x grid and noisy data
x = np.linspace(-5, 5, 100)
y_clean = A_true * np.exp(-0.5 * ((x - mu_true)/sigma_true)**2) + C_true
y = y_clean + 1 * np.random.randn(len(x))  # add noise

# -----------------------
# 2. Define log-probability functions
# -----------------------

# Gaussian model
def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

# Prior
A_window = 10
mu_window = 5
sigma_window = 5
C_window = 5
def log_prior(theta):
    A, mu, sigma, C, ln_sigma = theta
    if A_true - A_window < A < A_true + A_window and mu_true - mu_window < mu < mu_true + mu_window and max(0.0, sigma_true - sigma_window) < sigma < (sigma_true + sigma_window) and C_true - C_window < C < C_true + C_window and -10 < ln_sigma < 1:
        return 0.0  # flat prior inside bounds
    return -np.inf  # impossible outside bounds

# Likelihood
def log_likelihood(theta, x, y):
    A, mu, sigma, C, ln_sigma = theta
    model = gaussian(x, A, mu, sigma, C)
    sigma2 = np.exp(2 * ln_sigma)  # variance from free noise parameter
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

# Posterior
def log_posterior(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)

# -----------------------
# 3. Run emcee
# -----------------------
ndim = 5  # A, mu, sigma, C, ln_sigma
nwalkers = 32
initial = np.array([A_true, mu_true, sigma_true, C_true, np.log(0.2)])  # true values
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y))
sampler.run_mcmc(pos, 5000, progress=True)

tau = sampler.get_autocorr_time()
print(f"Autocorrelation times: {tau}")
avg_tau = int(np.mean(tau))
print(f"Mean autocorrelation time: {avg_tau}")
# -----------------------
# 4. Analyze results
# -----------------------
flat_samples = sampler.get_chain(discard=10*avg_tau, thin=avg_tau // 2, flat=True)
labels = ["A", "mu", "sigma", "C", "ln_sigma"]

# Print parameter estimates
for i, label in enumerate(labels):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{label:8s} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")

# -----------------------
# 5. Plot results
# -----------------------
# Corner plot
fig = corner.corner(flat_samples, labels=labels,
                    truths=[A_true, mu_true, sigma_true, C_true, np.log(0.2)])
plt.show()

# Data + model fit
best_params = np.median(flat_samples, axis=0)
y_best = gaussian(x, *best_params[:4])

plt.errorbar(x, y, yerr=0.0, fmt=".k", label="data")
plt.plot(x, y_clean, "k--", label="true model")
plt.plot(x, y_best, "r", label="best fit (emcee)")
plt.legend()
plt.show()

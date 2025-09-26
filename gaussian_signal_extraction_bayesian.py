import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import numpy as np
import scienceplots
import os
import random
import tqdm
import emcee
import corner
np.random.seed(420) # the REAL answer to life, the universe, and everything :)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
plt.style.use('science')
plt.rcParams['text.usetex'] = False
warnings.filterwarnings("ignore", category=OptimizeWarning)
freq_file = 'freq_40_110.txt'
sky_temp_file = 'cosine_beam_sky_40_110MHz.txt'

freqs_sky = np.rint(np.loadtxt(freq_file)).astype(int)
temps_sky = np.loadtxt(sky_temp_file)

signals_dir = "simulated_signals"
# files = [f for f in os.listdir(signals_dir) if os.path.isfile(os.path.join(signals_dir, f))]
# if not files:
#     raise FileNotFoundError(f"No files found in signals_dir: {signals_dir}")
# signal_file = os.path.join(signals_dir, random.choice(files))
# print("Using randomly selected file:", signal_file)
# selected_template_name = os.path.basename(signal_file)
signal_file = os.path.join(signals_dir, "signal_fstar_0p1_vc_16p5_fx_1_tau_0p159596_alpha_1p25_nu_min_1p5_R_mfp_30.csv")
selected_template_name = os.path.basename(signal_file)

signals = np.loadtxt(signal_file, delimiter=',', skiprows=1)
signal_freqs = signals[:, 0]
signal_temps = signals[:, 1] / 1000
# Find common frequencies between freqs_sky and signal_freqs
common_freqs, sky_idx, sig_idx = np.intersect1d(freqs_sky, signal_freqs, return_indices=True)

# Build freq_sum (just the common frequencies)
freq_sum = common_freqs
signal_temps_c = signal_temps[sig_idx]
temps_sky_c = temps_sky[sky_idx]

# Build signal_sum (sum of both temps at the matching frequencies)
signal_sum = temps_sky_c + signal_temps_c


Tsys = signal_sum.copy()
tau = 1e6
delta_nu = 1e6

sigma_arr = Tsys / np.sqrt(tau * delta_nu)
noise = np.random.normal(0, sigma_arr, len(sigma_arr))
signal_sum += noise


def make_poly(n_terms, nu0=150.0):
    def poly(nu, T0, *coeffs):
        logv = np.log10((nu / nu0))
        expo = 0.0
        for i in range(n_terms):
            expo += coeffs[i] * (logv ** (i + 1))
        return T0 * 10 ** expo
    return poly

# def make_poly_plus_pert(n_terms, signal_vec, nu0=150.0):
#     base_poly = make_poly(n_terms, nu0)
#     def poly_plus_pert(nu, *params):
#         # params = [T0, a1..a_nterms, c]
#         *poly_params, c = params
#         return base_poly(nu, *poly_params) + c * signal_vec
#     return poly_plus_pert

def inverted_gaussian(nu, A, nu0, sigma):
    return A * np.exp(-0.5 * ((nu - nu0) / sigma) ** 2)

def make_poly_plus_pert(n_terms, nu0=150.0):
    base_poly = make_poly(n_terms, nu0)
    def poly_plus_pert(nu, *params):
        # params = [T0, a1..a_nterms, c]
        *poly_params, A, nu_c, sigma = params
        poly_part = base_poly(nu, *poly_params)
        gaussian_part = inverted_gaussian(nu, A, nu_c, sigma)
        return poly_part + gaussian_part
    return poly_plus_pert



T0_initial = 320.0

terms_list = np.arange(4, 11)
coeff_initials = [-2.54, -0.074, 0.013] + [0] * (max(terms_list) - 3)

yerr = sigma_arr


# Bayesian fit
x_g = freq_sum
y_g = signal_temps_c
yerr_gauss = np.full_like(y_g, max(np.std(y_g) / 50.0, 1e-6))


def log_prior_gauss(theta):
    A, nu_c, sigma = theta
    return 0.0

def log_likelihood_gauss(theta):
    A, nu_c, sigma = theta
    model = inverted_gaussian(x_g, A, nu_c, sigma)
    return -0.5 * np.sum(((y_g - model) / yerr_gauss) ** 2 + np.log(2 * np.pi * yerr_gauss ** 2))


def log_posterior_gauss(theta):
    lp = log_prior_gauss(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_gauss(theta)


# Initial guess from least squares fit
try:
    p0_gauss, _ = curve_fit(
        inverted_gaussian,
        x_g,
        y_g,
        p0=[np.min(y_g), x_g[np.argmin(y_g)], 10.0],
        maxfev=10000,
    )
except Exception:
    p0_gauss = np.array([np.min(y_g), x_g[np.argmin(y_g)], 10.0], dtype=float)

ndim_gauss = 3
nwalkers_gauss = 100
pos_gauss = p0_gauss + 1e-3 * np.random.randn(nwalkers_gauss, ndim_gauss)
# Enforce positive sigma in the initial positions
pos_gauss[:, 2] = np.abs(pos_gauss[:, 2]) + 1e-3

sampler_gauss = emcee.EnsembleSampler(nwalkers_gauss, ndim_gauss, log_posterior_gauss)
nsteps_gauss = 3000
print("Running Gaussian-only MCMC...")
sampler_gauss.run_mcmc(pos_gauss, nsteps_gauss, progress=True)

# Compute highest-probability parameters from the posterior
try:
    acor_gauss = sampler_gauss.get_autocorr_time(tol=0)
    burn_gauss = int(5 * np.mean(acor_gauss))
    thin_gauss = max(1, int(np.mean(acor_gauss) / 2))
except Exception:
    burn_gauss = nsteps_gauss // 2
    thin_gauss = 1

flat_samples_gauss = sampler_gauss.get_chain(discard=burn_gauss, thin=thin_gauss, flat=True)
log_probs_gauss = sampler_gauss.get_log_prob(discard=burn_gauss, thin=thin_gauss, flat=True)
best_idx_gauss = np.argmax(log_probs_gauss)
best_params_gauss = flat_samples_gauss[best_idx_gauss]
A_hp, nu_c_hp, sigma_hp = best_params_gauss
print(
    "Gaussian-only best-fit (highest prob): A={:.6f}, nu_c={:.6f}, sigma={:.6f}".format(
        A_hp, nu_c_hp, sigma_hp
    )
)

plt.figure(figsize=(10, 6))
plt.plot(x_g, y_g, 'k.', label='Signal template (noiseless)', alpha=0.5)
plt.plot(x_g, inverted_gaussian(x_g, A_hp, nu_c_hp, sigma_hp), 'r-', label='Gaussian fit (highest prob)', alpha=0.8)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Temperature perturbation (K)")
plt.title("Inverted Gaussian Fit to Signal Template")
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(10)
plt.close()


for n_terms in terms_list:
    poly_n = make_poly(n_terms)
    # Clean (foreground-only) fit to sky data
    p0_clean = [T0_initial] + coeff_initials[:n_terms]
    # popt_clean, pcov_clean = curve_fit(poly_n, freq_sum, temps_sky_c, p0=p0_clean, maxfev=10000)
    # residuals_clean = temps_sky_c - poly_n(freq_sum, *popt_clean)
    popt_clean, pcov_clean = curve_fit(poly_n, freq_sum, signal_sum, p0=p0_clean, maxfev=10000)
    residuals_clean = signal_sum - poly_n(freq_sum, *popt_clean)
    print(f"[n_terms={n_terms}]")
    print(f"Clean sky LS fit std dev: {np.std(residuals_clean)}")
    print(f"Clean sky LS fit params: {popt_clean}")

    ndim = n_terms + 4
    def log_prior(theta):
        *params, A, nu_c, sigma = theta
        return 0.0
    
    def log_likelihood(theta, x, y, yerr):
        params = theta
        model = poly_pp_n(x, *params)
        yerr = np.asarray(yerr, dtype=float)
        return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))

    def log_posterior(theta, x, y, yerr):
        return log_prior(theta) + log_likelihood(theta, x, y, yerr)


    poly_pp_n = make_poly_plus_pert(n_terms)
    print("Initial center for MCMC:", np.concatenate([popt_clean, [-0.15, 70, 10]]))
    initial_center = np.concatenate([popt_clean, [-0.15, 70, 10]])
    nwalkers = 200
    positions = initial_center + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(freq_sum, signal_sum, yerr))
    nsteps = 3000
    print("Running MCMC...")
    sampler.run_mcmc(positions, nsteps, progress=True)

    tau = sampler.get_autocorr_time(tol=0)
    burn = int(5 * np.mean(tau))
    # thin = max(1, int(np.mean(tau) / 2))
    thin = 1
    print("Autocorr times:", tau, " -> burn, thin:", burn, thin)

    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    labels = []
    for i in range(1 + n_terms):
        if i == 0:
            labels.append("T0")
        else:
            labels.append(f"a{i}")
    labels += ["A", "nu", "sigma"]
    fig = corner.corner(flat_samples, labels=labels, truths=None)
    # plt.show()
    plt.savefig("corner_plot_{}.png".format(n_terms))
    plt.close(fig)


    log_probs = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    # flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    best_idx = np.argmax(log_probs)
    best_params = flat_samples[best_idx]
    print(f"Perturbed sky bayesian fit params: {best_params}")
    residuals_total = signal_sum - poly_pp_n(freq_sum, *best_params)
    std_total = np.std(residuals_total)
    

    # Plot models from all walkers (use each walker's last-step position) as thin translucent black lines
    chain = sampler.get_chain()  # shape (nsteps, nwalkers, ndim)
    last_step = chain[-1]  # (nwalkers, ndim)
    # Plot original signal, Gaussian-only fit, and current iteration's Gaussian fit
    A_b, nu_b, sigma_b = best_params[-3:]

    all_walker_gaussian_params = np.array([walker_params[-3:] for walker_params in last_step])
    print(f"Shape of all_walker_gaussian_params: {all_walker_gaussian_params.shape}")
    all_walker_gaussians = np.array([inverted_gaussian(x_g, *params) for params in all_walker_gaussian_params])
    print(f"Shape of all_walker_gaussians: {all_walker_gaussians.shape}")

    mean_y = np.mean(all_walker_gaussians, axis=0)
    std_y = np.std(all_walker_gaussians, axis=0)

    within_1sigma_mask = (all_walker_gaussians >= (mean_y - std_y)) & (all_walker_gaussians <= (mean_y + std_y))
    y_within_1sigma = [all_walker_gaussians[within_1sigma_mask[:, i], i] for i in range(all_walker_gaussians.shape[1])]





    plt.figure(figsize=(10, 6))
    for walker_params in last_step:
        plt.plot(x_g, inverted_gaussian(x_g, *walker_params[-3:]), color='k', alpha=0.1)
    for i, x_val in enumerate(x_g):
        plt.scatter([x_val]*len(y_within_1sigma[i]), y_within_1sigma[i], color='lightblue', s=10, alpha=0.6, label='_nolegend_')


    plt.plot(x_g, y_g, 'k.', label='Inserted Signal Template (true)', alpha=0.5)
    plt.plot(x_g, inverted_gaussian(x_g, A_hp, nu_c_hp, sigma_hp), 'r-', label='Gaussian fit to template', alpha=0.8)
    plt.plot(x_g, inverted_gaussian(x_g, A_b, nu_b, sigma_b), 'b-', label=f'Highest probability Gaussian fit (n_terms={n_terms})', zorder=5)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Temperature perturbation (K)")
    plt.title(f"Inverted Gaussian comparison (n_terms={n_terms})")
    plt.legend()
    plt.grid(True)
    plt.savefig("gaussian_fit_comparison_{}.png".format(n_terms))
    # plt.show()
    plt.close()



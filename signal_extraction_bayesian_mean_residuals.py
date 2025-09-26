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

def make_poly_plus_pert(n_terms, signal_vec, nu0=150.0):
    base_poly = make_poly(n_terms, nu0)
    def poly_plus_pert(nu, *params):
        # params = [T0, a1..a_nterms, c]
        *poly_params, c = params
        return base_poly(nu, *poly_params) + c * signal_vec
    return poly_plus_pert

# Initial guesses for coefficients (subset used depending on n_terms)
T0_initial = 320.0

terms_list = np.arange(1, 11)
coeff_initials = [-2.54, -0.074, 0.013] + [0] * (len(terms_list) - 3)

yerr = sigma_arr

stds = []
scale_factors = []
scale_factor_errors = []
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

    ndim = n_terms + 2
    def log_prior(theta):
        *params, c = theta
        if not (-100 < c < 100):
            return -np.inf
        return 0.0
    
    def log_likelihood(theta, x, y, yerr):
        params = theta
        model = poly_pp_n(x, *params)
        yerr = np.asarray(yerr, dtype=float)
        return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))

    def log_posterior(theta, x, y, yerr):
        return log_prior(theta) + log_likelihood(theta, x, y, yerr)


    poly_pp_n = make_poly_plus_pert(n_terms, signal_temps_c)
    initial_center = np.concatenate([popt_clean, [0.0]])
    nwalkers = 1000
    positions = initial_center + 1e-4 * np.random.randn(nwalkers, ndim)
    positions[:, 1 + n_terms] = 0.0 + 0.1 * np.random.randn(nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(freq_sum, signal_sum, yerr))
    nsteps = 3000
    print("Running MCMC...")
    sampler.run_mcmc(positions, nsteps, progress=True)

    tau = sampler.get_autocorr_time(tol=0)
    burn = int(5 * np.mean(tau))
    thin = max(1, int(np.mean(tau) / 2))
    print("Autocorr times:", tau, " -> burn, thin:", burn, thin)

    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    idx_c = 1 + n_terms

    c_samples = flat_samples[:, idx_c]
    c_m = np.percentile(c_samples, [16, 50, 84])
    print("c (median, -err, +err):", c_m[1], c_m[1]-c_m[0], c_m[2]-c_m[1])

    scale_factors.append(c_m[1])
    scale_factor_errors.append(c_m[2] - c_m[1])

    labels = []
    for i in range(1 + n_terms):
        if i == 0:
            labels.append("T0")
        else:
            labels.append(f"a{i}")
    labels += ["c"]
    fig = corner.corner(flat_samples, labels=labels, truths=None)
    # plt.show()
    plt.savefig("corner_plot_{}_mean_residuals.png".format(n_terms))
    plt.close(fig)


    # log_probs = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    # # flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    # best_idx = np.argmax(log_probs)
    # best_params = flat_samples[best_idx]
    # print(f"Perturbed sky bayesian fit params: {best_params}")
    # residuals_total = signal_sum - poly_pp_n(freq_sum, *best_params)
    # std_total = np.std(residuals_total)
    # stds.append(std_total)

    
    log_probs = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    samples_subset = flat_samples[::10]

    # mean_params = np.mean(samples_subset, axis=0)
    # print(f"Mean parameters (every 10th sample): {mean_params}")
    # residuals_total = signal_sum - poly_pp_n(freq_sum, *mean_params)
    # std_total = np.std(residuals_total)
    # stds.append(std_total)
    plt.figure(figsize=(10, 6))
    residuals_subset = []
    for sample in samples_subset:
        residuals_total = signal_sum - poly_pp_n(freq_sum, *sample)
        plt.plot(freq_sum, poly_pp_n(freq_sum, *sample), color='gray', alpha=0.1)
        std_total = np.std(residuals_total)
        residuals_subset.append(std_total)

    stds.append(np.mean(residuals_subset))
    plt.show()
    plt.close()




    # # Total fit (foreground + scaled perturbation)
    # poly_pp_n = make_poly_plus_pert(n_terms, signal_temps_c)
    # p0_total = list(popt_clean) + [0.5]
    # lower_bound = [-np.inf] * (1 + n_terms) + [-np.inf]
    # upper_bound = [np.inf] * (1 + n_terms) + [np.inf]
    # popt_total, pcov_total = curve_fit(
    #     poly_pp_n, freq_sum, signal_sum,
    #     p0=p0_total, bounds=(lower_bound, upper_bound), absolute_sigma=True, maxfev=10000
    # )

    # residuals_total = signal_sum - poly_pp_n(freq_sum, *popt_total)
    # std_total = np.std(residuals_total)
    # stds.append(std_total)
    # scale_factors.append(popt_total[-1])
    # scale_factor_errors.append(np.sqrt(np.diag(pcov_total))[-1])

    # print(f"[n_terms={n_terms}] Total fit std dev of residuals: {std_total}")
    # print(f"  Fitted perturbation scale factor: {popt_total[-1]}")

# Plot std dev of residuals vs number of polynomial terms
plt.figure(figsize=(10, 6))
plt.plot(terms_list, stds, color="#0047AB", marker='o', label='Residual std dev')
plt.yscale('log')  # use base-10 log scale on y-axis
plt.xlabel("Number of polynomial terms")
plt.ylabel("Std. dev. of residuals")
plt.axhline(0.001, color='r', linestyle='--', label='Target threshold (1 mK)')
plt.grid(True)
plt.title("Residual Standard Deviation vs. Number of Polynomial Terms")
plt.legend()
plt.savefig("residuals_std_vs_poly_terms_mean_of_residuals.png", dpi=900)
print("Residuals standard deviation vs. polynomial terms plot saved.")
# plt.show()
plt.close()

plt.figure(figsize=(10, 6))
# plt.plot(terms_list, scale_factors, 'o-', label='Perturbation scale factor')
plt.errorbar(terms_list, scale_factors, yerr=scale_factor_errors, color="#0047AB", fmt='o-', label='Perturbation scale factor', capsize=5)
plt.xlabel("Number of polynomial terms")
plt.ylabel("Perturbation scale factor")
plt.title("Perturbation Scale Factor vs. Number of Polynomial Terms")
# plt.ylim(0 - 0.1 * np.std(scale_factors), 1 + 0.1 * np.std(scale_factors))
plt.ylim(-2.1, 3.1)
plt.legend()
plt.grid(True)
plt.savefig("perturbation_scale_factor_vs_poly_terms_mean_of_residuals.png", dpi=900)
print("Perturbation scale factor vs. polynomial terms plot saved.")
# plt.show()
plt.close()

# # Worker to fit the summed signal against one perturbation template (one file)
# def fit_template_worker(fname):
#     try:
#         cand_path = os.path.join(signals_dir, fname)
#         cand_signals = np.loadtxt(cand_path, delimiter=',', skiprows=1)
#         cand_freqs = cand_signals[:, 0]
#         cand_temps = cand_signals[:, 1] / 1000.0

#         # Align candidate template to the summed signal's frequency grid
#         common, idx_in_freq_sum, idx_in_cand = np.intersect1d(freq_sum, cand_freqs, return_indices=True)
#         if common.size < 10:
#             return fname, None  # skip if not enough points

#         x = freq_sum[idx_in_freq_sum]
#         sky_y = temps_sky_c[idx_in_freq_sum]
#         sum_y = signal_sum[idx_in_freq_sum]
#         template_vec = cand_temps[idx_in_cand]

#         sf_list = []
#         err_list = []
#         for n_terms in terms_list:
#             poly_n = make_poly(n_terms)
#             p0_clean = [T0_initial] + coeff_initials[:n_terms]
#             try:
#                 popt_clean, _ = curve_fit(poly_n, x, sky_y, p0=p0_clean, maxfev=10000)
#             except Exception:
#                 popt_clean = p0_clean  # fallback if clean fit fails

#             poly_pp_n = make_poly_plus_pert(n_terms, template_vec)
#             p0_total = list(popt_clean) + [0.5]
#             lower_bound = [-np.inf] * (1 + n_terms) + [-np.inf]
#             upper_bound = [np.inf] * (1 + n_terms) + [np.inf]

#             try:
#                 popt_total, pcov_total = curve_fit(
#                     poly_pp_n, x, sum_y,
#                     p0=p0_total, bounds=(lower_bound, upper_bound), absolute_sigma=True, maxfev=10000
#                 )
#                 c = popt_total[-1]
#                 c_err = np.sqrt(np.diag(pcov_total))[-1]
#             except Exception:
#                 c, c_err = np.nan, np.nan

#             sf_list.append(c)
#             err_list.append(c_err)

#         return fname, {
#             "scale_factors": np.array(sf_list),
#             "errors": np.array(err_list),
#         }
#     except Exception:
#         return fname, None

# # Fit the summed signal against perturbation templates from all files in simulated_signals
# # Parallelized with a progress bar
# template_results = {}
# # with ThreadPoolExecutor() as executor:
# #     futures = [executor.submit(fit_template_worker, fname) for fname in files]
# #     for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fitting templates"):
# #         fname, res = fut.result()
# #         if res is not None:
# #             template_results[fname] = res


# with ProcessPoolExecutor() as executor:
#     futures = [executor.submit(fit_template_worker, fname) for fname in files]
#     for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Fitting templates"):
#         fname, res = fut.result()
#         if res is not None:
#             template_results[fname] = res

# # Print which templates have scale factor = 1 ± (1σ error) at n_terms = 5
# n_terms_target = 5
# if n_terms_target in terms_list:
#     idx_target = list(terms_list).index(n_terms_target)
#     matching_files = []
#     for fname, res in template_results.items():
#         sf = res["scale_factors"][idx_target]
#         se = res["errors"][idx_target]

#         # Print the original template's scale factor and error when tested in this loop
#         if fname == selected_template_name:
#             print(f"Original template '{fname}': c={sf:.6f}, σ={se:.6f} at n_terms={n_terms_target}")
#             n_sigma = abs((sf - 1.0) / se) if np.isfinite(se) else 0.0
#             print(f"  This is {n_sigma:.3f}σ from 1.0")

#         if np.isfinite(sf) and np.isfinite(se) and abs(sf - 1.0) <= 1:#0.02 * se:
#             matching_files.append(fname)

#     # If the original template was not present in results, notify
#     if selected_template_name not in template_results:
#         print(f"Original template '{selected_template_name}' was not tested or returned no result.")

#     # print(f"Templates with scale factor within 1±10σ at n_terms={n_terms_target}:")
#     # for mf in matching_files:
#     #     print(" ", mf)
# else:
#     print(f"{n_terms_target} not in terms_list; cannot evaluate.")



# # Plot all matching templates on the same axes (scaled by their fitted factor) for the target n_terms
# if not matching_files:
#     print("No matching templates to plot.")
# else:
#     plt.figure(figsize=(10, 6))
#     # plot the summed signal for reference (use full freq_sum grid)
#     # plt.plot(freq_sum, signal_sum, color='k', linestyle='--', label='Summed signal', alpha=0.6)

#     for fname in matching_files:
#         path = os.path.join(signals_dir, fname)
#         data = np.loadtxt(path, delimiter=',', skiprows=1)
#         cand_freqs = data[:, 0]
#         cand_temps = data[:, 1] / 1000.0

#         # align to freq_sum
#         common, idx_in_freq_sum, idx_in_cand = np.intersect1d(freq_sum, cand_freqs, return_indices=True)
#         if common.size < 2:
#             continue

#         x = freq_sum[idx_in_freq_sum]
#         y = cand_temps[idx_in_cand]

#         # use the fitted scale factor at the target n_terms
#         sf = template_results[fname]["scale_factors"][idx_target]
#         # label and color
#         label = f"{fname} (c={sf:.3f})"
#         if fname == selected_template_name:
#             label = f"{fname} (c={sf:.3f}, original)"

#         if fname == selected_template_name:
#             plt.plot(x, y, label=label, alpha=1.0, color='red', linewidth=2, zorder=5)
#         else:
#             plt.plot(x, y, alpha=0.2, color='black', linewidth=0.5)

#     plt.xlabel("Frequency (MHz)")
#     plt.ylabel("Temperature perturbation (K)")
#     plt.title(f"Matching templates (n_terms={n_terms_target})")
#     plt.legend(loc='best', fontsize='small')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"matching_templates_n{n_terms_target}.png", dpi=300)
#     # plt.show()
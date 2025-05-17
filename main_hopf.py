# # import os
# # import copy
# # import gc
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from pathos.multiprocessing import ProcessPool
# #
# # from neuronumba.simulator.simulator import simulate_nodelay
# # from neuronumba.simulator.models.hopf import Hopf
# # from neuronumba.simulator.integrators.euler import EulerStochastic
# # from neuronumba.observables import FC
# # from neuronumba.observables.accumulators import AveragingAccumulator
# # from neuronumba.observables.measures import PearsonSimilarity
# # from neuronumba.tools.filters import BandPassFilter
# # from neuronumba.tools import filterps, hdf
# # from neuronumba.tools.loader import load_2d_matrix
# #
# # from DataLoaders.ADNI_A_Reparcellated import ADNI_A_Reparcellated
# # import pickle as pkl
# # from p_values import plotComparisonAcrossLabels2
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # CONFIGURATION
# # # ───────────────────────────────────────────────────────────────────────────────
# # OUT_ROOT = "Hopfs_Result"           # output directory
# # os.makedirs(OUT_ROOT, exist_ok=True)
# #
# # N_PROCS  = 5                        # number of parallel workers
# # GS       = np.arange(0.0, 6.1, 0.1) # grid of global coupling
# #
# # DT        = 0.1    # integration step in seconds
# # T_WARMUP  = 10     # warm-up time in seconds
# # T_MAX     = 220    # simulation time in seconds
# # SIGMA     = 1e-2   # noise intensity
# #
# # # band-pass for FC\BPF_K   = 2
# # BPF_K   = 2
# # BPF_LO  = 0.01
# # BPF_HI  = 0.1
# #
# # print(f"[DEBUG] CONFIG: N_PROCS={N_PROCS}, |GS|={len(GS)}, DT={DT}, T_WARMUP={T_WARMUP}, T_MAX={T_MAX}")
# #
# #
# # # Estimate each node's intrinsic frequency
# # def calc_H_freq(all_ts: dict, tr_s: float) -> np.ndarray:
# #     tr_ms = tr_s * 1000.0
# #     bpf = BandPassFilter(
# #         k=BPF_K, flp=BPF_LO, fhi=BPF_HI,
# #         tr=tr_ms, apply_detrend=True, apply_demean=True, remove_artifacts=False
# #     )
# #     filtered = {s: bpf.filter(ts) for s, ts in all_ts.items()}
# #     peak_freq = filterps.filt_pow_spetra_multiple_subjects(filtered, tr_ms)
# #     omega = 2 * np.pi * peak_freq
# #     print(f"[DEBUG] Obtained omega shape {omega.shape}")
# #     return omega
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # Simulation and downsampling
# # # ───────────────────────────────────────────────────────────────────────────────
# # def simulate(exec_env, g):
# #     model      = exec_env['model']
# #     weights    = exec_env['weights']
# #     obs_var    = exec_env['obs_var']
# #     t_warmup   = exec_env['t_warmup']
# #     t_max      = exec_env['t_max_neuronal']
# #     integrator = exec_env['integrator']
# #     samp       = exec_env['sampling_period']
# #     model.configure(weights=weights, g=g)
# #     return simulate_nodelay(
# #         model, integrator, weights, obs_var,
# #         sampling_period=samp,
# #         t_max_neuronal=t_max,
# #         t_warmup=t_warmup
# #     )
# #
# # def simulate_single_subject(exec_env, g):
# #     x   = simulate(exec_env, g)
# #     tr  = exec_env['tr']
# #     dt  = exec_env['sampling_period']
# #     n   = int(round(tr / dt))
# #     if n < 1:
# #         raise ValueError(f"TR {tr}s < dt {dt}s; need at least one sample per TR")
# #     T   = x.shape[0]
# #     pad = (-T) % n
# #     xp  = np.pad(x, ((0,pad),(0,0)), mode='constant', constant_values=np.nan)
# #     xp  = xp.reshape(xp.shape[0]//n, n, xp.shape[1])
# #     bds = np.nanmean(xp, axis=1)
# #     print(f"[DEBUG] simulate_single_subject g={exec_env['g']}: raw len={T}, pad={pad}, downsampled shape={bds.shape}")
# #     return bds
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # Process empirical signals
# # # ───────────────────────────────────────────────────────────────────────────────
# # def process_empirical_subjects(bold_signals, observables, bpf=None, verbose=False):
# #     nsub = len(bold_signals)
# #     N    = next(iter(bold_signals.values())).shape[1]
# #     vals = {k: acc.init(nsub, N) for k,(_,acc,_,_) in observables.items()}
# #     for i, ts in enumerate(bold_signals.values()):
# #         if verbose:
# #             print(f"[DEBUG] Empirical {i+1}/{nsub}, raw shape={ts.shape}")
# #         d = ts if bpf is None else bpf.filter(ts)
# #         for k,(meas,acc,_,_) in observables.items():
# #             M = meas.from_fmri(d)[k]
# #             vals[k] = acc.accumulate(vals[k], i, M)
# #     return {k: acc.postprocess(vals[k]) for k,(_,acc,_,_) in observables.items()}
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # Compute G sweep
# # # ───────────────────────────────────────────────────────────────────────────────
# # def eval_one_param(exec_env, g):
# #     sims = {}
# #     for r in range(exec_env['num_subjects']):
# #         sims[r] = simulate_single_subject(exec_env, g)
# #     meas = process_bold_signals(sims, exec_env)
# #     meas['g'] = g
# #     return meas
# #
# #
# # def compute_g(exec_env, g):
# #     print(f"[DEBUG] compute_g start g={g}, i={exec_env.get('i')}")
# #     out_fn = exec_env['out_file_name_pattern'].format(round(g,3))
# #     if not exec_env.get('force_recomputations', False) and os.path.exists(out_fn):
# #         sim = hdf.loadmat(out_fn)
# #         print(f"[DEBUG] Loaded existing sim for g={g}")
# #     else:
# #         sim = eval_one_param(exec_env, g)
# #         hdf.savemat(out_fn, sim)
# #     res = {k: dist.distance(sim[k], exec_env['processed'][k])
# #            for k,(_,_,dist,_) in exec_env['observables'].items()}
# #     res['g'] = g
# #     print(f"[DEBUG] compute_g end g={g}: FC={res['FC']:.4f}")
# #     return res
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # Per-subject fitting
# # # ───────────────────────────────────────────────────────────────────────────────
# # def prepro_G_Optim(sc_norm, fmri_ts, subj, tr_s, h_freq):
# #     model      = Hopf(a=-0.02, omega=h_freq)
# #     integrator = EulerStochastic(dt=DT, sigmas=np.r_[SIGMA, SIGMA])
# #     tr_ms = tr_s * 1000.0
# #     bpf   = BandPassFilter(k=BPF_K, flp=BPF_LO, fhi=BPF_HI, tr=tr_ms)
# #     obs   = {'FC': (FC(), AveragingAccumulator(), PearsonSimilarity(), bpf)}
# #
# #     out_d = os.path.join(OUT_ROOT, subj)
# #     os.makedirs(out_d, exist_ok=True)
# #     emp   = os.path.join(out_d, 'fNeuro_emp.mat')
# #     if not os.path.exists(emp):
# #         print(f"[DEBUG] Computing empirical FC for {subj}")
# #         proc = process_empirical_subjects(fmri_ts, obs, bpf=bpf, verbose=True)
# #         hdf.savemat(emp, proc)
# #     else:
# #         proc = {k: load_2d_matrix(emp, index=k) for k in obs}
# #
# #     jobs = []
# #     pat  = os.path.join(out_d, f"{subj}_fitting_g{{:.1f}}.mat")
# #     for i,G in enumerate(GS):
# #         jobs.append({
# #             'model': copy.deepcopy(model),
# #             'integrator': copy.deepcopy(integrator),
# #             'weights': sc_norm,
# #             'processed': proc,
# #             'tr': tr_s,
# #             'observables': obs,
# #             'obs_var': 'x',
# #             'bold': False,
# #             'out_file_name_pattern': pat,
# #             'num_subjects': 1,
# #             't_warmup': T_WARMUP,
# #             't_max_neuronal': T_MAX,
# #             'sampling_period': DT,
# #             'force_recomputations': False,
# #             'g': G,
# #             'verbose': True,
# #             'i': i
# #         })
# #
# #     print(f"[DEBUG] Launching pool.map with {N_PROCS} workers on {len(GS)} G-values")
# #     with ProcessPool(nodes=N_PROCS) as pool:
# #         results = pool.map(compute_g, jobs, GS)
# #
# #
# #     results = sorted(results, key=lambda r: r['g'])
# #     g_list = [r['g'] for r in results]
# #     fc_list = [r['FC'] for r in results]
# #     best = g_list[int(np.argmax(fc_list))]
# #     print(f"Best G for {subj} = {best}")
# #
# #     return {'g_list': g_list, 'FC': fc_list, 'FC_g': best, 'raw': results}
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # Plot & save per-subject
# # # ───────────────────────────────────────────────────────────────────────────────
# # def fit_and_plot(subj, fmri_ts, tr_s, sc_norm, h_freq):
# #     print(f"[>>>] Starting fit subj={subj}")
# #     out = prepro_G_Optim(sc_norm, fmri_ts, subj, tr_s, h_freq)
# #     d = os.path.join(OUT_ROOT, subj)
# #     with open(os.path.join(d, f"{subj}_results.pkl"), 'wb') as f:
# #         pkl.dump(out, f)
# #     plt.figure()
# #     plt.plot(out['g_list'], out['FC'], 'o-')
# #     plt.axvline(out['FC_g'], linestyle='--', color='r')
# #     plt.xlabel('G'); plt.ylabel('FC similarity')
# #     plt.title(f"Hopf fit: {subj}")
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(d, f"{subj}_fit.png")), plt.close()
# #
# # # ───────────────────────────────────────────────────────────────────────────────
# # # MAIN & GROUP SUMMARY
# # # ───────────────────────────────────────────────────────────────────────────────
# # if __name__=='__main__':
# #     DL    = ADNI_A_Reparcellated()
# #     sc    = DL.get_AvgSC_ctrl(normalized=True)
# #     tr    = DL.TR()
# #     subs  = list(DL.get_classification().keys())
# #
# #
# #     for s in subs:
# #         raw = DL.get_subjectData(s)[s]['timeseries']
# #         trans = raw.T
# #         print(f"  {s}: raw={raw.shape}, transposed={trans.shape}")
# #
# #     all_ts = {s: DL.get_subjectData(s)[s]['timeseries'].T for s in subs}
# #     h_freq = calc_H_freq(all_ts, tr)
# #
# #     for s in subs:
# #         raw = DL.get_subjectData(s)[s]['timeseries']
# #         fit_and_plot(s, {s: raw}, tr, sc, h_freq)
# #
# #     by_grp={'HC':[], 'MCI':[], 'AD':[]}
# #     for s,grp in DL.get_classification().items():
# #         p = os.path.join(OUT_ROOT, s, f"{s}_results.pkl")
# #         if not os.path.isfile(p): continue
# #         with open(p,'rb') as f: r = pkl.load(f)
# #         by_grp[grp].append(r['FC_g'])
# #     print("Group stats:")
# #     for g,v in by_grp.items(): print(f"{g}: mean={np.mean(v):.2f}, n={len(v)}")
# #
# #     plotComparisonAcrossLabels2(
# #         by_grp,
# #         columnLables=['HC','MCI','AD'],
# #         graphLabel="Best G by group",
# #         test='Mann-Whitney'
# #     )
# #     fp=os.path.join(OUT_ROOT,'group_best_G.png')
# #     plt.savefig(fp,dpi=300)
# #     print(f"Saved {fp}")
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# # NeuroNumba imports
# from neuronumba.simulator.models import Hopf
# from neuronumba.observables import FC
# from neuronumba.observables.linear.linearfc import LinearFC
# from neuronumba.tools import hdf
# from neuronumba.tools.filters import BandPassFilter
# from neuronumba.tools import filterps
# from neuronumba.simulator.simulator import simulate_nodelay
# from neuronumba.simulator.integrators import EulerStochastic
#
# from DataLoaders.ADNI_A_Reparcellated import ADNI_A_Reparcellated
# from DataLoaders.ADNI_A import ADNI_A
# #from Plotting.p_values import plotComparisonAcrossLabels2
# from p_values import plotComparisonAcrossLabels2
#
# # Use linear Hopf? Set to False to use non-linear Hopf
# USE_LINEAR = False
#
# # where all subjects’ results will go:
# OUT_ROOT = "Data_Produced/" + ("LinearHopf_Results" if USE_LINEAR else "NonLinearHopf_Results")
#
# # grid for G
# G_RANGE = np.arange(0.0, 6, 0.1)
# A_PARAM = -0.02
# SIGMA   = 1e-2
#
# # How many simulations to average each hopf FC computation
# HOPF_NUM_SIMS = 10
#
# # ===================== Normalize a SC matrix
# normalizationFactor = 0.2
# avgHuman66 = 0.0035127188987848714
# areasHuman66 = 66  # yeah, a bit redundant... ;-)
# maxNodeInput66 = 0.7275543904602363
# def correctSC(SC):
#     N = SC.shape[0]
#     logMatrix = np.log(SC+1)
#     # areasSC = logMatrix.shape[0]
#     # avgSC = np.average(logMatrix)
#     # === Normalization ===
#     # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
#     # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
#     maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
#     finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
#     return finalMatrix
#
# def upper_triangular_corr(fc1, fc2):
#     iu = np.triu_indices_from(fc1, k=1)
#     return np.corrcoef(fc1[iu], fc2[iu])[0,1]
#
# def compute_linear_hopf_fc(model, g, sc, tr_ms):
#     A = model.get_jacobian(g * sc)
#     lin_fc_obs = LinearFC()
#     lin_fc_obs.lyap_method = 'scipy' # Don't use slycot
#     Qn = model.get_noise_matrix(SIGMA, len(sc))
#     lin_fc = lin_fc_obs.from_matrix(A, Qn)['FC']
#     return lin_fc
#
# def compute_hopf_fc(model, g, sc, tr_ms):
#     tr_s = tr_ms / 1000.0
#     dt = 0.1
#     sampling_period = 1.0
#     t_max_neuronal = 220e3
#     t_warmup = 10e3
#
#     integrator = EulerStochastic(dt=dt, sigmas=np.r_[SIGMA, SIGMA])
#     obs_var = 'x'
#
#     model.configure(
#         weights=sc,
#         g=g
#     )
#
#     acc_res = None
#     for i in range(HOPF_NUM_SIMS):
#         signal = simulate_nodelay(model, integrator, sc, obs_var, sampling_period, t_max_neuronal, t_warmup)
#
#         if np.isnan(np.min(signal)):
#             print(f'Signal {i} has NaN')
#
#         # We need to convert signal to samples of size tr
#         n = int(tr_s / sampling_period)
#         l = signal.shape[0]
#         tmp1 = np.pad(signal, ((0, n - l % n), (0, 0)),
#                                     mode='constant',
#                                     constant_values=np.nan)
#         tmp2 = tmp1.reshape(n, int(tmp1.shape[0]/n), -1)
#         bds = np.nanmean(tmp2, axis=0)
#
#         # Once we have the simulated bold, we can compute its FC
#         result = FC().from_fmri(bds)
#
#         if acc_res is None:
#             acc_res = result['FC']
#         else:
#             acc_res += result['FC']
#
#     return acc_res / float(HOPF_NUM_SIMS)
#
# # If use_linear if False, then use normal Hopf simulation, else use the linear-Hopf
# def compare(use_linear=True):
#     os.makedirs(OUT_ROOT, exist_ok=True)
#
#     DL = ADNI_A_Reparcellated()
#     # DL = ADNI_A()
#     tr_ms = DL.TR() * 1000.0
#     subjects = list(DL.get_classification().keys())
#
#     # STEP 1: compute omegas pooled (you may reuse outside loop)
#     # if you already have h_freq from your full-Hopf code, you can import it
#     all_ts = {s: DL.get_subjectData(s)[s]['timeseries'].T for s in subjects}
#     bpf = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr_ms,
#                          apply_detrend=True, apply_demean=True, remove_artifacts=False)
#     filt_ts = {s: bpf.filter(ts) for s, ts in all_ts.items()}
#     h_freq = 2*np.pi * filterps.filt_pow_spetra_multiple_subjects(filt_ts, tr_ms)
#
#     # STEP 2: per‐subject G‐sweep
#     best_g = {}
#     for subj in subjects:
#         if use_linear:
#             print(f"Fitting Linear‐Hopf: {subj}")
#         else:
#             print(f"Fitting Non-Linear‐Hopf: {subj}")
#         subj_dir = os.path.join(OUT_ROOT, subj)
#         os.makedirs(subj_dir, exist_ok=True)
#
#         # empirical FC (no filtering)
#         data = DL.get_subjectData(subj)[subj]
#         ts = data['timeseries'].T
#         emp_fc = FC().from_fmri(ts)['FC']
#         # empirical SC
#         sc = data['SC']
#         # sc = sc / np.max(sc) * 0.1
#         sc = correctSC(sc)
#
#         scores = []
#
#         # save empirical FC once
#         hdf.savemat(os.path.join(subj_dir, f"{subj}_fNeuro_emp.mat"), {"FC": emp_fc})
#
#         # build Hopf‐Jacobian and noise once
#         model = Hopf(a=A_PARAM, omega=h_freq)
#
#         for G in G_RANGE:
#             # Compute the simluated FC (linear or non-linear)
#             if use_linear:
#                 sim_fc = compute_linear_hopf_fc(model, G, sc, tr_ms)
#             else:
#                 sim_fc = compute_hopf_fc(model, G, sc, tr_ms)
#
#             # save this .mat
#             fname = f"{subj}_fitting_g{G:.1f}.mat"
#             hdf.savemat(os.path.join(subj_dir, fname), {"FC": sim_fc})
#
#             scores.append(upper_triangular_corr(emp_fc, sim_fc))
#
#         scores = np.array(scores)
#         idx = np.argmax(scores)
#         gopt = float(G_RANGE[idx])
#         best_g[subj] = gopt
#         print(f"  → best G = {gopt:.2f}")
#
#         # plot
#         plt.figure(figsize=(5,4))
#         plt.plot(G_RANGE, scores, 'o-')
#         plt.axvline(gopt, linestyle='--')
#         plt.xlabel('G')
#         plt.ylabel('FC similarity (r)')
#         if use_linear:
#             plt.title(f'Linear‐Hopf fit: {subj}')
#         else:
#             plt.title(f'NonLinear‐Hopf fit: {subj}')
#         plt.tight_layout()
#         if use_linear:
#             plt.savefig(os.path.join(subj_dir, f"{subj}_linear_fit.png"))
#         else:
#             plt.savefig(os.path.join(subj_dir, f"{subj}_non_linear_fit.png"))
#         plt.close()
#
#     # STEP 4: group‐level summary
#     mapping = DL.get_classification()
#     by_group = {'HC':[], 'MCI':[], 'AD':[]}
#     for subj, grp in mapping.items():
#         if subj in best_g:
#             by_group[grp].append(best_g[subj])
#
#     print("\n=== Best‐G summary per group ===")
#     for grp, vals in by_group.items():
#         if vals:
#             print(f"{grp}: n={len(vals)}, mean={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")
#
#     plotComparisonAcrossLabels2(
#         by_group,
#         columnLables=['HC','MCI','AD'],
#         graphLabel="Linear‐Hopf best G by group" if use_linear else "NonLinear-Hopf best G by group",
#         test='Mann-Whitney',
#         comparisons_correction=None
#     )
#
# if __name__=="__main__":
#     compare(use_linear=USE_LINEAR)

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool

from neuronumba.simulator.models import Hopf
from neuronumba.observables import FC
from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools import filterps
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.simulator.integrators import EulerStochastic

from DataLoaders.ADNI_A_Reparcellated import ADNI_A_Reparcellated
from p_values import plotComparisonAcrossLabels2

# Config
USE_LINEAR    = False
G_RANGE       = np.arange(0.0, 0.501, 0.01)
A_PARAM       = -0.02
SIGMA         = 1e-2
HOPF_NUM_SIMS = 5
maxNodeInput66 = 0.7275543904602363

# Helpers
def correctSC(SC):
    logM = np.log(SC + 1)
    mni = np.max(np.sum(logM, axis=0))
    return logM * maxNodeInput66 / mni

def upper_triangular_corr(fc1, fc2):
    iu = np.triu_indices_from(fc1, k=1)
    return np.corrcoef(fc1[iu], fc2[iu])[0,1]

# Worker: compute or load one (subject, G)
def worker(subj, model_proto, G, sc, tr_ms, emp_fc, subj_dir, use_linear):

    fname = os.path.join(subj_dir, f"{subj}_fitting_g{G:.2f}.mat")
    if os.path.exists(fname):
        data = hdf.loadmat(fname)
        sim_fc = data.get('FC')
        if sim_fc is None:
            raise KeyError(f"'FC' not found in {fname}")

    else:
        model = copy.deepcopy(model_proto)
        if use_linear:
            A = model.get_jacobian(G * sc)
            lf = LinearFC(); lf.lyap_method = 'scipy'
            Qn = model.get_noise_matrix(SIGMA, len(sc))
            sim_fc = lf.from_matrix(A, Qn)['FC']
        else:
            tr_s = tr_ms / 1000.0
            integrator = EulerStochastic(dt=0.1, sigmas=np.r_[SIGMA, SIGMA])
            model.configure(weights=sc, g=G)
            acc = None
            for i in range(HOPF_NUM_SIMS):

                sig = simulate_nodelay(model, integrator, sc, 'x', 1.0, 220e3, 10e3)
                n = int(tr_s)
                pad = (-sig.shape[0]) % n
                xp = np.pad(sig, ((0, pad), (0, 0)), constant_values=np.nan)
                bds = np.nanmean(xp.reshape(-1, n, sig.shape[1]), axis=1)
                fc = FC().from_fmri(bds)['FC']
                acc = fc if acc is None else acc + fc
            sim_fc = acc / float(HOPF_NUM_SIMS)
        hdf.savemat(fname, {'FC': sim_fc})

    score = upper_triangular_corr(emp_fc, sim_fc)

    return (subj, G, score)

# Main compare: single pool using map
def compare(use_linear=USE_LINEAR, n_proc=None):
    DL = ADNI_A_Reparcellated()
    tr_ms = DL.TR() * 1000.0
    subjects = list(DL.get_classification().keys())

    print("[Main] Computing intrinsic frequencies...")
    all_ts = {s: DL.get_subjectData(s)[s]['timeseries'].T for s in subjects}
    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr_ms,
                         apply_detrend=True, apply_demean=True, remove_artifacts=False)
    filt_ts = {s: bpf.filter(ts) for s, ts in all_ts.items()}
    h_freq = 2 * np.pi * filterps.filt_pow_spetra_multiple_subjects(filt_ts, tr_ms)

    out_root = os.path.join('Data_Produced', 'LinearHopf_Results' if use_linear else 'NonLinearHopf_Results')
    os.makedirs(out_root, exist_ok=True)

    tasks = []
    for subj in subjects:
        print(f"[Main] Preparing subject: {subj}")
        data = DL.get_subjectData(subj)[subj]
        emp_fc = FC().from_fmri(data['timeseries'].T)['FC']
        subj_dir = os.path.join(out_root, subj)
        os.makedirs(subj_dir, exist_ok=True)
        hdf.savemat(os.path.join(subj_dir, f"{subj}_fNeuro_emp.mat"), {'FC': emp_fc})
        sc = correctSC(data['SC'])
        model_proto = Hopf(a=A_PARAM, omega=h_freq)
        for G in G_RANGE:
            tasks.append((subj, model_proto, G, sc, tr_ms, emp_fc, subj_dir, use_linear))

    proc_count = n_proc or os.cpu_count() or 1

    pool = ProcessPool(nodes=proc_count)
    results = pool.map(lambda args: worker(*args), tasks)
    pool.close(); pool.join()

    best_g = {}
    for subj in subjects:
        sub_res = sorted([r for r in results if r[0]==subj], key=lambda x: x[1])
        Gs, scs = zip(*[(r[1], r[2]) for r in sub_res])
        best_idx = np.nanargmax(scs)
        best_g[subj] = Gs[best_idx]
        plt.figure()
        plt.plot(Gs, scs, 'o-')
        plt.axvline(Gs[best_idx], ls='--')
        plt.title(f"{'Linear' if use_linear else 'NonLinear'}-Hopf: {subj}")
        plt.xlabel('G'); plt.ylabel('r')
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, subj, f"{subj}_fit.png"))
        plt.close()

    print("[Main] Computing group statistics...")
    mapping = DL.get_classification()
    by_group = {'HC': [], 'MCI': [], 'AD': []}
    for subj, grp in mapping.items():
        if subj in best_g:
            by_group[grp].append(best_g[subj])

    print("\n=== Best‐G summary per group ===")

    for grp, vals in by_group.items():
        if vals:
            print(f"{grp}: n={len(vals)}, mean={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    plotComparisonAcrossLabels2(by_group, columnLables=['HC','MCI','AD'],
        graphLabel=("Linear-Hopf best G" if use_linear else "NonLinear-Hopf best G"),
        test='Mann-Whitney', comparisons_correction=None)
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "group_best_G.png"))

if __name__=='__main__':
    compare()

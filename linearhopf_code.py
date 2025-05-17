import os
import numpy as np
import matplotlib.pyplot as plt

# NeuroNumba imports
from neuronumba.simulator.models import Hopf
from neuronumba.observables import FC
from neuronumba.observables.linear.linearfc import LinearFC
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools import filterps
from neuronumba.simulator.simulator import simulate_nodelay
from neuronumba.simulator.integrators import EulerStochastic

from DataLoaders.ADNI_A_Reparcellated import ADNI_A_Reparcellated
#from DataLoaders.ADNI_A import ADNI_A
#from Plotting.p_values import plotComparisonAcrossLabels2
from p_values import plotComparisonAcrossLabels2
# Use linear Hopf? Set to False to use non-linear Hopf
USE_LINEAR = True

# where all subjects’ results will go:
OUT_ROOT = "Data_Produced/" + ("LinearHopf_Results" if USE_LINEAR else "NonLinearHopf_Results")

# grid for G
G_RANGE = np.arange(0.0, 0.501, 0.01)
A_PARAM = -0.02
SIGMA   = 1e-2

# How many simulations to average each hopf FC computation
HOPF_NUM_SIMS = 10

# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix

def upper_triangular_corr(fc1, fc2):
    iu = np.triu_indices_from(fc1, k=1)
    return np.corrcoef(fc1[iu], fc2[iu])[0,1]

def compute_linear_hopf_fc(model, g, sc, tr_ms):
    A = model.get_jacobian(g * sc)
    lin_fc_obs = LinearFC()
    lin_fc_obs.lyap_method = 'scipy' # Don't use slycot
    Qn = model.get_noise_matrix(SIGMA, len(sc))
    lin_fc = lin_fc_obs.from_matrix(A, Qn)['FC']
    return lin_fc

def compute_hopf_fc(model, g, sc, tr_ms):
    tr_s = tr_ms / 1000.0
    dt = 0.1
    sampling_period = 1.0
    t_max_neuronal = 220e3
    t_warmup = 10e3

    integrator = EulerStochastic(dt=dt, sigmas=np.r_[SIGMA, SIGMA])
    obs_var = 'x'

    model.configure(
        weights=sc,
        g=g
    )

    acc_res = None
    for i in range(HOPF_NUM_SIMS):
        signal = simulate_nodelay(model, integrator, sc, obs_var, sampling_period, t_max_neuronal, t_warmup)

        if np.isnan(np.min(signal)):
            print(f'Signal {i} has NaN')

        # We need to convert signal to samples of size tr
        n = int(tr_s / sampling_period)
        l = signal.shape[0]
        tmp1 = np.pad(signal, ((0, n - l % n), (0, 0)),
                                    mode='constant',
                                    constant_values=np.nan)
        tmp2 = tmp1.reshape(n, int(tmp1.shape[0]/n), -1)
        bds = np.nanmean(tmp2, axis=0)

        # Once we have the simulated bold, we can compute its FC
        result = FC().from_fmri(bds)

        if acc_res is None:
            acc_res = result['FC']
        else:
            acc_res += result['FC']

    return acc_res / float(HOPF_NUM_SIMS)

# If use_linear if False, then use normal Hopf simulation, else use the linear-Hopf
def compare(use_linear=True):
    os.makedirs(OUT_ROOT, exist_ok=True)

    DL = ADNI_A_Reparcellated()
    # DL = ADNI_A()
    tr_ms = DL.TR() * 1000.0
    subjects = list(DL.get_classification().keys())

    # STEP 1: compute omegas pooled (you may reuse outside loop)
    # if you already have h_freq from your full-Hopf code, you can import it
    all_ts = {s: DL.get_subjectData(s)[s]['timeseries'].T for s in subjects}
    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr_ms,
                         apply_detrend=True, apply_demean=True, remove_artifacts=False)
    filt_ts = {s: bpf.filter(ts) for s, ts in all_ts.items()}
    h_freq = 2*np.pi * filterps.filt_pow_spetra_multiple_subjects(filt_ts, tr_ms)

    # STEP 2: per‐subject G‐sweep
    best_g = {}
    for subj in subjects:
        if use_linear:
            print(f"Fitting Linear‐Hopf: {subj}")
        else:
            print(f"Fitting Non-Linear‐Hopf: {subj}")
        subj_dir = os.path.join(OUT_ROOT, subj)
        os.makedirs(subj_dir, exist_ok=True)

        # empirical FC (no filtering)
        data = DL.get_subjectData(subj)[subj]
        ts = data['timeseries'].T
        emp_fc = FC().from_fmri(ts)['FC']
        # empirical SC
        sc = data['SC']
        # sc = sc / np.max(sc) * 0.1
        sc = correctSC(sc)

        scores = []

        # save empirical FC once
        hdf.savemat(os.path.join(subj_dir, f"{subj}_fNeuro_emp.mat"), {"FC": emp_fc})

        # build Hopf‐Jacobian and noise once
        model = Hopf(a=A_PARAM, omega=h_freq)

        for G in G_RANGE:
            # Compute the simluated FC (linear or non-linear)
            if use_linear:
                sim_fc = compute_linear_hopf_fc(model, G, sc, tr_ms)
            else:
                sim_fc = compute_hopf_fc(model, G, sc, tr_ms)

            # save this .mat
            fname = f"{subj}_fitting_g{G:.2f}.mat"
            hdf.savemat(os.path.join(subj_dir, fname), {"FC": sim_fc})

            scores.append(upper_triangular_corr(emp_fc, sim_fc))

        scores = np.array(scores)
        idx = np.argmax(scores)
        gopt = float(G_RANGE[idx])
        best_g[subj] = gopt
        print(f"  → best G = {gopt:.2f}")

        # plot
        plt.figure(figsize=(5,4))
        plt.plot(G_RANGE, scores, 'o-')
        plt.axvline(gopt, linestyle='--')
        plt.xlabel('G')
        plt.ylabel('FC similarity (r)')
        if use_linear:
            plt.title(f'Linear‐Hopf fit: {subj}')
        else:
            plt.title(f'NonLinear‐Hopf fit: {subj}')
        plt.tight_layout()
        if use_linear:
            plt.savefig(os.path.join(subj_dir, f"{subj}_linear_fit.png"))
        else:
            plt.savefig(os.path.join(subj_dir, f"{subj}_non_linear_fit.png"))
        plt.close()

    # STEP 4: group‐level summary
    mapping = DL.get_classification()
    by_group = {'HC':[], 'MCI':[], 'AD':[]}
    for subj, grp in mapping.items():
        if subj in best_g:
            by_group[grp].append(best_g[subj])

    print("\n=== Best‐G summary per group ===")
    for grp, vals in by_group.items():
        if vals:
            print(f"{grp}: n={len(vals)}, mean={np.mean(vals):.2f}, min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    plotComparisonAcrossLabels2(
        by_group,
        columnLables=['HC','MCI','AD'],
        graphLabel="Linear‐Hopf best G by group" if use_linear else "NonLinear-Hopf best G by group",
        test='Mann-Whitney',
        comparisons_correction=None
    )

if __name__=="__main__":

    compare(use_linear=USE_LINEAR)
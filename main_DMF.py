import copy
import os

import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool

from examples.global_coupling_fitting import compute_g, process_empirical_subjects
from neuronumba.simulator.models import Deco2014
from neuronumba.bold.stephan_2008 import BoldStephan2008
from neuronumba.observables import FC
from neuronumba.observables.accumulators import ConcatenatingAccumulator, AveragingAccumulator
from neuronumba.observables.measures import KolmogorovSmirnovStatistic, PearsonSimilarity
from neuronumba.observables.sw_fcd import SwFCD
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.tools import hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix
import glob
from DataLoaders.ADNI_A_Reparcellated import ADNI_A_Reparcellated
# from DataLoaders.ADNI_A import ADNI_A
from p_values import plotComparisonAcrossLabels2

out_file_path = "Results_DMF"
OUT_GROUP_PNG = os.path.join(out_file_path, "DMF_group_bestG.png")
OUT_GROUP_PKL = os.path.join(out_file_path, "DMF_bestG_by_subject.pkl")

def prepro_G_Optim(sc_norm, all_fMRI, name, tr_):
    print(f'Simulating {name}')
    # J_file_name_pattern = os.path.join(out_file_path, "FIC", "BenjiBalancedWeights-{}.mat")

    # %%%%%%%%%%%%%%% Set General Model Parameters
    # ------------- Simulation parameters
    wStart = 0.; wEnd = 5.5; wStep = 0.1  # 0.05 ???
    gs = np.arange(wStart, wEnd, wStep)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    # ------------- Integration parameters
    dt = 0.1  # milliseconds
    sampling_period = 1.0  # Sampling period from the raw signal data (ms)
    n_sim_subj = 10  # rule of thumb: same as all_fMRI -> 10 at least !!!
    t_max_neuronal = 220 * 1000  # milliseconds
    t_warmup = 10 * 1000
    tr = tr_ * 1000  # neuronumba works in milliseconds

    # ------------- Model Parameters
    model = Deco2014(auto_fic=True)
    sigma = 0.01
    integrator = EulerStochastic(dt=dt, sigmas=np.r_[sigma, sigma])
    bold = True
    obs_var = 're'

    # ------------- signal processing and observable parameters
    bpf = BandPassFilter(k=2, flp=0.02, fhi=0.1, tr=tr)
    observables = {'FC': (FC(), AveragingAccumulator(), PearsonSimilarity(), bpf),
                   'swFCD': (SwFCD(), ConcatenatingAccumulator(), KolmogorovSmirnovStatistic(), bpf)}

    # ------------- paths, directories and filenames
    out_folder_name_pattern = os.path.join(out_file_path, name)
    if not os.path.exists(out_folder_name_pattern):
        os.makedirs(out_folder_name_pattern)
    out_file_name_pattern = os.path.join(out_file_path, name, 'fitting_g{}.mat')
    emp_filename = os.path.join(out_file_path, name, 'fNeuro_emp.mat')

    # ------------- pre-process empirical data
    if not os.path.exists(emp_filename):
        # bpf_emp = BandPassFilter(k=2, flp=0.01, fhi=0.09, tr=tr, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects(all_fMRI, observables, bpf=bpf)
        hdf.savemat(emp_filename, processed)
    else:
        processed = {o: load_2d_matrix(emp_filename, index=o) for o in observables.keys()}

    # ------------- and simulate!!!!
    pool = ProcessPool(nodes=3)
    rn = list(range(len(gs)))
    # Not entirely sure that the deepcopy() function is needed, but I use it when the object is going to be accessed
    # in read-write mode.
    ee = [{
        'verbose': True,
        'i': i,
        'model': copy.deepcopy(model),
        'integrator': copy.deepcopy(integrator),
        'weights': sc_norm,
        'processed': processed,
        'tr': tr,
        'observables': copy.deepcopy(observables),
        'obs_var': obs_var,
        'bold': bold,
        'bold_model': BoldStephan2008().configure(),
        'out_file_name_pattern': out_file_name_pattern,
        # 'J_file_name_pattern': J_file_name_pattern,
        'num_subjects': n_sim_subj,
        't_max_neuronal': t_max_neuronal,
        't_warmup': t_warmup,
        'sampling_period': sampling_period,
        'force_recomputations': False,
    } for i, _ in enumerate(rn)]

    results = pool.map(compute_g, ee, gs)

    # ------------- plot!
    # fig, ax = plt.subplots()
    rs = sorted(results, key=lambda r: r['g'])
    g = [r['g'] for r in rs]
    res = {}
    for o in observables.keys():
        data = [r[o] for r in rs]
        if o == 'FC':
            res[o] = np.max(data)
        else:
            res[o] = np.max(data)
        # ax.plot(g, data, label=o)
    # ax.legend()
    # ax.set(xlabel=f'G (global coupling)',
    #        title='fitting')
    # plt.show()
    return res

def find_subject_dirs(base_dir):
    """Return list of subdirectories (subjects) under base_dir."""
    return [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
from neuronumba.tools import hdf

def extract_fc_from_mat(mat_path):
    """
    Reads a .mat saved by hdf.savemat(...) containing a full FC matrix under key 'FC',
    then returns the mean of its upper‐triangle (off‐diagonal) entries as a single scalar.
    """
    data = hdf.loadmat(mat_path)
    if 'FC' not in data:
        raise KeyError(f"'FC' key not found in {mat_path}")
    fc_mat = np.array(data['FC'])
    # Ensure it's square:
    if fc_mat.ndim != 2 or fc_mat.shape[0] != fc_mat.shape[1]:
        raise ValueError(f"FC matrix in {mat_path} is not square: shape={fc_mat.shape}")
    # Extract strict upper‐triangle (k=1):
    iu = np.triu_indices(fc_mat.shape[0], k=1)
    return float(fc_mat[iu].mean())

def run():
    DL = ADNI_A_Reparcellated()
    # DL = ADNI_A()
    C = DL.get_AvgSC_ctrl(normalized=True)

    # all_data = DL.get_fullGroup_data('HC')
    # tc = {s: all_data[s]['timeseries'] for s in all_data}
    # prepro_G_Optim(C, tc, 'HG_Group', DL.TR())

    # ----------- subject-level simulation
    res = {}
    for subject in DL.get_classification():
        tc = {subject: DL.get_subjectData(subject)[subject]['timeseries']}
        name = subject
        res[subject] = prepro_G_Optim(C, tc, name, DL.TR())

    out_folder_name_pattern = os.path.join(out_file_path, 'fitting_G')
    fileObject = open(out_folder_name_pattern, 'wb')
    pkl.dump(res, fileObject)
    fileObject.close()

    # --- PER‐SUBJECT PROCESSING -----------------------------------------------
    best_g = {}
    for subj_dir in find_subject_dirs(out_file_path):
        subj = os.path.basename(subj_dir)
        # gather all fitting_g*.mat files
        mat_files = sorted(glob.glob(os.path.join(subj_dir, "fitting_g*.mat")))
        if not mat_files:
            print(f"[!] no fitting files for {subj}, skipping")
            continue

        g_list = []
        fc_list = []
        for mf in mat_files:
            # parse G from filename: fitting_g{G:.1f}.mat
            fname = os.path.basename(mf)
            g_str = fname.replace("fitting_g", "").replace(".mat", "")
            try:
                G = float(g_str)
            except ValueError:
                continue
            fc_val = extract_fc_from_mat(mf)
            g_list.append(G)
            fc_list.append(fc_val)

        g_arr = np.array(g_list)
        fc_arr = np.array(fc_list)

        # find best G
        idx_best = np.nanargmax(fc_arr)
        g_opt = g_arr[idx_best]

        best_g[subj] = g_opt

        # plot per‐subject
        plt.figure(figsize=(5, 4))
        plt.plot(g_arr, fc_arr, 'o-', label="FC")
        plt.axvline(g_opt, linestyle='--', color='r',
                    label=f"best G={g_opt:.2f}")
        plt.title(f"DMF fit: {subj}")
        plt.xlabel("Global coupling G")
        plt.ylabel("Pearson FC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, f"{subj}_DMF_fit.png"))
        plt.close()

    # --- GROUP‐LEVEL COMPARISON -----------------------------------------------
    # save best‐G dict for record
    with open(OUT_GROUP_PKL, 'wb') as f:
        pkl.dump(best_g, f)

    # build per‐group lists
    DL = ADNI_A_Reparcellated()
    mapping = DL.get_classification()  # subj -> 'HC'|'MCI'|'AD'
    by_group = {'HC': [], 'MCI': [], 'AD': []}
    for subj, grp in mapping.items():
        if subj in best_g:
            by_group[grp].append(best_g[subj])

    # print summary
    print("\nDMF: Best‐G summary by group:")
    for grp, vals in by_group.items():
        if vals:
            print(f"  {grp}: n={len(vals)}, mean={np.mean(vals):.2f}, "
                  f"min={np.min(vals):.2f}, max={np.max(vals):.2f}")

    # final group plot
    plotComparisonAcrossLabels2(
        by_group,
        columnLables=['HC', 'MCI', 'AD'],
        graphLabel="DMF best G by diagnostic group",
        test='Mann-Whitney',
        comparisons_correction=None
    )
    plt.tight_layout()
    plt.savefig(OUT_GROUP_PNG, dpi=300)
    plt.close()

    print(f"\nSaved group‐comparison plot to {OUT_GROUP_PNG}")
    print(f"Saved best‐G values to {OUT_GROUP_PKL}")



# ================================================================================================================
if __name__ == "__main__":
    # plt.ion()  # Activate interactive mode
    run()
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
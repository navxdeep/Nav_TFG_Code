# --------------------------------------------------------------------------------------
# Full code for loading the reparcellated data (in the Schaefer2018 parcellation 100)
# Subjects: HC 17, MCI 9 - RoIs: 10 - TR = 3 - timepoints: 197
# Info for each subject: timeseries
#
# Code by Gustavo Patow
# Note: We have 36 subjects and 100 areas
# --------------------------------------------------------------------------------------
import numpy as np

from DataLoaders.baseDataLoader import DataLoader
#from DataLoaders.Parcellations import Schaefer2018


# ==========================================================================
# Important config options: filenames
# ==========================================================================
from DataLoaders.WorkBrainFolder import *
base_folder = WorkBrainDataFolder + "ADNI-A/"


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class ADNI_A_Reparcellated(DataLoader):
    def __init__(self, path=None,
                 ):
        self.groups = ['HC', 'MCI', 'AD']
        self.file_path = base_folder + 'Reparcellated/'
        self.data = {}
        self.__loadAllData()

    def __loadAllData(self):
        import pandas as pd
        df = pd.read_csv(self.file_path + 'subjects.csv', header=None)[[0,1]]
        df.columns = ['subj', 'group']
        df = df[df['group'] != 'SMC']
        self.classification = df.set_index('subj').to_dict()['group']

        for subj in self.classification:
            d = np.load(self.file_path + f'{subj}.npz')
            self.data[subj] = {}
            for k in d.files:
                self.data[subj][k] = d[k]

    def name(self):
        return 'ADNI_A_Reparcellated'

    def TR(self):
        return 3  # Repetition Time (seconds)

    def N(self):
        return 100

    def get_classification(self):
        return self.classification

    def get_subjectData(self, subjectID):
        return {subjectID: self.data[subjectID]}

    #def get_parcellation(self):
        #return Schaefer2018.Schaefer2018(N=self.N(), normalization=1, RSN=7)
    def get_AvgSC_ctrl(self, normalized=True, scaling_factor=0.2):
        """
        Compute the average Structural Connectivity (SC) matrix over healthy controls.

        Parameters:
          normalized (bool): If True, scales the average SC matrix so that
                             its maximum equals scaling_factor.
          scaling_factor (float): The factor to scale the SC matrix if normalized.

        Returns:
          avg_sc (np.ndarray): The average (and optionally normalized) SC matrix.
        """
        # Get healthy control subjects.
        hc_subjects = [subj for subj in self.classification if self.classification[subj] == 'HC']
        if len(hc_subjects) == 0:
            raise ValueError("No healthy control subjects found!")
        sc_sum = None
        count = 0
        for subj in hc_subjects:
            if "SC" in self.data[subj]:
                sc = self.data[subj]["SC"]
                sc = sc.astype(np.float64)
                if sc_sum is None:
                    sc_sum = sc
                else:
                    sc_sum += sc
                count += 1
        if count == 0:
            raise ValueError("No SC data found for healthy controls!")
        avg_sc = sc_sum / count
        if normalized:
            avg_sc = scaling_factor * avg_sc / np.max(avg_sc)
        return avg_sc


# ================================================================================================================
print('Data loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_A_Reparcellated()
    sujes = DL.get_classification()
    print(f'Classification: {sujes}')
    print(f'Group labels: {DL.get_groupLabels()}')
    gMCI = DL.get_groupSubjects('MCI')
    s1 = DL.get_subjectData('003_S_6067')  # HC
    s2 = DL.get_subjectData('022_S_5004')  # MCI
    s3 = DL.get_subjectData('041_S_4974')  # AD
    avgSC = DL.get_AvgSC_ctrl()
    print('done! ;-)')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
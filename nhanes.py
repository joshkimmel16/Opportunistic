import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection
from random import choices

class FeatureColumn:
    def __init__(self, category, field, preprocessors, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessors = preprocessors
        self.args = args
        self.data = None
        self.cost = cost

class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns # Depricated
        self.dataset = None # Depricated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                #df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process' + field)
            df.append(df_col)
        df = pd.concat(df, axis=1)
        #df = pd.merge(df, df_sel, how='outer')
        # do preprocessing steps
        df_proc = []#[df['SEQN']]
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessors is not None:
                prepr_col = df[field]
                for x in range(len(fe_col.preprocessors)):
                    prepr_col = fe_col.preprocessors[x](prepr_col, fe_col.args[x])
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset
    
    
# Preprocessing functions
def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_map_mode(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    df_col[df_col > args['cutoff']] = np.nan
    df_col[pd.isna(df_col)] = df_col.mode()
    return df_col

def preproc_map_median(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    df_col[df_col > args['cutoff']] = np.nan
    df_col[pd.isna(df_col)] = df_col.median()
    return df_col

def preproc_real(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical normalization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_real_mapped(df_col, args):
    if args is None or not 'mappedValue' in args:
        raise Exception("Missing required argument 'mappedValue'!")
    if not 'cutoff' in args:
        args['cutoff'] = np.inf
    if not 'normalize' in args:
        args['normalize'] = True
        
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mappedValue
    df_col[pd.isna(df_col)] = args['mappedValue']
    # statistical normalization
    if args['normalize'] == True:
        df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_real_transform(df_col, args):
    if args is None or not 'toTransform' in args:
        raise Exception("Missing required argument 'toTransform'!")
    if not 'cutoff' in args:
        args['cutoff'] = np.inf
    if not 'useOldMean' in args:
        args['useOldMean'] = True
    if not 'normalize' in args:
        args['normalize'] = True
        
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    old_mean = df_col.mean()
    
    # replace certain values with other values
    for x in args['toTransform']:
        df_col[df_col == x['selectValue']] = x['mappedValue']
    
    if 'mappedValue' in args:
        # nan replaced by mappedValue
        df_col[pd.isna(df_col)] = args['mappedValue']
    else:
        # nan replaced by mean
        df_col[pd.isna(df_col)] = df_col.mean() if args['useOldMean'] == False else old_mean
    # statistical normalization
    if args['normalize'] == True:
        df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col
    
def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col

def preproc_cut(df_col, bins):
    # limit values to the bins range
    #df_col = df_col[df_col >= bins[0]]
    #df_col = df_col[df_col <= bins[-1]]
    #return pd.cut(df_col.iloc[:,0], bins, labels=False)
    return pd.cut(df_col, bins, labels=False)

def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col

def preproc_cutoff(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    df_col[df_col > args['cutoff']] = np.nan
    return df_col

def preproc_probfill(df_col, args=None):
    tmp = df_col.dropna(axis=0, how='any')
    prob = tmp.value_counts(normalize=True)
    c = df_col.isna().sum()
    new_vals = choices(prob.index.tolist(), prob.values.tolist(), k=c)
    df_col[pd.isna(df_col)] = new_vals
    return df_col

#### Add your own preprocessing functions ####

# Dataset loader
class Dataset():
    """ 
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.costs = None
        
    def load_cancer(self, opts=None):
        columns = [
            # TARGET: cancer/malignance diagnosis
            FeatureColumn('Questionnaire', 'MCQ220', 
                                    None, None),
            
            ### DEMOGRAPHIC INFO - POTENTIAL CLUSTERING TARGET
            
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 [preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 2, 'normalize': False}, None]),
            
            # Age at time of screening (in years)
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 [preproc_real], [{'cutoff':85}]),
            
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH3', 
                                 [preproc_real_mapped], [{'mappedValue': 3, 'normalize': False}]),
            
            # Citizenship status
            FeatureColumn('Demographics', 'DMDCITZN', 
                                 [preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 1, 'normalize': False}, None]),
            
            # Annual household income
            #FeatureColumn('Demographics', 'INDHHINC', 
                                 #[preproc_real_transform, preproc_map_median], [{'toTransform': [{'selectValue': 13, 'mappedValue': 3}], 'normalize': False}, {'cutoff': 11}]),
            FeatureColumn('Demographics', 'INDHHINC', 
                                 [preproc_real_transform, preproc_cutoff, preproc_probfill], [{'toTransform': [{'selectValue': 13, 'mappedValue': 3}], 'normalize': False}, {'cutoff': 11}, None]),
            
            # Education level
            #FeatureColumn('Demographics', 'DMDEDUC2', 
                                 #[preproc_map_median], [{'cutoff':5}]),
            # LESS THAN 50% OF DATA PRESENT
            FeatureColumn('Demographics', 'DMDEDUC2', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff':5}, None]),
            
            # Total # of People in the Household
            FeatureColumn('Demographics', 'DMDHHSIZ', 
                                 [preproc_map_mode], [{'cutoff':7}]),
                                 
            ### EXAMINATION INFO - POTENTIAL CLUSTERING TARGET
            
            ## Potential Targets: BIX, BMX, BPX, CVX, DEX, DXX, MGX, PAXRAW
            ## Non-Targets: ARX, AUX.., BAX, CSX, DXXAG, DXXFEM, DXXFRX, DXXSPN, DXXV, DXXVFA, ENX, LEXAB, LEXABPI, LEXPN, MSX, OHX..., OPX...
            
            # Body Fat Percentage
            # LESS THAN 50% OF DATA PRESENT
            FeatureColumn('Examination', 'BIDPFAT', 
                                 [preproc_real], [None]),
            # BMI
            FeatureColumn('Examination', 'BMXBMI', 
                                 [preproc_real], [None]),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST', 
                                 [preproc_real], [None]),
            
            # Blood Pressure - Systolic
            FeatureColumn('Examination', 'BPXSY2', 
                                 [preproc_real], [None]),
            
            # Blood Pressure - Diastolic
            FeatureColumn('Examination', 'BPXDI2', 
                                 [preproc_real], [None]),
            
            # V02 Max
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Examination', 'CVDESVO2',
                                 [preproc_real], [None]),
            
            # Fitness level
            #FeatureColumn('Examination', 'CVDFITLV', 
                                 #[preproc_real_mapped], [{'mappedValue': 1, 'normalize': False}]),
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Examination', 'CVDFITLV', 
                                 [preproc_probfill], [None]),
            
            # Fitzpatrick Skin Type
            #FeatureColumn('Examination', 'DEX1FITZ', 
                                 #[preproc_real_mapped], [{'cutoff':6, 'mappedValue': 3, 'normalize': False}]),
            
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Examination', 'DEX1FITZ', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff':6}, None]),
            
            # Grip Strength
            FeatureColumn('Examination', 'MGDCGSZ', 
                                 [preproc_real], [None]),
                                 
            ### LABORATORY INFO - POTENTIAL CLUSTERING TARGET
            
            ## Potential Targets: AMDGYB, APOB, BIOPRO, CAFE, CBC, DEET, HCAA..., HDL, L13..., L18..., L25..., L40..., LAB10, LAB13, LAB18, LAB25, TB, TFA, TCHOL, TRIGLY
            ## Non-Targets: AL_IGE, ALB_CR, ALCR, ALDUST, B12, B27, BFRPOL, CARB, CHLM..., CMV, COT..., CRCO, CRP, CUSEZN, DOXPOL, EPH, EPP, FASTQX, FERTIN, FETIB, FLDEP, FLDEW, FOL..., GHB, GLU, HCY, HEP..., HIV, HP..., HSV, HUKM, HULT, IHG..., INS, L02..., L03..., L04..., L05..., L06..., L09..., L10..., L11..., L16..., L17..., L19..., L20..., L24..., L26..., L28..., L31..., L34..., L35..., L36..., L37..., L39..., L43..., L45..., L52..., LAB02, LAB03, LAB04, LAB05, LAB06, LAB07, LAB09, LAB11, LAB16, LAB17, LAB18-22, LAB26, LAB28, OGTT, OHPV, OPD, ORHPV, PAH, PBCD, PCBPOL, PERNT, PFAS, PFC, PH..., POOLTF, PP, PSA, PSTPOL, PTH, SER, SSAFB, SSAMH, SS..., SW..., TELO, TFR, TGEMA, THYROD, TRICH, TST, U..., VIT, VOC, WPIN
            
            # Acrylamide
            FeatureColumn('Laboratory', 'LBXACR', 
                                 [preproc_real], [None]),
            # Caffeine
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Laboratory', 'URXAMU', 
                                 [preproc_real], [None]),
            # White Blood Cells
            FeatureColumn('Laboratory', 'LBXWBCSI', 
                                 [preproc_real], [None]),
            # Red Blood Cells
            FeatureColumn('Laboratory', 'LBXRBCSI', 
                                 [preproc_real], [None]),
            # Hemoglobin
            FeatureColumn('Laboratory', 'LBXHGB', 
                                 [preproc_real], [None]),
            # Platelets
            FeatureColumn('Laboratory', 'LBXPLTSI', 
                                 [preproc_real], [None]),
            # DEET
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Laboratory', 'URXDEE', 
                                 [preproc_real], [None]),
            # HDL Cholesterol
            FeatureColumn('Laboratory', 'LBDHDD', 
                                 [preproc_real], [None]),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC', 
                                 [preproc_real], [None]),
            # Triglycerides
            # LESS THAN 40% OF DATA PRESENT
            FeatureColumn('Laboratory', 'LBXTR', 
                                 [preproc_real], [None]),
            # LDL Cholesterol
            # LESS THAN 40% OF DATA PRESENT
            FeatureColumn('Laboratory', 'LBDLDL', 
                                 [preproc_real], [None]),
            
            ### QUESTIONNAIRE INFO - POTENTIAL CLUSTERING TARGET
            
            ## Potential Targets: ACQ, ALQ, AQQ, BPQ, CBQ, CDQ, DBQ, DEQ, DUQ, FSQ, HOQ, HUQ, INQ, MCQ, MPQ, OCQ, PAQ, PFQ, PUQ, RXQ, SLQ, SMQ, TBQ, WHQ
            ## Non-Targets: AGQ, ARQ, AUQ, BAQ, BHQ, CFQ, CIQ..., CKQ, CSQ, DIQ, DLQ, DPQ, ECQ, HCQ, HEQ, HIQ, HSQ, IMO, KIQ, OHQ, OSQ, PSQ, RDQ, RHQ, SSQ, SXQ, VIQ, VTQ, Y...
            
            # 1+ Alcoholic Drink
            # LESS THAN 40% OF DATA PRESENT
            FeatureColumn('Questionnaire', 'ALQ120Q', 
                                 [preproc_real], [{'cutoff':365}]),
            
            # 5+ Alcoholic Drinks
            # LESS THAN 30% OF DATA PRESENT
            FeatureColumn('Questionnaire', 'ALQ140Q', 
                                 [preproc_real], [{'cutoff':365}]),
            
            # Home Meals
            FeatureColumn('Questionnaire', 'CBQ190', 
                                 [preproc_real], [{'cutoff':35}]),
            
            # Restaurant Meals
            FeatureColumn('Questionnaire', 'DBD090', 
                                 [preproc_real_transform], [{'cutoff':21, 'toTransform': [{'selectValue': 6666, 'mappedValue': 0.5}], 'useOldMean': False}]),
            
            # Melanoma Close Relative
            #FeatureColumn('Questionnaire', 'DEQ050', 
                                 #[preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 2, 'normalize': False}, None]),
            FeatureColumn('Questionnaire', 'DEQ050', 
                                 [preproc_cutoff, preproc_probfill, preproc_onehot], [{'cutoff':2}, None, None]),
            
            # Age of Home
            #FeatureColumn('Questionnaire', 'HOD040', 
                                 #[preproc_real_mapped], [{'cutoff':6, 'mappedValue': 2, 'normalize': False}]),
            FeatureColumn('Questionnaire', 'HOD040', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff':6}, None]),
            
            # Health Condition
            #FeatureColumn('Questionnaire', 'HUQ010', 
                                 #[preproc_real_mapped], [{'cutoff':5, 'mappedValue': 2, 'normalize': False}]),
            FeatureColumn('Questionnaire', 'HUQ010', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff':5}, None]),
            
            # Medical Treament Frequency
            #FeatureColumn('Questionnaire', 'HUQ050', 
                                 #[preproc_real_mapped], [{'cutoff':5, 'mappedValue': 1, 'normalize': False}]),
            FeatureColumn('Questionnaire', 'HUQ050', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff':5}, None]),
            
            # Blood Transfusion
            #FeatureColumn('Questionnaire', 'MCQ092', 
                                 #[preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 2, 'normalize': False}, None]),
            FeatureColumn('Questionnaire', 'MCQ092', 
                                 [preproc_cutoff, preproc_probfill, preproc_onehot], [{'cutoff':2}, None, None]),
            
            # Work Days Missed
            #FeatureColumn('Questionnaire', 'MCQ245B', 
                                 #[preproc_real_mapped], [{'cutoff':365, 'mappedValue': 365}]),
            FeatureColumn('Questionnaire', 'MCQ245B', 
                                 [preproc_real], [{'cutoff':365}]),
            
            # Hours Working
            # LESS THAN 40% OF DATA PRESENT
            FeatureColumn('Questionnaire', 'OCQ180', 
                                 [preproc_real], [{'cutoff':105}]),
            
            # Hours Sleeping
            FeatureColumn('Questionnaire', 'SLD010H', 
                                 [preproc_real], [{'cutoff':12}]),
            
            # Restlessness
            #FeatureColumn('Questionnaire', 'SLQ110', 
                                 #[preproc_real_mapped], [{'cutoff': 4, 'mappedValue': 0, 'normalize': False}]),
            FeatureColumn('Questionnaire', 'SLQ110', 
                                 [preproc_cutoff, preproc_probfill], [{'cutoff': 4}, None]),
            
            # 100+ Cigarettes
            #FeatureColumn('Questionnaire', 'SMQ020', 
                                 #[preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 2, 'normalize': False}, None]),
            # LESS THAN 50% OF DATA PRESENT
            FeatureColumn('Questionnaire', 'SMQ020', 
                                 [preproc_cutoff, preproc_probfill, preproc_onehot], [{'cutoff':2}, None, None]),
            
            # 20+ Chewing Tobacco
            #FeatureColumn('Questionnaire', 'SMQ210', 
                                 #[preproc_real_mapped, preproc_onehot], [{'cutoff':2, 'mappedValue': 2, 'normalize': False}, None])
            # LESS THAN 50% OF DATA PRESENT
            FeatureColumn('Questionnaire', 'SMQ210', 
                                 [preproc_cutoff, preproc_probfill, preproc_onehot], [{'cutoff':2}, None, None]),
            
            # Anyone smoke in the home?
            FeatureColumn('Questionnaire', 'SMD410', 
                                 [preproc_cutoff, preproc_probfill, preproc_onehot], [{'cutoff':2}, None, None])
            
            # Prescription Medication Total
            # THIS CAUSES ERROR WHEN LOADING
            #FeatureColumn('Questionnaire', 'RXD295', 
                                 #preproc_real, None)
                                 
            ### DIETARY INFO - seems to be too specific to work with ###
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ220'], axis=1)
        features = fe_cols.values
        target = df['MCQ220'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        #targets = np.zeros(target.shape[0])
        targets = np.full((target.shape[0]), 3)
        targets[target == 1] = 0 # yes cancer
        targets[target == 2] = 1 # no cancer

        # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
        
        
    #### Add your own dataset loader ####

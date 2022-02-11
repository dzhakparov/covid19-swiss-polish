from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, \
    GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


# CONFIG

simulation_name = "knn_imputation_troponin_all"
target = 'group'
working_file = 'working_file_base_2021-03-22'  # will be stored as .pkl and .csv

# 1. preprocessing (script final_preprocessing.py)

# troponin >= 0.1
threshold_troponin = (False, '>=', 0.1)  # should threshold be applied (True,False), threshold

store_path = f"data/simulation_{simulation_name}"

# input-files
data_files = ['data/input/CoV-2_positive_final_lab_17_08.csv',
              'data/input/CoV-2_negative_final_lab_17_08.csv',
              'data/input/CoV-2_final_lab_10_29_appendix.csv'
              ]

# indices of columns which occurs also in the file of the negative patients
columns_id = [0, 1, 2, 3, 4, 5, 6, 7, 9, 41, 42, 46, 47, 59, 60, 72, 87, 99, 100, 111, 112, 123, 135, 147, 154, 162,
              171, 178, 185, 195, 205, 206, 212, 227, 239, 254, 276, 277, 287, 288]

columns_names = ['id', 'final_result', 'date_of_hosp', 'sex', 'age', 'ward', 'day_of_transfer_icu',
                 'n_days_hosp', 'max_mews', 'albumin1_', 'albumin2_', 'crp1_', 'crp2_', 'pct1_', 'pct2_', 'wbc_',
                 'lymphocytes_', 'reactive-lympho1_', 'reactive-lympho2_', 'antibody-synth-lymph1_',
                 'antibody-synth-lymph2_', 'neutrophil-granules_', 'neutrophil-reactivity_', 'ck_', 'ck_mb_',
                 'troponin_', 'ldh_', 'ferritin_', 'ast_', 'alt_', 'amylaza1_', 'amylaza2_', 'plt_',
                 'pdw_', 'mpv_', 'pt_', 'aptt1_', 'aptt2_', 'd-dimer1_', 'd-dimer2_']

# columns with possibly missing values in first columns -> second value will be taken (as first)
first_value = ['albumin', 'crp', 'pct', 'reactive-lympho', 'amylaza', 'd-dimer', 'aptt', 'antibody-synth-lymph']

remove_cols_file1 = ['date_of_hosp']

# negative patients file
new_names = {'Code': 'id',
             'Final Result': 'final_result',
             'Date of hospitalisation': 'date_of_hosp',
             'Sex': 'sex',
             'Age': 'age',
             'Ward': 'ward',
             'Day of transfer to ICU': 'day_of_transfer_ICU',
             'Duration of hospitalisation [days]': 'n_days_hosp',
             'Last MEWS': 'max_mews',
             'Albumin [g/dl]': 'albumin_',
             'CRP [mg/dl]': 'crp_',
             'Procalcitonin [ug/ml]': 'pct_',
             'WBC [10^3/ul]': 'wbc_',
             'Lymphocytes [10^3/ul]': 'lymphocytes_',
             'Reactive lymphocytes [10^3/ul]': 'reactive-lympho_',
             'Antibody synthesising lymphocytes [10^3/ul]': 'antibody-synth-lymph_',
             'Neutrophil granules [SI]': 'neutrophil-granules_',
             'Neutrophil reactivity [FL]': 'neutrophil-reactivity_',
             'CK [U/l]': 'ck_',
             'CK-MB [ug/ml]': 'ck_mb_',
             'Troponin I [ng/ml]': 'troponin_',
             'LDH [U/l]': 'ldh_',
             'Ferritin [ng/ml]': 'ferritin_',
             'AST [U/l]': 'ast_',
             'ALT [U/l]': 'alt_',
             'Amylase [U/l]': 'amylaza_',
             'PLT [10^3/ul]': 'plt_',
             'PDW [%]': 'pdw_',
             'MPV [fL]': 'mpv_',
             'PT [s]': 'pt_',
             'APTT [s]': 'aptt_',
             'D-dimer [ug/ml]': 'd-dimer_'
             }

remove_cols_file2 = ['unnamed: 9', 'date_of_hosp']

# take only numeric part of cell and remove text part
repl_text_with_number = ['ast_', 'ldh_', 'albumin_', 'plt_', 'aptt_', 'd-dimer_', 'pt_']

# arguments for class Preprocessor

replace_values = {
    'pattern': [['< 0.10'], ['< 0.02'], ['< 0.3'], ['< 7'], ['< 0.002'], ['< 5'], ['< 4'], ['> 1675.56']],
    'value': ['0.05', '0.01', '0.15', '3.5', '0.001', '2.5', '2', '1675.56'],
    'columns': ['crp_', 'pct_', 'ck-mb_', 'ck_', 'troponin_', 'alat_', 'amylaza_', 'ferritin_']
}

convert_categorical = ('sex', 'final_result', 'group', 'ward')  # rest will be numeric

build_quotient = [
    ('neu_', 'lymphocytes_'),
    ('eos_', 'lymphocytes_'),
    ('mon_', 'lymphocytes_'),
    ('plt_', 'lymphocytes_'),
    ('baso_', 'lymphocytes_'),
    ('plt_', 'neu_'),
                  ]


# 2. eda (script: 2-eda.py)

# REMARK: modified working_file (..._adjusted.pkl and ..._adjusted.csv) will be stores in input-path due
# file are at same place as original working_files

# parameters

usable_columns = ['group', 'albumin_', 'alt_', 'amylaza_', 'antibody-synth-lymph_', 'aptt_', 'ast_', 'ck_',
                  'ck_mb_', 'creatinine_', 'crp_', 'd-dimer_', 'hgb_', 'ldh_',
                  'mpv_', 'neutrophil-granules_', 'neutrophil-reactivity_', 'pct_', 'pdw_',
                  'pt_', 'reactive-lympho_', 'troponin_', 'wbc_', 'ferritin_',
                  'quot_neu_lymphocytes_', 'quot_eos_lymphocytes_', 'quot_mon_lymphocytes_',
                  'quot_plt_lymphocytes_', 'quot_baso_lymphocytes_', 'quot_plt_neu_'
                  # 'age', 'sex'
                  ]

drop_threshold_subgroup = 0.4  # features in subgroup positive or negative that exceeds threshold will be dropped

# 3. train_model (script: 3-final_train_model)

# random_states = [1, 2, 3, 25, 26, 10, 11, 12, 13, 14, 33, 34, 35, 36, 37]  # ORIGINAL
random_states = [1, 2]

# test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # other test-sizes are removed  # ORIGINAL
test_sizes = [0.2, 0.25]  # other test-sizes are removed

scoring = ['accuracy', 'roc_auc', 'precision']
# scoring = ['accuracy']

params = (
    [
        {'prep__num_pipeline__imputer__n_neighbors': [1, 3, 5]}
    ],

    [
        {'estimator': [LogisticRegression()],
         # 'estimator__C': [0.1, 1, 10],  # best C=1, 10  # ORIGINAL
         'estimator__C': [1],  # best C=1, 10
         'estimator__penalty': ['l2'],  # only l2 (default) works
         },

        {'estimator': [KNeighborsClassifier()],
         # 'estimator__n_neighbors': [3, 5, 7, 9]},  # 1 neighbors is overfitting  # ORIGINAL
         'estimator__n_neighbors':[3]},  # 1 neighbors is overfitting

        # ORIGINAL : comment-in following lines
        # {'estimator': [RandomForestClassifier()],
        #  'estimator__max_depth': [1],
        #  'estimator__min_samples_leaf': [0.005, 0.01, 0.05, 0.10],  # slightly positive
        #  'estimator__min_samples_split': [0.005, 0.01, 0.05, 0.10],  # no big difference
        #  'estimator__criterion': ['gini', 'entropy']  # ?
        #  },
        #
        # {'estimator': [AdaBoostClassifier()],
        #  'estimator__base_estimator': [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 2)],
        #  'estimator__learning_rate': [0.1, 0.25, 0.50, 0.75, 1.0]
        #  },
        #
        # {'estimator': [BaggingClassifier()],
        #  'estimator__base_estimator': [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 5)],
        #  'estimator__max_features': [0.2, 0.4, 0.6, 0.8, 1.0]
        #  },
        #
        # {'estimator': [GradientBoostingClassifier()],
        #  "estimator__max_depth": [1, 2],
        #  "estimator__learning_rate": [0.15, 0.1, 0.05, 0.01],
        #  "estimator__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
        #  "estimator__max_features": ["auto", "sqrt", "log2"],
        #  "estimator__subsample": [0.8, 0.9, 1],
        #  },
        #
        # {'estimator': [SVC()],
        #  "estimator__C": [0.1, 0.5, 1, 5, 10],
        #  "estimator__kernel": ["linear", "rbf", "poly"],
        #  }
    ]
)
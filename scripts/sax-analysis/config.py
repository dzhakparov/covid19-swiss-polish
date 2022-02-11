import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# output
simulation_name = "knn_imputation_new_seeds_2"
target = 'final_result'
working_file = 'working_file_long_2021-03-22'  # will be stored as .pkl and .csv
store_path = f"data/simulation_{simulation_name}"


# input-files
data_files = [
    'data/input/CoV-2_motherdatabase_lab_17_08.csv',
    'data/input/CoV-2_motherdatabase_lab_29_10_appendix.csv',
    'data/input/CoV-2_motherdatabase_clinics_17_08.csv',
    'data/input/CoV-2_motherdatabase_epidem_symptoms_17_08.csv'
]

renaming_files = [
    'data/input/renaming_lab_17_08.csv',
    'data/input/renaming_lab_29_10_appendix.csv',
    'data/input/renaming_clinics_17_08.csv',
    'data/input/renaming_epidem_symptoms_17_08.csv'
]

drop_threshold = 0.6   # columns with more than 60% missing values will be dropped (except sax_features!)

drop_columns = ['other_1?', 'other_2?', 'drug_name', 'day_transfer_icu',
                'remark', 'remark_1', 'remark_2', 'overweight', 'rheumatologicals', 'drug_name',
                'date_hospitalisation',
                'date_pcr_1', 'days_hospitalisation_1', 'result_1',
                'date_pcr_2', 'days_hospitalisation_2', 'result_2',
                'date_pcr_3', 'days_hospitalisation_3', 'result_3',
                'date_pcr_4', 'days_hospitalisation_4', 'result_4',
                'date_pcr_5', 'days_hospitalisation_5', 'result_5',
                'date_pcr_6', 'days_hospitalisation_6', 'result_6',
                'date_pcr_7', 'days_hospitalisation_7', 'result_7',
                'date_pcr_8', 'days_hospitalisation_8', 'result_8',
                'days_neg_pcr', 'covid_signs_ct', 'control_ct_1', 'control_ct_2', 'max_mews_2', 'mews_1', 'mews_2',
                'mews_3', 'mews_4', 'mews_5', 'mews_6', 'mews_7', 'mews_8', 'mews_9', 'mews_10', 'mews_11',
                'mews_12', 'mews_13', 'mews_14', 'mews_15', 'mews_16', 'mews_17', 'mews_18', 'mews_19', 'mews_20',
                'mews_21', 'mews_22', 'mews_23', 'mews_24', 'mews_25', 'mews_26', 'mews_27', 'mews_28', 'mews_29',
                'mews_30', 'mews_31', 'mews_32', 'mews_33', 'mews_34', 'mews_35', 'mews_36', 'mews_37', 'mews_38',
                'mews_39', 'mews_40', 'mews_41', 'mews_42', 'mews_43', 'mews_44', 'mews_45', 'mews_46',
                'mews_47', 'mews_48', 'mews_49', 'mews_50', 'mews_51', 'mews_52', 'mews_53', 'mews_54', 'mews_55',
                'pregnant_birth', 'saturation_oxygen_sup', 'temp_lower36', 'temp_36_36.9', 'temp_37_38',
                'temp_higher38',
                'dyspnoea_1w', 'cough_1w', 'loss_smell_taste_1w', 'low_physical_activity_1w', 'dizziness_1w',
                'deterioration_1w', 'headache_1w', 'chest_pain_1w', 'last-pcr', 'duration_hospitalisation', 'ward',
                'ID', 'respirator',
                'contact_with_covid_person_2w', 'travelling_high_prevalence_cases', 'quarantine_2w',
                'contact_health_service_2w', 'hc_house_resident', 'active_hc_worker', 'much_social_contact_work',
                'max_mews']

repl_text_with_number = ['d-dimer_1', 'ast_1', 'ast_2', 'ast_3', 'ast_4', 'ast_5', 'ldh_1', 'albumin_1', 'plt_1',
                         'plt_2', 'plt_3', 'plt_5', 'plt_6', 'pt_s_5', 'aptt_1', 'aptt_2', 'aptt_3', 'aptt_4',
                         'aptt_5', 'aptt_6', 'aptt_7', 'aptt_8', 'd-dimer_2', 'd-dimer_3', 'd-dimer_4',
                         'd-dimer_6', 'pt_percent_1', 'pt_percent_4', 'feritin_1', 'feritin_2']

combine_features = {
    'allergy': ['atopic_dermatitis', 'allergy_pokarmowa', 'allergy_pollen_gras_hair_mites', 'other_allergy',
                'drugs'],
    'anti_inflammatory': ['paracetamol', 'nsaids', 'sulfasalazine'],
    'anti_respiratory_disease': ['antileukotriene', 'gcs_inhaled', 'saba', 'laba', 'lama'],
    'anti-cholesterol': ['statins', 'ezetimibe'],
    'anti-platelets': ['asa', 'clopidogrel', 'cilostazole'],
    'anti-thrombosis': ['vka/doac', 'heparin'],
    'autoimmunity_inflammation': ['arthrithis', 'psoriasis_arthritis', 'lupus', 'ankylosing_spondylitis',
                                  'ulcerative_colitis', 'psoriasis', 'multiple_sclerosis'],
    'cancer': ['neoplastic_diseases'],
    'cardiovascular_disease': ['heart_failure', 'coronary_artery_disease', 'atrial_fibrillation', 'stemi_past',
                               'stroke_past'],
    'DMRD': ['methotrexate', 'leflunomide', 'cyklosporyna', 'hydroxychloriquine'],
    'elew_temp_2w': ['elev_temp_2w', 'temp_37_38_2w', 'temp_higher38_2w'],
    'gastro': ['diarrhoe_1w', 'abdominal_pain_1w', 'loss_appetit_1w', 'vomitting_1w', 'nausea_1w'],
    'hematological_cancer': ['lymphocytic_leukaemia', 'lymphoma', 'multiple_myeloma', 'polycythemia'],
    'infectious_disease': ['hepatitis_b', 'hepatitis_c', 'hiv', 'tuberculosis'],
    'med_cardiovascular_disease': ['acei', 'arb', 'ccb', 'bb', 'mcra', 'diuretics'],
    'metabolic_endocrine': ['obesity', 'gout', 'osteoporosis', 'hipercholesterolemia', 'hyperthyroidism'],
    'nephro': ['kidney_failure'],
    'neurological': ['migraine', 'epilepsy', 'parkinson', 'other_neurological_diseases'],
    'OCS': ['steroids'],
    'psychiatric': ['schizophrenia', 'depression', 'other_psychiatric_distress'],
    'respiratory_disease': ['asthma', 'copd', 'pulmonary_hypertension', 'pulmonary_fibrosis',
                            'other_pulmonary_diseases'],
    'thrombotic': ['thrombosis', 'pulmonary_embolism', 'cteph']
}

column_conversion_dict = {
    ('final_result', 'respirator', 'sex', 'ward'): ('category', ''),
    ('diabetes', 'alcoholism', 'nicotine_addiction', 'anemia', 'antihistamine', 'allergy',
     'anti_inflammatory', 'anti_respiratory_disease', 'anti-cholesterol', 'anti-platelets', 'anti-thrombosis',
     'autoimmunity_inflammation', 'cancer', 'cardiovascular_disease', 'DMRD', 'gastro', 'hematological_cancer',
     'infectious_disease', 'med_cardiovascular_disease', 'metabolic_endocrine', 'nephro', 'neurological', 'OCS',
     'psychiatric', 'respiratory_disease', 'thrombotic', 'elew_temp_2w', 'hypertension', 'dementia'):
        ('category', (0, 1)),  # contact ...
    ('ID',): ('object', ''),
    ('max_mews', 'albumin_1', 'crp_1', 'crp_2', 'crp_3', 'crp_4', 'crp_5', 'crp_7', 'crp_9', 'pct_1', 'pct_2',
     'pct_3', 'pct_4', 'pct_5', 'wbc_2', 'reactive-lympho_1', 'reactive-lympho_2', 'reactive-lympho_3',
     'reactive-lympho_4', 'reactive-lympho_5', 'reactive-lympho_7', 'antibody-lympho_1', 'antibody-lympho_2',
     'antibody-lympho_3', 'antibody-lympho_4', 'antibody-lympho_5', 'antibody-lympho_7', 'ck_2', 'ck-mb_1',
     'ck-mb_2', 'ck-mb_4', 'troponin_1', 'troponin_2', 'troponin_3', 'ldh_1', 'feritin_1', 'feritin_2', 'alat_1',
     'pdw_1', 'pdw_2', 'pdw_4', 'pdw_5', 'pdw_6', 'mpv_2', 'mpv_3', 'mpv_4', 'pt-s_1', 'pt-percent_1',
     'pt-percent_4', 'aptt_1', 'aptt_2', 'aptt_3', 'aptt_4', 'aptt_5', 'aptt_6', 'd-dimer_1', 'd-dimer_2',
     'd-dimer_3', 'd-dimer_4', 'd-dimer_6', 'max_mews_clinics', 'lymphocytes_1', 'lymphocytes_2', 'lymphocytes_3',
     'lymphocytes_4',
     'plt_1', 'plt_2', 'plt_3', 'plt_4', 'plt_5'): ('numeric', '')
}

# needed for dropping columns with more than x% missing values EXCEPT sax_features!
sax_features = ('albumin_', 'amylaza_', 'crp_', 'troponin_', 'neutrophil-reactivity_', 'ck_', 'wbc_',
                'neutrophil-granules_', 'lymphocytes_', 'alat_', 'd-dimer_', 'mpv_', 'ck-mb_', 'pct_',
                'reactive-lympho_', 'plt_', 'ast_', 'pdw_', 'pt-s_', 'pt-percent_', 'aptt_', 'ldh_',
                'antibody-lympho_', 'feritin_', 'hgb_', 'creatinine_', 'eos_', 'mon_', 'neu_', 'baso_')

# out of Preprocessor-Class
outliers = [('SC2_Pos98', 'ldh_2'), ('SC2_Pos190', 'reactive-lympho_2'),
            ('SC2_Pos4', 'wbc_3')]  # 23670.00000 / 2.5 / 500

recoding_final_results = {'Recovered': 'survived', 'Death': 'died', 'Symptoms': 'survived', 'Infected': 'survived'}

repl_cell_content_strings = ['allergy_pollen_gras_hair_mites', 'statins', 'asa', 'nicotine_addiction',
                             'other_pulmonary_diseases', 'other_neurological_diseases', 'neoplastic_diseases',
                             'other_psychiatric_distress', 'diabetes']

build_quotient_ts_cols = [('plt', 'neu'), ('plt', 'lymphocytes'), ('neu', 'lymphocytes'), ('eos', 'lymphocytes'),
                          ('mon', 'lymphocytes'),('baso', 'lymphocytes')]

repl_pattern = [['< 0,10'], ['< 0,02'], ['< 0,3'], ['< 7'], ['< 0,002'], ['< 5'], ['----', 'x', 'X']]
repl_pattern_values = [0.05, 0.01, 0.15, 3.5, 0.001, 2.5, np.nan]


# 2 . EDA

# ******************** (1) new trial with knn-imputing missing values instead of drop ********************

# usable_columns = ['group', 'albumin_', 'alt_', 'amylaza_', 'antibody-synth-lymph_', 'aptt_', 'ast_', 'ck_',
#                   'ck_mb_', 'creatinine_', 'crp_', 'd-dimer_', 'hgb_', 'ldh_',
#                   'mpv_', 'neutrophil-granules_', 'neutrophil-reactivity_', 'pct_', 'pdw_',
#                   'pt_', 'reactive-lympho_', 'troponin_', 'wbc_', 'ferritin_', 'sex', 'age',
#                   'quot_neu_lymphocytes_', 'quot_eos_lymphocytes_', 'quot_mon_lymphocytes_',
#                   'quot_plt_lymphocytes_', 'quot_baso_lymphocytes_']


# 3.

# random_states = [1, 2, 3, 25, 26, 10, 11, 12, 13, 14, 33, 34, 35, 36, 37]
# random_states = [1211,2341,4351,5641,1521,6541,8961,9851,7471,8681,9011,4321,9801,1921,8561]
random_states = [1211,2341]

# test_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # other test-sizes are removed
test_sizes = [0.25]  # other test-sizes are removed

scoring = ['accuracy', 'roc_auc', 'precision']

sax_transformer = {
        'n_letters': 2,
        'n_length': 2,
        'scaler': 'z_all',
        'thresholds': {'troponin': [0.1]},
        'cat_ordered': False
}

# THRESHOLD: < 5% MISSING VALUES per FEATURE
categories = {'laboratory': ['crp', 'troponin', 'neutrophil-reactivity', 'wbc',
                             'neutrophil-granules', 'lymphocytes', 'alat', 'd-dimer', 'mpv', 'ck-mb',
                             'pct', 'reactive-lympho', 'ast', 'pdw', 'pt-s', 'pt-percent', 'aptt', 'ldh', 'antibody-lympho',
                             'hgb', 'creatinine', 'eosQlymphocytes', 'monQlymphocytes', 'neuQlymphocytes',
                             'basoQlymphocytes', 'pltQneu', 'pltQlymphocytes'],
              'comorbidity': ['cancer', 'nicotine_addiction', 'anemia', 'psychiatric', 'metabolic_endocrine',
                              'cardiovascular_disease', 'dementia', 'diabetes', 'nephro',
                              'hypertension', 'metabolism_endocrine', 'respiratory_disease', 'age'],
              'premedication': ['med_cardiovascular_disease', 'anti-cholesterol', 'anti-thrombosis', 'anti-platelets',
                                'antihistamine']
              }

# coding of each column -> static
sax_coded = ('pt-s', 'pt-percent', 'albumin', 'amylaza', 'crp', 'troponin', 'neutrophil-reactivity', 'ck',
             'wbc', 'neutrophil-granules', 'lymphocytes', 'alat', 'd-dimer', 'mpv', 'ck-mb',
             'pct', 'reactive-lympho', 'ast', 'pdw', 'aptt', 'ldh', 'antibody-lympho', 'feritin',
             'hgb', 'creatinine', 'eosQlymphocytes', 'monQlymphocytes', 'neuQlymphocytes', 'basoQlymphocytes',
             'pltQneu', 'pltQlymphocytes')

num_coded = ('age', 'duration_hospitalisation', 'max_mews', 'saturation_oxygen_sup')

bin_coded = ('diabetes', 'alcoholism', 'nicotine_addiction', 'anemia', 'antihistamine', 'allergy', 'anti_inflammatory',
             'anti_respiratory_disease', 'anti-cholesterol', 'anti-platelets', 'anti-thrombosis',
             'autoimmunity_inflammation', 'cancer', 'cardiovascular_disease', 'contact', 'DMRD', 'elew_temp_2w',
             'gastro', 'hematological_cancer', 'infectious_disease', 'med_cardiovascular_disease',
             'metabolic_endocrine', 'nephro', 'neurological', 'OCS', 'psychiatric', 'respiratory_disease', 'thrombotic',
             'hc_house_resident', 'hypertension', 'dementia', 'pregnant_birth', 'temp_<36', 'temp_36_36.9',
             'temp_37_38', 'temp_>38', 'dyspnoea_1w', 'cough_1w', 'loss_smell_taste_1w', 'low_physical_activity_1w',
             'dizziness_1w', 'deterioration_1w', 'headache_1w', 'chest_pain_1w')

include_cols = [categories['laboratory'], categories['comorbidity'], categories['premedication']]

# hyper-parameter space
params = (
        [{
            'prep__num_pipeline__imputer_num__n_neighbors': [1, 3],
            'prep__cat_pipeline__imputer_cat__n_neighbors': [1, 3]
        }],

        [

            {'estimator': [LogisticRegression()],
             # 'estimator__C': [0.1, 1, 10],  # best C=1, 10
             'estimator__C': [1],  # best C=1, 10
             'estimator__penalty': ['l2'],  # only l2 (default) works
             },

            {'estimator': [KNeighborsClassifier()],
             # 'estimator__n_neighbors': [3, 5, 7, 9],
             'estimator__n_neighbors': [3],
             'estimator__weights': ['uniform', 'distance']
             },

            # {'estimator': [RandomForestClassifier()],
            #  'estimator__max_depth': [1, 2, 3],
            #  'estimator__min_samples_leaf': [0.01, 0.05, 0.10],  # slightly positive
            #  'estimator__min_samples_split': [0.001, 0.01, 0.10],  # no big difference
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

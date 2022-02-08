from sklearn.pipeline import Pipeline
from sklearn_custom.feature_selection.VarianceThreshold import VarianceThreshold
from sklearn_custom.imputers.SimpleImputer import SimpleImputer
from sklearn_custom.encoders.OneHotEncoder import OneHotEncoder
from sklearn_custom.encoders.OrdinalEncoder import OrdinalEncoder
from sklearn_custom.transformers.ColumnTransformer import ColumnTransformer
from sklearn_custom.preprocessing.MinMaxScaler import MinMaxScaler

# scoring options: https://scikit-learn.org/stable/modules/model_evaluation.html

# GENERAL
general = {

    'output_path': 'data/output/',
    'input_path': 'data/input/',

    'store_infos': {'location': "data/store",
                    # this files will NOT be stored in store-folder (_cloud) intended upload
                    # cloud (dropbox or google-drive)
                    'sensitive_files': [
                        'connected_file_before_preprocessing.csv',
                        'working_file.csv', 'working_file.pkl',
                        'X_test.pkl', 'X_test_ml.pkl', 'X_test_sax.csv', 'X_train.pkl', 'X_train_ml.csv',
                        'X_train_ml.pkl', 'X_train_sax.csv', 'y_test.pkl', 'y_train.pkl',
                        'df_results.joblib', 'feature_selection.joblib', 'fitted_models.joblib'
                    ]
                    },

    'data_files': ['CoV-2_motherdatabase_lab_17_08',
                   'CoV-2_motherdatabase_clinics_17_08',
                   'CoV-2_motherdatabase_epidem_symptoms_17_08',
                   'CoV-2_motherdatabase_lab_29_10_appendix'],

    'output_filename': 'working_file',

    'COLORS': {'train': 'rgba(61, 153, 112, 1)', 'cv': '#FF4136', 'test': 'rgba(0, 102, 204, 1)', 'importance': 'rgb(112,112,112)'},
    'COLORS_GROUPING': ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                        'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                        'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                        'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                        'rgb(188, 189, 34)', 'rgb(23, 190, 207)'],

    'image_format': 'html',
    'show_browser': False,
    # 'visualizations': ['visualize_matrix', 'matrix_plot', 'boxplot_sax_features', 'survivalplot_sax_features']
    'visualizations': ['boxplot_sax_features', 'survivalplot_sax_features']
}

# PRE-PROCESSING
preprocessing = {

    'mode': {'doc': True, 'details': True},

    'renaming_files': ['renaming_lab_17_08',
                       'renaming_clinics_17_08',
                       'renaming_epidem_symptoms_17_08',
                       'renaming_lab_29_10_appendix'],

    'last_row_in_bases_files': 201,

    'columns_remove': [
        ['date_pcr_1', 'days_hospitalisation_1', 'result_1', 'date_pcr_2', 'days_hospitalisation_2',
         'result_2', 'date_pcr_3', 'days_hospitalisation_3', 'result_3', 'date_pcr_4',
         'days_hospitalisation_4', 'result_4', 'date_pcr_5', 'days_hospitalisation_5', 'result_5',
         'date_pcr_6', 'days_hospitalisation_6', 'result_6', 'date_pcr_7', 'days_hospitalisation_7',
         'result_7',
         'date_pcr_8', 'days_hospitalisation_8', 'result_8', 'remark'],

        ['day_transfer_icu', 'final_result', 'date_hospitalisation', 'sex', 'age', 'ward',
         'duration_hospitalisation', 'last-pcr', 'respirator', 'days_neg_pcr', 'date_pcr_1',
         'days_hospitalisation_1', 'result_1', 'date_pcr_2', 'days_hospitalisation_2',
         'result_2', 'date_pcr_3', 'days_hospitalisation_3', 'result_3', 'date_pcr_4',
         'days_hospitalisation_4', 'result_4', 'date_pcr_5', 'days_hospitalisation_5',
         'result_5', 'date_pcr_6', 'days_hospitalisation_6', 'result_6', 'date_pcr_7',
         'days_hospitalisation_7', 'result_7',
         'date_pcr_8', 'days_hospitalisation_8', 'result_8', 'remark_1', 'remark_2',
         'max_mews_clinics', 'max_mews',
         'overweight', 'rheumatologicals', 'other_1?','other_2?', 'drug_name'
         ],

        ['final_result', 'date_hospitalisation', 'sex', 'age'],

        ['final_result','date_hospitalisation','sex','age']
    ],

    'value_replacement': [
        {'column': 'crp', 'pattern': '< 0,10', 'value': '0.05'},
        {'column': 'pct', 'pattern': '< 0,02', 'value': '0.01'},
        {'column': 'ck-mb', 'pattern': '< 0,3', 'value': '0.15'},
        {'column': 'ck', 'pattern': '< 7', 'value': '3.5'},
        {'column': 'troponin', 'pattern': '< 0.002', 'value': '0.001'},
        {'column': 'alat', 'pattern': '< 5', 'value': '2.5'},
    ],

    'forced_numeric': ['wbc_2', 'wbc_3', 'wbc_4', 'wbc_6', 'pt_s_1', 'pt_s_2', 'pt_s_3', 'pt_s_4', 'pt_s_6',
                       'saturation_oxygen_sup', 'max_mews'],

    'replace_with_nan': ['----'],

    'repl_text_with_number': ['ast_1', 'ast_2', 'ast_3', 'ast_4', 'ast_5', 'ldh_1', 'albumin_1', 'plt_1', 'plt_2',
                              'plt_3', 'plt_5', 'plt_6', 'pt_s_5', 'aptt_1', 'aptt_2', 'aptt_3', 'aptt_4', 'aptt_5',
                              'aptt_6', 'aptt_7', 'aptt_8', 'd-dimer_1', 'd-dimer_2', 'd-dimer_3', 'd-dimer_4',
                              'd-dimer_6', 'other_1?', 'pt_percent_1', 'pt_percent_4', 'feritin_1', 'feritin_2'],

    'repl_cell_content_strings': ['allergy_pollen_gras_hair_mites', 'statins', 'asa', 'nicotine_addiction',
                                  'other_pulmonary_diseases', 'other_neurological_diseases', 'neoplastic_diseases',
                                  'other_psychiatric_distress', 'diabetes'],

    # if value of any feature is 1, the whole new category will be 1!
    'combine_features': {'allergy': ['atopic_dermatitis','allergy_pokarmowa', 'allergy_pollen_gras_hair_mites', 'other_allergy', 'drugs'],
                         'anti_inflammatory': ['paracetamol','nsaids','sulfasalazine'],
                         'anti_respiratory_disease': ['antileukotriene','gcs_inhaled','saba','laba','lama'],
                         'anti-cholesterol': ['statins', 'ezetimibe'],
                         'anti-platelets': ['asa','clopidogrel','cilostazole'],
                         'anti-thrombosis': ['vka/doac','heparin'],
                         'autoimmunity_inflammation': ['arthrithis','psoriasis_arthritis','lupus','ankylosing_spondylitis','ulcerative_colitis','psoriasis','multiple_sclerosis'],
                         'cancer': ['neoplastic_diseases'],
                         'cardiovascular_disease': ['heart_failure','coronary_artery_disease','atrial_fibrillation','stemi_past','stroke_past'],
                         'contact': ['travelling_high_prevalence_cases','contact_with_covid_person_2w','quarantine_2w','active_hc_worker','much_social_contact_work','contact_healt_service_2w'],
                         'DMRD': ['methotrexate','leflunomide', 'cyklosporyna', 'hydroxychloriquine'],
                         'elew_temp_2w': ['elev_temp_2w','temp_37_38_2w','temp_>38_2w'],
                         'gastro': ['diarrhoe_1w','abdominal_pain_1w','loss_appetit_1w','vomitting_1w','nausea_1w'],
                         'hematological_cancer': ['lymphocytic_leukaemia','lymphoma','multiple_myeloma','polycythemia'],
                         'infectious_disease': ['hepatitis_b','hepatitis_c','hiv','tuberculosis'],
                         'med_cardiovascular_disease': ['acei','arb','ccb','bb','mcra','diuretics'],
                         'metabolic_endocrine': ['obesity','gout','osteoporosis', 'hipercholesterolemia', 'hyperthyroidism'],
                         'nephro': ['kidney_failure'],
                         'neurological': ['migraine','epilepsy','parkinson','other_neurological_diseases'],
                         'OCS': ['steroids'],
                         'psychiatric': ['schizophrenia','depression','other_psychiatric_distress'],
                         'respiratory_disease': ['asthma','copd','pulmonary_hypertension','pulmonary_fibrosis','other_pulmonary_diseases'],
                         'thrombotic': ['thrombosis','pulmonary_embolism','cteph']
                         },


    'convert_ordered_categorical': [('covid_signs_ct', ["0", "0?", "1", "1?", "2", "2?"]),
                                    ('control_ct_1', ["0", "0?", "1", "1?", "2", "2?"]),
                                    ('control_ct_2', ["0", "0?", "1", "1?", "2", "2?"])],

    'convert_categorical': ['ward', 'respirator', 'final_result'],

    'convert_categorical_binaries': ['diabetes', 'alcoholism', 'nicotine_addiction', 'anemia',
                                     'antihistamine', 'last-pcr', 'sex',
                                     'allergy', 'anti_inflammatory', 'anti_respiratory_disease', 'anti-cholesterol',
                                     'anti-platelets', 'anti-thrombosis', 'autoimmunity_inflammation', 'cancer',
                                     'cardiovascular_disease', 'contact', 'DMRD', 'gastro',
                                     'hematological_cancer', 'infectious_disease', 'med_cardiovascular_disease', 'metabolic_endocrine',
                                     'nephro', 'neurological', 'OCS', 'psychiatric', 'respiratory_disease',
                                     'thrombotic', 'elew_temp_2w'],

    'convert_date': ['date_hospitalisation'],

    'build_quotient_ts': [('neu','lymphocytes'),  ('eos','lymphocytes'), ('mon','lymphocytes'), ('plt','lymphocytes'),
                          ('plt','neu'), ('baso','lymphocytes')],

    'remove_ts': ['neu_', 'eos_', 'mon_', 'plt_', 'baso_']

}

# EXPLORATORY DATA ANALYSIS (EDA)
eda = {'static_analysis': True,  # boolean if static_analysis should take place for this section

       # features to analyze in profile-report (if 'to_analyze' is an empty list, profile-report will not be executed
       'to_analyze': ['final_result', 'sex', 'respirator', 'crp_1', 'crp_2', 'crp_3', 'crp_4', 'albumin_1',
                      'albumin_2', 'albumin_3', 'albumin_4', 'wbc_1', 'wbc_2', 'wbc_3', 'wbc_4'],

       # features is a list of feature-names (exp. 'albumin_' or 'crp_'): always take the stem of the feature
       # followed by '_'
       'visualize_grouped_timeseries': {
           'features': ['crp_', 'troponin_', 'albumin_', 'wbc_']
       }}


# SAX_ANALYSIS
sax_params = {
    'static_analysis': True,  # boolean if static_analysis should take place for this section

    'features': ['albumin_', 'amylaza_', 'crp_', 'troponin_', 'neutrophil-reactivity_', 'ck_', 'wbc_',
                 'neutrophil-granules_', 'lymphocytes_', 'alat_', 'd-dimer_', 'mpv_', 'ck-mb_',
                  'pct_', 'reactive-lympho_', 'ast_', 'pdw_', 'pt_', 'aptt_', 'ldh_', 'antibody-lympho_',
                  'hgb_', 'creatinine_', 'eosQlymphocytes_', 'monQlymphocytes_', 'neuQlymphocytes_', 'basoQlymphocytes_',
                  'pltQneu_','pltQlymphocytes_'],

    'scaler': ['z_all'],  # 'z_all' or 'z_serie'
    'n_letters': [2],
    'n_length': [2],
    'thresholds': {'troponin': [0.1]},  # set a defined threshold for the sax-clustering instead of taking the calculated one
    'target': 'final_result',
    'split_by': ['None']
}

sax_scan = {
    'warnings_threshold': 0.3,
    'warnings_n': 10
}

sax_plot = {
    'plot_selected': False,
    'show_browser': False,
    # 'plot_modes': ['xy', 'timeseries'],
    'plot_modes': ['timeseries'],
    'y': ['survivalrate', 'max_mews']   # 'survivalrate', 'max_mews', 'age', 'duration_hospitalisation'
}

# ml_pipeline
ml_params = {
    'target': 'final_result',   # define target column as string (category, binary)
    'shuffle_target': False,   # shuffling randomly  y_train and y_test  (for baseline -> guessing)
    'shuffle_state': 52,
    'test_size': 0.25,  # size of test data in percent (value: 0 to 1)
    'random_state': [10, 20],  # , 30, 40, 50, 60, 70, 80, 90, 100],  # list of integers -> number of integers gives number of simulations, value is random seed for according simulation
    'remove_cols': [],   # list of column headers which should be removed for simulation (rest will be used for simulation)


    # laboratory
    # 'remain_cols': ['albumin', 'amylaza', 'crp', 'troponin', 'neutrophil-reactivity', 'ck', 'wbc',
    #              'neutrophil-granules', 'lymphocytes', 'alat', 'd-dimer', 'mpv', 'ck-mb',
    #               'pct', 'reactive-lympho', 'ast', 'pdw', 'pt', 'aptt', 'ldh', 'antibody-lympho',
    #               'hgb', 'creatinine', 'eosQlymphocytes', 'monQlymphocytes', 'neuQlymphocytes', 'basoQlymphocytes',
    #               'pltQneu','pltQlymphocytes'],

    # comorbidity (all)
    # 'remain_cols': ['metabolic_endocrine', 'alcoholism', 'allergy', 'anemia', 'autoimmunity_inflammation',
    #                 'cancer', 'cardiovascular_disease', 'dementia', 'diabetes', 'nephro', 'hypertension',
    #                 'metabolism_endocrine', 'respiratory_disease', 'psychiatric', 'hematological_cancer', 'thrombotic',
    #                 'neurological', 'infectious_disease', 'nicotine_addiction', 'age'],

    # comorbidity > 10% observation (1: 7)
    # 'remain_cols': ['hypertension', 'cardiovascular_disease','diabetes','metabolic_endocrine','dementia',
    #                 'respiratory_disease', 'nephro', 'age'],

    # premedication
    # 'remain_cols': ['med_cardiovascular_disease', 'anti-cholesterol', 'anti-thrombosis', 'anti-platelets',
    #                 'anti_inflammatory', 'anti_respiratory_disease', 'antihistamine', 'OCS', 'DMRD'],

    # premedication > 10% observation (1: 4)
    # 'remain_cols': ['med_cardiovascular_disease', 'anti-cholesterol','anti-thrombosis','anti-platelets'],

    # premedication + comorbidity > 10% observations
    # 'remain_cols': ['hypertension', 'cardiovascular_disease','diabetes','metabolic_endocrine','dementia',
    #                 'respiratory_disease', 'nephro', 'age', 'med_cardiovascular_disease', 'anti-cholesterol',
    #                 'anti-thrombosis','anti-platelets'],

    # best combination
    # 'remain_cols': ['crp', 'wbc', 'pct', 'albumin', 'amylaza', 'neuQlymphocytes', 'nephro', 'hypertension'],
    'remain_cols': ['crp', 'pct', 'wbc', 'ldh', 'pltQneu','pltQlymphocytes', 'neuQlymphocytes',
                    'diabetes','nephro','anemia','age'],

    # epidem_symptoms
    # 'remain_cols': ['contact', 'elew_temp_2w', 'gastro', 'hc_house_resident'],

    # best trial from earlier
    # 'remain_cols': ['crp', 'wbc', 'pct', 'troponin', 'creatinine', 'ck', 'neuQlymphocytes', 'lymphocytes',
    #                 'anemia', 'alcoholism', 'med_cardiovascular_disease', 'age'],

    'dropping_MISSING': True,  # ..._MISSING in sax_features will be removed (makes only sense in combination with
    # drop=None in 'cat_nom_pipeline'
    'drop_frac_na_cols': 1,  # more than x% missing values will be excluded (if value is 1 -> nothing removed)
    'corr_method': 'spearman',  # 'pearson'
    'corr_threshold': 0.9,  # features with correlation > threshold will be removed (if value is 1 -> nothing removed)
    'corr_filter': (0.5, 1),  # only for plotting reasons
    'var_threshold': 0.0,   # float between 0 and 1: if a column has less variance than this threshold it will be removed from simulation (0.0 -> no feature will be removed)
}


# sax preprocessing pipelines: define (preprocess) pipelines for categorical and numerical features
# real column transformer is defined in 'ml_pipeline_transformation.py.' directly
# 'num', 'cat_ord', 'cat_oh', 'cat_bin' have to defined!

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', df_out=True)),
        ('min_max', MinMaxScaler()),
        ('zero_var', VarianceThreshold(threshold=ml_params['var_threshold'], df_out=True)),
    ])

cat_ord_pipeline = Pipeline([
        ('imputer_ord', SimpleImputer(strategy='constant', fill_value='MISSING', df_out=True)),
        ('ord', OrdinalEncoder(df_out=True)),
        ('min_max', MinMaxScaler())
    ])

cat_nom_pipeline = Pipeline([
        ('imputer_nom', SimpleImputer(strategy='constant', fill_value='MISSING', df_out=True)),
        ('oh', OneHotEncoder(sparse=False, df_out=True, new_cats=True, fill_value='MISSING', drop=None)),
        ('zero_var', VarianceThreshold(threshold=(.95 * (1 - .95)), df_out=True)),
    ])

cat_bin_pipeline = Pipeline([
    ('imputer_bin', SimpleImputer(strategy='most_frequent', df_out=True)),
    ('zero_var', VarianceThreshold(threshold=(.95 * (1 - .95)), df_out=True)),
])

sax_pipe = {

    # features to be converted to sax code
    'sax_groups': [
                   'pt_s_', 'pt_percent_', 'albumin_', 'amylaza_', 'crp_', 'troponin_', 'neutrophil-reactivity_', 'ck_',
                   'wbc_', 'neutrophil-granules_', 'lymphocytes_', 'alat_', 'd-dimer_', 'mpv_', 'ck-mb_',
                   'pct_', 'reactive-lympho_', 'ast_', 'pdw_', 'aptt_', 'ldh_', 'antibody-lympho_','feritin_',
                   'hgb_', 'creatinine_', 'eosQlymphocytes_', 'monQlymphocytes_', 'neuQlymphocytes_', 'basoQlymphocytes_',
                   'pltQneu_','pltQlymphocytes_'
                   ],

    'sax_transformer': {'n_letters': 2,
                        'n_length': 2,
                        'scaler': 'z_all',
                        'thresholds': {'troponin': [0.1]},
                        'cat_ordered': False,  # false=one-hot encoded in transformer, true=ordinal_encoded
                        'reminder': 'passthrough'   # ('drop', 'passthrough')
                        }
}

feature_selection = {
    'fw_bw_feature_selection': {
                  'k_features':  4,  # original: 20!  # the lower the features we want, the longer this will take (for forward=False).
                                     # if parameter is not supported, maximal number of features will be taken
                                     # automatically (take long)
                  'forward': True,   # False=Backward
                  'floating': False,
                  'verbose': 2,
                  'scoring': 'accuracy',
                  'cv': 2,  # original 5
                  'n_jobs': 50
    }
}


# MODELING
modeling = {'cv': 5,
            'scoring': 'accuracy'}

# evaluation = {'scoring_methods': ['accuracy_score', 'precision_score', 'recall_score', 'f1_score'],
#               'args': {'precision_score': {'average': 'macro', 'pos_label': 'died'},
#                        'recall_score': {'pos_label': 'died'},
#                        'f1_score': {'pos_label': 'died'}}
#               }

evaluation = {'scoring_methods': ['accuracy_score', 'precision_score', 'recall_score', 'f1_score'],
              'args': {'precision_score': {'average': 'macro', 'pos_label': 'died'},
                       'recall_score': {'pos_label': 'died'}}
              }

evaluation_data = {'data': ['train', 'test']}
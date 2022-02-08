import pandas as pd
from sklearn.preprocessing import LabelEncoder

""" Covid19_data preprocessing
Only specific to positive survival analysis """


class CovidPreprocessor:

    def __init__(self, lab_data, cat_data, appendix, new_cats):
        self.lab_data = lab_data
        self.cat_data = cat_data
        self.appendix = appendix
        self.new_cats = new_cats

    def transform_lab_data(self):
        lab = self.lab_data
        appdx = self.appendix
        # lab values preprocessing ####
        lab.set_index(keys='Code', inplace=True)
        lab.drop(lab.filter(regex='Unnamed').columns, axis=1, inplace=True)
        # convert column names to lower case
        lab.columns = lab.columns.str.lower()
        # remove white spaces in column names
        lab.columns = lab.columns.str.replace(' ', '_')
        # drop the unneeded columns
        lab.drop(axis=1, labels=['date_of_hospitalisation', 'ward', 'day_of_transfer_to_icu',
                                 'last_pcr', 'respirator',
                                 'days_to_negative_pcr_from_the_beginning_of_hospitalisation_or_from_the_positive_pcr_before_hospitalisation',
                                 'date_of_pcr'], inplace=True)
        # drop the columns that have these keywrods, ^ - starts with, & - ends with
        lab.drop(lab.filter(regex='^result|days_of|_pcr').columns, axis=1, inplace=True)

        # drop the patients that are 'Symptoms' or 'Infected'
        lab = lab[~lab.final_result.str.contains('Symptoms|Infected')]
        lab.drop(labels=['covid-19_signs_in_ct', 'control_ct', 'control_ct.1'],
                 axis=1,
                 inplace=True)

        lab_vars = lab.loc[:, lab.filter(regex='admission').columns]
        lab = lab.loc[:, ['final_result', 'duration_of_hospitalisation_[days]']]
        lab = pd.concat(objs=[lab, lab_vars], axis=1)

        lab.rename(columns={'duration_of_hospitalisation_[days]': 'time', 'final_result': 'outcome'}, inplace=True)
        lab.drop(lab.filter(regex='.[0-9]$').columns, axis=1, inplace=True)
        lab.columns = lab.columns.str.replace('_on_admission_', '_')

        # do the encoding of categorical variables: outcome, sex
        label = LabelEncoder()
        lab['outcome'] = label.fit_transform(lab['outcome'])
        # convert all columns to numeric
        lab[lab.columns] = lab[lab.columns].apply(pd.to_numeric, errors='coerce')

        # appendix values preprocessing ####
        appdx.set_index(keys='Code', inplace=True)
        appdx.drop(appdx.filter(regex='Unnamed').columns, axis=1, inplace=True)
        # convert column names to lower case
        appdx.columns = appdx.columns.str.lower()
        # remove white spaces in column names
        appdx.columns = appdx.columns.str.replace(' ', '_')
        # drop the unneeded columns
        appdx.drop(axis=1, labels=['date_of_hospitalisation', 'final_result', 'sex', 'age'], inplace=True)
        appdx = appdx.loc[:, appdx.filter(regex='admission').columns]
        appdx.columns = appdx.columns.str.replace('_on_admission_', '_')

        # combine the dataframes for complete laboratory values dataframe
        comp_df = pd.concat(objs=[lab, appdx], axis=1)

        # calculations of the ratios of the immune cells ####
        # original features are the drop ####
        comp_df['neu/lymph'] = comp_df['neu_[10^3/ul]'] / comp_df['lymphocytes_[*10^3/ul]']
        comp_df['eos/lymph'] = comp_df['eos_[10^3/ul]'] / comp_df['lymphocytes_[*10^3/ul]']
        comp_df['mono/lymp'] = comp_df['mon_[*10^3/ul]'] / comp_df['lymphocytes_[*10^3/ul]']
        comp_df['plt/lymph'] = comp_df['plt_[*10^3/ul]'] / comp_df['lymphocytes_[*10^3/ul]']
        comp_df['plt/neu'] = comp_df['plt_[*10^3/ul]'] / comp_df['neu_[10^3/ul]']

        # drop the features with more than 40% of NaNs in either of groups
        comp_df.drop(labels=['ck_[u/l]', 'ldh_[u/l]', 'pt_[%]', 'd-dimer_[ug/ml]', 'albumin_[g/dl]',
                             'neu_[10^3/ul]', 'lymphocytes_[*10^3/ul]', 'eos_[10^3/ul]', 'mon_[*10^3/ul]',
                             'plt_[*10^3/ul]'], axis=1, inplace=True)

        # drop the samples that were not in the cohort (Infected/Symptoms)
        comp_df.drop(labels=['SC2_Pos196', 'SC2_Pos41', 'SC2_Pos77'], axis=0, inplace=True)
        print(comp_df.head())

        return comp_df

    def transform_cat_data(self):
        data = self.cat_data
        cats = self.new_cats
        # drop the unnamed columns
        data.drop(data.filter(regex='Unnamed').columns, axis=1, inplace=True)

        # convert column names to lower case
        data.columns = data.columns.str.lower()

        # remove white spaces in column names
        data.columns = data.columns.str.replace(' ', '_')

        # drop the unneeded columns
        data.drop(axis=1, labels=['date_of_hospitalisation', 'ward', 'day_of_transfer_to_icu',
                                  'last_pcr', 'max_mews', 'respirator',
                                  'days_to_negative_pcr_from_the_beginning_of_hospitalisation_or_from_the_positive_pcr_before_hospitalisation',
                                  'date_of_pcr'], inplace=True)

        # set patients ID as index in the dataframe
        # data.set_index('code', inplace = True)
        # drop the columns that have these keywrods, ^ - starts with, & - ends with
        data.drop(data.filter(regex='^result|days_of|_pcr|mews').columns, axis=1, inplace=True)

        # drop the drug columns
        data.drop(axis=1, labels=['acei', 'arb', 'ccb', 'bb', 'mcra', 'diuretics',
                                  'statins', 'ezetimibe', 'vka/doac', 'heparin', 'asa',
                                  'clopidogrel', 'cilostazole', 'paracetamol', 'nsaids',
                                  'sulfasalazine/mesalazine', 'steroids', 'antileukotriene',
                                  'antihistamine', 'other.1', 'methotrexate', 'leflunomide',
                                  'cyklosporyna', 'hydroxychloroquine/chloroquine', 'gcs_inhaled', 'drugs', 'drug_name',
                                  'laba', 'lama', 'saba'], inplace=True)

        # rename the final_result column to outcome
        data.rename(columns={'final_result': 'outcome'}, inplace=True)
        data.rename(columns={'duration_of_hospitalisation_[days]': 'time'}, inplace=True)

        # drop the patients that are 'Symptoms' or 'Infected'
        data = data[~data.outcome.str.contains('Symptoms|Infected')]
        data['sex'] = LabelEncoder().fit_transform(data['sex'])

        # collapse smaller categories into broader
        cats.drop(columns=['number_of_patients', 'Unnamed: 5'], inplace=True)
        cats.rename({'Unnamed: 2': 'cat'}, axis=1, inplace=True)
        cats = cats[cats.cat == 'comorbidity']
        cats.dropna(axis=0, inplace=True)
        cats['new_category'] = cats['new_category'].str.lower()

        # extract the columns in the form of dictionary
        mapping = cats[['category', 'new_category']].set_index('category').to_dict()['new_category']

        # new_cat - new categories after collapsing that will be further used in the analysis
        new_cat = data.set_index(['code', 'outcome', 'sex', 'age', 'time']).groupby(mapping, axis=1).max()
        new_cat.reset_index(inplace=True)
        new_cat = new_cat.set_index('code')
        new_cat.drop(columns=['no', 'outcome', 'time', 'alcoholism'], inplace=True)

        print(new_cat.head())

        return new_cat

    # complete_data = pd.concat([comp_df,new_cat], axis = 1)

import pandas as pd

def preprocess_data(data):
    # Define a dictionary to map old column names to new ones
    column_mapping = {
        'O_us_ovr_sc_worst_del_12m_2': 'S01',
        'O_A_cnt_evr_1MIAp_rt_12m_2': 'S02',
        'O_cl_perc_ovr_ttl_cnt_12m': 'S03',
        'O_A_perc_wrst_1MIAp_12m_2': 'S04',
        'O_A_delBAL_ovr_A_ttlBAL_2': 'S05',
        'O_cl_perc_ovr_cl_lmt_12m': 'S06',
        'O_cc_max_bureau_vintage': 'S07',
        'O_pl_min_bureau_vintage': 'S08',
        'O_cl_ttlBAL_ovr_ttlBAL': 'S09',
        'O_cc_perc_ovr_ttl_cnt': 'S10',
        'O_cc_perc_ovr_ttl_lmt': 'S11',
        'A_cnt_appl_le100k_12m': 'S12',
        'O_cc_max_utilisation': 'S13',
        'O_cc_avg_utilisation': 'S14',
        'O_A_sum_del_rt_12m_2': 'S15',
        'O_us_ovr_sc_balance': 'S16',
        'O_A_mth_snc_1MIAp_2': 'S17',
        'O_A_mth_snc_2MIAp_2': 'S18',
        'O_sc_Coll_Property': 'S19',
        'O_A_decrease_12m_2': 'S20',
        'O_rv_utilisation': 'S21',
        'D_Yr_Emp_Curr': 'S22',
        'D_max_dep_l6': 'S23',
        'O_cc_Limit': 'S24',
        'O_cc_cnt': 'S25'
    }

    # Rename columns
    data.rename(columns=column_mapping, inplace=True)

    # Select relevant columns
    columns_to_keep = ['Bad', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09',
                       'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
                       'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'decision_mth2']
    data = data[columns_to_keep]

    # Drop rows with missing values
    data = data.dropna()

    return data

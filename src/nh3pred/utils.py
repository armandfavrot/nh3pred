import torch
import pandas as pd

def generate_tensors_predictors(df, pmid, device):
    
    data_filtered = df[df['pmid'] == pmid]

    x_cont = data_filtered[['ct', 'dt', 'air_temp', 'wind_2m', 'rain_rate', 'tan_app', 'app_rate', 'man_dm', 'man_ph', 't_incorp']]

    x_cont_tensor = torch.tensor(x_cont.values, dtype=torch.float32).to(device)

    x_cat = data_filtered[['app_mthd', 'incorp', 'man_source']]
    
    x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).to(device)
    x_cat_tensor = torch.unbind (x_cat_tensor, dim = 1)

    output = [x_cont_tensor, x_cat_tensor]
    
    return output

def check_allowed_cols (_df):

    allowed_cols = ['pmid', 'ct', 'air_temp', 'wind_2m', 'rain_rate', 'tan_app', 'app_rate', 'man_dm', 'man_ph', 'app_mthd', 'man_source', 'incorp', 't_incorp']

    invalid_cols = list(set(_df.columns) - set(allowed_cols))

    if invalid_cols:
        raise ValueError(f"The following columns are not allowed: {invalid_cols}")

def check_needed_cols (_df):

    needed_cols = ['pmid', 'ct', 'tan_app']
    
    missing_cols = list (set(needed_cols) - set(_df.columns))
    
    if missing_cols:
        raise ValueError (f"The following column(s) is(are) missing: {missing_cols}")    

def check_nan (_df):

    for col in list (set (_df.columns) - set (['t_incorp'])):

        if _df[col].isna().any():
            raise ValueError(f"NaN values in {col}.")

def check_unique_values_for_plot_level_variables (_df):

    plot_level_variables = ["app_mthd", "man_source", "incorp", "man_ph", "man_dm"]

    pmids = _df['pmid'].unique()
    variables_to_check = list (set (plot_level_variables) & set (_df.columns))

    for var in variables_to_check:
        df_tmp = _df.groupby('pmid')[var].unique()
        for i in pmids:
            if len (df_tmp[i]) != 1:
                raise ValueError (f"Not unique value for {var} in pmid {i}")

def check_correct_values_for_categorical_variables (_df):

    categorical_variables = ['app_mthd', 'incorp', 'man_source']
    
    expected_values = {'app_mthd': ['bc', 'bsth', 'ts', 'os', 'cs'],
                       'incorp': ['none', 'deep', 'shallow'],
                       'man_source': ['pig', 'cat']}
    
    variables_to_check = list (set (categorical_variables) & set (_df.columns))
    
    for var in variables_to_check:
        unvalid_values = list (set (_df[var].unique()) - set (expected_values[var]))
        if unvalid_values:
            raise ValueError(f"The following values are not allowed in {var}: {unvalid_values}")

def check_t_incorp_col (_df):
    if 'incorp' in _df.columns:
        if list (set (_df['incorp'].unique()) - set (["none"])):
            if 't_incorp' not in _df.columns:
                raise ValueError ("Column t_incorp is missing.")
    else:
        if 't_incorp' in _df.columns:
            raise ValueError ("Column t_incorp is not allowed since there is no incorp column.")

def check_t_incorp_values (_df):

    if 't_incorp' in _df.columns:
    
        df_tmp = _df [(_df['incorp'] == "none") & (_df ['t_incorp'].notna())]
        if df_tmp.shape[0] > 0:
            raise ValueError ("t_incorp should be set to NaN when incorp = none.")
        
        df_tmp = _df [(_df['incorp'] != "none") & (_df ['t_incorp'].isna())]
        if df_tmp.shape[0] > 0:
            raise ValueError ("NaN values in t_incorp for incorp != none.")

def complete_incorp (_df):

    if 'incorp' in _df.columns:
        if 't_incorp' in _df.columns:
            _df['t_incorp'] = _df['t_incorp'].fillna (1000)

        elif 't_incorp' not in _df.columns:
            _df["t_incorp"] = [1000 if val == "none" else 0 for val in _df["incorp"]]
            

    elif 'incorp' not in _df.columns:
        _df.insert (len (_df.columns), 'incorp', "none")
        _df.insert (len (_df.columns), 't_incorp', 1000)

    return _df

def complete_other (_df):
    all_columns = ['pmid', 'ct', 'air_temp', 'wind_2m', 'rain_rate', 'app_mthd', 'man_source', 'tan_app', 
                   'app_rate', 'man_dm', 'man_ph']

    default_values = pd.Series({
        'air_temp': 13.89,
        'wind_2m': 3.11,
        'rain_rate': 0,
        'app_mthd': "bsth",
        'man_source' : 'cat',
        'tan_app' : 62.07,
        'app_rate': 29.38,
        'man_dm': 6.25,
        'man_ph': 7.38,
        
    })

    missing_col = list (set (all_columns) - set (_df.columns))

    for col in missing_col:
        _df.insert (len (_df.columns), col, default_values[col])

    return _df

def add_dt (_df):
    _df.insert (1, 'ct_shift', _df.groupby("pmid")["ct"].shift(1, fill_value=0))
    _df['dt'] = _df['ct'] - _df['ct_shift']
    _df = _df.drop (['ct_shift'], axis = 1)
    return _df

def convert_cat_variables (_df):
    _df["app_mthd"] = _df["app_mthd"].map({"bc": 0, "bsth": 1, "os": 2, "ts": 3, "cs": 4})
    _df["man_source"] = _df["man_source"].map({"pig": 0, "cat": 1})
    _df["incorp"] = _df["incorp"].map({"none": 0, "shallow": 1, "deep": 2})

    return _df

def reorder_variables (_df):
    return _df[['pmid', 'ct', 'dt', 'air_temp', 'wind_2m', 'rain_rate', 'app_mthd', 'incorp', 'man_source', 'tan_app', 
                   'app_rate', 'man_dm', 'man_ph', 't_incorp']]

    

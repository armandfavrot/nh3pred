import torch
from importlib.resources import files, as_file

from .model_def import AmmoniaRNN

from .utils import generate_tensors_predictors

from .utils import check_allowed_cols
from .utils import check_needed_cols
from .utils import check_nan 
from .utils import check_unique_values_for_plot_level_variables
from .utils import check_correct_values_for_categorical_variables
from .utils import check_t_incorp_col
from .utils import check_t_incorp_values

from .utils import complete_incorp
from .utils import complete_other
from .utils import add_dt
from .utils import convert_cat_variables
from .utils import reorder_variables

DEVICE = "cpu"
num_layers = 1
nonlinearity = "relu"
bidirectional = True
hidden_size = 512 

cat_dims = [5, 3, 2]  
embedding_dims = [10, 9, 8]  
input_size = 13   
output_size = 1

model = AmmoniaRNN(input_size = input_size, 
                   output_size = output_size, 
                   hidden_size = hidden_size, 
                   nonlinearity = nonlinearity,
                   num_layers = num_layers,
                   bidirectional = bidirectional,
                   cat_dims = cat_dims, 
                   embedding_dims = embedding_dims).to(DEVICE)


resource = files(__package__).joinpath("data").joinpath("final_model.pth")
with as_file(resource) as path:
    model.load_state_dict(torch.load(path, weights_only = True, map_location=torch.device('cpu')))



def predict (df):

    """
    Prediction of ammonia emissions following field fertilization.

    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame containing the environmental conditions for which predictions are made. 
        This DataFrame must include some mandatory columns and may include optional columns.

        Mandatory columns:
            - pmid: identifier for plots
            - ct: time since fertilizer application (h)
            - tan_app: total ammoniacal nitrogen applied (kgN/ha)

        Optional columns: 
            Dynamic variables:
                - air_temp: air temperature (°C)
                - wind_2m: wind speed (m/s)
                - rain_rate: rainfall rate (mm/h)
            Plot-level variables:
                - app_rate: application rate (t/ha)
                - man_dm: manure dry matter content (%)
                - man_ph: manure pH
                - app_mthd: application method (must belong to {bc, bsth, ts, os, cs} (1))
                - man_source: manure source (must belong to {pig, cat})
                - incorp: incorporation (must belong to {none, shallow, deep})
                - t_incorp: time of incorporation (h) (2)

        (1) bc = broadacst, bsth = band spreading trailing hose, ts = trailing shoe, os = open slot, cs = closed slot.

        (2) When incorp = none, t_incorp must be set to NaN.

        Default values for optional columns: air_temp = 13.89, wind_2m = 3.11, rain_rate = 0, app_rate = 29.38, man_dm = 6.25, man_ph = 7.38, app_mthd = bsth, man_source = cat, incorp = none. 

    Returns
    -------
    pandas.DataFrame
        The DataFrame completed with the column 'prediction_ecum' (cumulative prediction, kgN/ha) as well as the optional columns not initially provided, filled with their default values.

    Example
    --------
    >>> # Prediction for two plots, identified with pmid = 1 and pmid = 2
    >>> import pandas as pd
    >>> from nh3pred import predict

    >>> df = pd.DataFrame ({
    ...     "pmid": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    ...     "ct": [3, 6, 10, 24, 48, 72, 1, 3, 6, 10, 21, 46],
    ...     "tan_app": [42, 42, 42, 42, 42, 42, 120, 120, 120, 120, 120, 120],
    ...     "air_temp": [18, 23, 24, 15, 21, 20, 9, 10, 11, 6, 7, 10],
    ...     "wind_2m": [2, 2, 1, 1, 2, 2, 4, 5, 5, 6, 3, 4],
    ...     "rain_rate": 0,
    ...     "app_rate": [20, 20, 20, 20, 20, 20, 28, 28, 28, 28, 28, 28],
    ...     "man_dm": [8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 13, 13, 13, 13, 13, 13],
    ...     "man_ph": [7.1, 7.1, 7.1, 7.1, 7.1, 7.1, 7.7, 7.7, 7.7, 7.7, 7.7, 7.7],
    ...     "app_mthd": ["ts", "ts", "ts", "ts", "ts", "ts", "bc", "bc", "bc", "bc", "bc", "bc"],
    ...     "man_source": ["cat", "cat", "cat", "cat", "cat", "cat", "pig", "pig", "pig", "pig", "pig", "pig"]
    ... })

    >>> predict (df)
        pmid  ct  air_temp  wind_2m  rain_rate app_mthd incorp man_source  tan_app  app_rate  man_dm  man_ph  t_incorp  prediction_ecum
    0      1   3        18        2          0       ts   none        cat       42        20     8.3     7.1       NaN         3.090000
    1      1   6        23        2          0       ts   none        cat       42        20     8.3     7.1       NaN         5.620000
    2      1  10        24        1          0       ts   none        cat       42        20     8.3     7.1       NaN         7.460000
    3      1  24        15        1          0       ts   none        cat       42        20     8.3     7.1       NaN         8.030000
    4      1  48        21        2          0       ts   none        cat       42        20     8.3     7.1       NaN         9.760000
    5      1  72        20        2          0       ts   none        cat       42        20     8.3     7.1       NaN        10.480000
    6      2   1         9        4          0       bc   none        pig      120        28    13.0     7.7       NaN         8.550000
    7      2   3        10        5          0       bc   none        pig      120        28    13.0     7.7       NaN        27.150000
    8      2   6        11        5          0       bc   none        pig      120        28    13.0     7.7       NaN        41.639999
    9      2  10         6        6          0       bc   none        pig      120        28    13.0     7.7       NaN        48.549999
    10     2  21         7        3          0       bc   none        pig      120        28    13.0     7.7       NaN        52.610001
    11     2  46        10        4          0       bc   none        pig      120        28    13.0     7.7       NaN        61.220001
    """

    check_allowed_cols (df)
    check_needed_cols (df)
    check_nan (df)
    check_unique_values_for_plot_level_variables (df)
    check_correct_values_for_categorical_variables (df)
    check_t_incorp_col (df)
    check_t_incorp_values (df)

    data_predictions = df.copy()

    data_predictions = complete_incorp (data_predictions)
    data_predictions = complete_other (data_predictions)
    data_predictions = add_dt (data_predictions)
    data_predictions = convert_cat_variables (data_predictions)
    data_predictions = reorder_variables (data_predictions)


    pmids = data_predictions['pmid'].unique()
    
    data_predictions['prediction_ecum'] = None
    data_predictions['prediction_delta_ecum'] = None
        
    with torch.no_grad():
    
        all_predictions = torch.empty(0).to(DEVICE)
    
            
        for i in pmids:
    
            x = generate_tensors_predictors (data_predictions, i, device = DEVICE)
            y = model(x)
            all_predictions = torch.cat ((all_predictions, y.squeeze()), 0)
    
        data_predictions['prediction_delta_ecum'] = all_predictions.to("cpu").detach()
    
    data_predictions['prediction_ecum'] = data_predictions.groupby('pmid')['prediction_delta_ecum'].cumsum()
    data_predictions["prediction_ecum"] = data_predictions["prediction_ecum"].round(2)

    data_predictions = data_predictions.drop (['dt', 'prediction_delta_ecum'], axis = 1)


    data_predictions["app_mthd"] = data_predictions["app_mthd"].map({0: "bc", 1: "bsth", 2: "os", 3: "ts", 4:"cs"})
    data_predictions["man_source"] = data_predictions["man_source"].map({0: "pig", 1: "cat"})
    data_predictions["incorp"] = data_predictions["incorp"].map({0: "none", 1: "shallow", 2: "deep"})
    data_predictions["t_incorp"] = data_predictions["t_incorp"].replace(1000, float("nan"))

    return data_predictions

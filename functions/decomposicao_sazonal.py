import  warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

def decomposicao_sazonal(df):
    
    df_decompose = seasonal_decompose(df, model='additive', period=30)
    
    return df_decompose

    
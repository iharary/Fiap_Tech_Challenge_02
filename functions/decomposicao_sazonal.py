import  warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

def decomposicao_sazonal(dataframe, modelo, periodo):
    
    df_decompose = seasonal_decompose(df=dataframe, model=modelo, period=periodo)
    
    return df_decompose

    
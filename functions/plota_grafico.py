import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_seasonal_decompose(df, column, model='additive', period=30):
    # Realiza a decomposição sazonal
    result = seasonal_decompose(df[column], model=model, period=period)

    # Plota o resultado da decomposição
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(15,8))
    result.observed.plot(ax=ax1)
    result.trend.plot(ax=ax2)
    result.seasonal.plot(ax=ax3)
    result.resid.plot(ax=ax4)
    plt.tight_layout()

    # Usa st.pyplot() para renderizar a figura no Streamlit
    st.pyplot(fig)
    
def plot_stats_forecast(model, treino, forecast):
        
    df = model    
    df.plot(treino, forecast, level=[90], unique_ids=['MEATS', 'PERSONAL CARE'],engine ='matplotlib', max_insample_length=90)
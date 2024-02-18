from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage, AutoARIMA
from functions.erro_medio_absoluto import wmape

def naive_model(treino, valid, h=30):
        
    model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    #forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
    forecast_df = forecast_df.reset_index().merge(valid, on=['Date'], how='left')
    
    wmape1 = wmape(forecast_df['y'].values, forecast_df['Naive'].values)
    #print(f"WMAPE: {wmape1:.2%}")

    return wmape1
   
def stats_forecast_model(treino, valid, h=30):
        
    model = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    #forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape2 = wmape(forecast_df['y'].values, forecast_df['SeasonalNaive'].values)
    print(f"WMAPE: {wmape2:.2%}")

    #model.plot(treino, forecast_dfs, level=[90], unique_ids=['MEATS', 'PERSONAL CARE'],engine ='matplotlib', max_insample_length=90)
    
    
def seasonal_window_average_model(treino, valid, h=30):
        
    model = StatsForecast(models=[SeasonalWindowAverage(season_length=7, window_size=2)], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape3 = wmape(forecast_df['y'].values, forecast_df['SeasWA'].values)
    print(f"WMAPE: {wmape3:.2%}")

    #model_sm.plot(treino, forecast_dfsm, level=[90], unique_ids=['MEATS', 'PERSONAL CARE'],engine ='matplotlib', max_insample_length=90)
    
    
def auto_arima_model(treino, valid, h=30):
        
    model = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape4 = wmape(forecast_df['y'].values, forecast_df['AutoARIMA'].values)
    print(f"WMAPE: {wmape4:.2%}")

    #model_a.plot(treino, forecast_dfa, level=[90], unique_ids=['MEATS', 'PERSONAL CARE'],engine ='matplotlib', max_insample_length=90)
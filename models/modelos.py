from statsforecast import StatsForecast
from statsforecast.models import Naive, AutoARIMA
from prophet import Prophet

def naive_model(treino, valid, h):
        
    model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
    
    return model, forecast_df

def auto_arima_model(treino, valid, h):
        
    model = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_df = model.predict(h=h, level=[90])
    forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
    
    return model, forecast_df

def prophet_model(treino, valid, h, cps, sps):
        
    model = Prophet(daily_seasonality=True,seasonality_mode='multiplicative',changepoint_prior_scale=cps,seasonality_prior_scale=sps)
    model.fit(treino)

    future = model.make_future_dataframe(periods=h)
    forecast = model.predict(future)

    # fazendo previsões com os dados de teste
    test_forecast = model.predict(valid)
    # olhando os resultados das previsões com os dados de teste
    forecast_df = test_forecast.reset_index().merge(valid, on=['ds'], how='left')

    return model, forecast, forecast_df



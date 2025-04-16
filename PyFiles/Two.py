
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

if not os.path.exists('graphs_arima'):
    os.makedirs('graphs_arima')

data = pd.read_csv('dataset/Time_Series.csv', delimiter=';', decimal=',')
data['Год'] = pd.to_datetime(data['Год'], format='%Y').dt.year
data.set_index('Год', inplace=True)

columns_to_forecast = ['oil_prod', 'gas_prod', 'oil_export', 
                       'energy_consumption', 'investment', 
                       'oil_price', 'tax_revenue']

forecast_year = 2035
start_year = data.index.min()
end_year = data.index.max()
n_years = forecast_year - end_year

results = {}
def forecast_arima(series, column_name):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=n_years)
    forecast_year_value = forecast.iloc[-1]

    combined = pd.concat([series, forecast])
    
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, label='Фактические данные', marker='o')
    plt.plot(combined.index, combined.values, label='Прогноз (ARIMA)', linestyle='--', marker='x')
    plt.axvline(x=end_year, color='gray', linestyle='--', label='Начало прогноза')
    plt.scatter(forecast_year, forecast_year_value, color='green', label='Прогноз 2035')
    plt.title(f'ARIMA прогноз для {column_name}')
    plt.xlabel('Год')
    plt.ylabel(column_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'graphs_arima/{column_name}_arima_forecast.png')
    plt.close()
    
    print(f"\nARIMA прогноз для {column_name}: {forecast_year_value:.2f}")
    return forecast_year_value

for col in columns_to_forecast:
    results[col] = forecast_arima(data[col], col)

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.title('Прогноз показателей на 2035 год (ARIMA)')
plt.ylabel('Значение')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('graphs_arima/all_predictions_arima.png')
plt.close()

print("\nИтоговый прогноз на 2035 год (ARIMA):")
print("="*50)
for k, v in results.items():
    print(f"{k}: {v:.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

if not os.path.exists('graphs'):
    os.makedirs('graphs')

data = pd.read_csv('dataset/Time_Series.csv', delimiter=';', decimal=',')
data['Год'] = pd.to_datetime(data['Год'], format='%Y').dt.year
data.set_index('Год', inplace=True)

def forecast_linear(df, column_name):
    X = np.array(df.index).reshape(-1, 1)
    y = df[column_name].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    year_2035 = np.array([[2035]])
    prediction = model.predict(year_2035)[0]
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\nАнализ для показателя: {column_name}")
    print("="*50)
    print(f"Коэффициенты: intercept={model.intercept_:.2f}, slope={model.coef_[0]:.2f}")
    print(f"Среднеквадратичная ошибка: {mse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"Прогноз на 2035 год: {prediction:.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Фактические данные')
    plt.plot(X, y_pred, color='red', label='Линия регрессии')
    plt.scatter(year_2035, prediction, color='green', label='Прогноз на 2035')
    plt.title(f'Линейная регрессия для {column_name}')
    plt.xlabel('Год')
    plt.ylabel(column_name)
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{column_name}_regression.png')
    plt.close()
    
    return prediction

results = {}
columns_to_forecast = ['oil_prod', 'gas_prod', 'oil_export', 
                      'energy_consumption', 'investment', 
                      'oil_price', 'tax_revenue']

for col in columns_to_forecast:
    results[col] = forecast_linear(data, col)

print("\nИтоговый прогноз на 2035 год:")
print("="*50)
for k, v in results.items():
    print(f"{k}: {v:.2f}")

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.title('Прогноз показателей на 2035 год (линейная регрессия)')
plt.ylabel('Значение')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('graphs/all_predictions.png')
plt.close()

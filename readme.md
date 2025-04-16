# Прогноз топливно-энергетического баланса Ханты-Мансийского автономного округа до 2035 года

## Описание проекта
Прогнозирование ключевых показателей топливно-энергетического комплекса с использованием линейной регрессии на основе исторических данных за 2017–2023 годы.

## Методология
- Использована простая линейная регрессия 
- Для каждого показателя построена отдельная модель
- Оценка качества через R² и MSE
- Прогноз выполнен на 12 лет вперёд (до 2035 года)

## Результаты прогнозирования


# Прогноз топливно-энергетического баланса Ханты-Мансийского АО до 2035 года

### Нефтедобыча (oil_prod)
- **Прогноз на 2035:** 164.16 млн тонн
- **Тренд:** -4.25 млн тонн/год
- **Точность модели (R²):** 0.80

### Добыча газа (gas_prod)
- **Прогноз на 2035:** 28.62 млрд м³
- **Тренд:** -0.46 млрд м³/год
- **Точность модели (R²):** 0.77

### Экспорт нефти (oil_export)
- **Прогноз на 2035:** 129.82 млн тонн
- **Тренд:** -3.46 млн тонн/год
- **Точность модели (R²):** 0.80

### Потребление энергии (energy_consumption)
- **Прогноз на 2035:** 37.66
- **Тренд:** -0.37/год
- **Точность модели (R²):** 0.80

### Инвестиции (investment)
- **Прогноз на 2035:** 193.11 млрд руб
- **Тренд:** -1.11 млрд руб/год
- **Точность модели (R²):** 0.14

### Цена нефти (oil_price)
- **Прогноз на 2035:** $87.86/барр
- **Тренд:** +1.70 $/год
- **Точность модели (R²):** 0.10

### Налоговые доходы (tax_revenue)
- **Прогноз на 2035:** 289.11 млрд руб
- **Тренд:** -1.96 млрд руб/год
- **Точность модели (R²):** 0.32


### Основные показатели
| Показатель            | Прогноз на 2035 | Изменение к 2023 | Качество модели (R²) |
|-----------------------|----------------|------------------|----------------------|
| Добыча нефти (млн т)  | 164.16         | 24%              | 0.80                 |
| Добыча газа (млрд м³) | 28.62          | 16%              | 0.77                 |
| Экспорт нефти (млн т) | 129.82         | 25%              | 0.80                 |
| Потребление энергии   | 37.66          | 10%              | 0.80                 |
| Инвестиции (млрд руб) | 193.11         | 7%               | 0.14                 |
| Цена нефти ($/барр)   | 87.86          | 40%              | 0.10                 |
| Налоговые доходы      | 289.11         | 7%               | 0.32                 |

## Визуализация результатов

### Тренды показателей
![1](/graphs/oil_prod_regression.png)  

![2](/graphs/gas_prod_regression.png)  

![3](/graphs/oil_price_regression.png)

### Сравнительный прогноз
![Все показатели на 2035 год](graphs/all_predictions.png)

## Интерпретация результатов
1. **Высокая точность моделей** (R² > 0.75) для:
   - Добычи нефти и газа
   - Экспорта нефти
   - Потребления энергии

2. **Низкая точность моделей** (R² < 0.35) для:
   - Инвестиций
   - Цен на нефть
   - Налоговых доходов

3. Основные тренды:
   - Снижение добычи углеводородов
   - Рост цен на нефть
   - Снижение налоговых поступлений

## Ограничения
- Линейная регрессия не учитывает внешние факторы
- Долгосрочные прогнозы могут быть неточными
- Для инвестиций и цен требуются более сложные модели

## Результаты прогнозов на 2035 год

| Показатель            | Линейная регрессия | ARIMA    | Разница | Качество (R²) |
|-----------------------|--------------------|----------|---------|---------------|
| Добыча нефти (млн т) | 164.16             | 214.88   | +50.72  | 0.80          |
| Добыча газа (млрд м³) | 28.62              | 34.09    | +5.47   | 0.77          |
| Экспорт нефти (млн т) | 129.82             | 170.99   | +41.17  | 0.80          |
| Потребление энергии   | 37.66              | 41.94    | +4.28   | 0.80          |
| Инвестиции (млрд руб) | 193.11             | 209.68   | +16.57  | 0.14          |
| Цена нефти ($/барр)   | 87.86              | 62.37    | -25.49  | 0.10          |
| Налоговые доходы      | 289.11             | 315.51   | +26.40  | 0.32          |

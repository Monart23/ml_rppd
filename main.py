import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Функция для генерации синтетических данных
def generate_data(filename='synthetic_data.xlsx', num_samples=1000, seed=42):
    np.random.seed(seed)

    # Генерируем входные параметры
    P1 = np.random.uniform(10, 100, num_samples)  # Давление 1 (кПа)
    P2 = P1 - np.random.uniform(1, 10, num_samples)  # Давление 2 (кПа)
    T1 = np.random.uniform(20, 80, num_samples)  # Температура 1 (°C)
    T2 = T1 + np.random.uniform(-5, 5, num_samples)  # Температура 2 (°C)

    # Целевой показатель: расход (условная формула)
    dP = P1 - P2
    Flow = 0.1 * np.sqrt(dP) * (1 + 0.01 * (T1 + T2) / 2)

    # Добавляем шум
    Flow += np.random.normal(0, 0.05, num_samples)

    # Собираем всё в DataFrame
    data = pd.DataFrame({
        'P1': P1,
        'P2': P2,
        'T1': T1,
        'T2': T2,
        'Flow': Flow
    })

    # Сохраняем данные в Excel
    data.to_excel(filename, index=False)
    print(f"Синтетические данные сохранены в {filename}")


# Основная функция
def main():
    # Проверяем, нужно ли сгенерировать новые данные
    generate_new = input("Сгенерировать новые данные? (да/нет): ").strip().lower()

    filename = 'synthetic_data.xlsx'
    if generate_new in ['да', 'yes', 'y']:
        generate_data(filename=filename)
    elif not os.path.exists(filename):
        print(f"Файл {filename} не найден. Генерация новых данных.")
        generate_data(filename=filename)
    else:
        print(f"Используем существующий файл {filename}.")

    # Загружаем данные
    data = pd.read_excel(filename)
    print("Первые строки данных:")
    print(data.head())

    # Разделяем данные
    X = data[['P1', 'P2', 'T1', 'T2']].values
    y = data['Flow'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабируем данные
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Создаём и обучаем модель
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Обучение модели...")
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

    # Оценка модели
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # График предсказаний
    y_pred = model.predict(X_test_scaled).flatten()

    import matplotlib.pyplot as plt
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Flow")
    plt.ylabel("Predicted Flow")
    plt.title("Actual vs Predicted Flow")
    plt.show()


# Запуск приложения
if __name__ == "__main__":
    main()

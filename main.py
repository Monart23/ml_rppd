import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

def generate_data(filename='synthetic_data.xlsx', num_samples=1000, seed=42):
    """Генерация синтетических данных и сохранение их в Excel."""
    np.random.seed(seed)
    P1 = np.random.uniform(10, 100, num_samples)  # Давление 1 (кПа)
    P2 = P1 - np.random.uniform(1, 10, num_samples)  # Давление 2 (кПа)
    T1 = np.random.uniform(20, 80, num_samples)  # Температура 1 (°C)
    T2 = T1 + np.random.uniform(-5, 5, num_samples)  # Температура 2 (°C)
    dP = P1 - P2
    Flow = 0.1 * np.sqrt(dP) * (1 + 0.01 * (T1 + T2) / 2)  # Формула расхода
    Flow += np.random.normal(0, 0.05, num_samples)  # Добавляем шум
    data = pd.DataFrame({'P1': P1, 'P2': P2, 'T1': T1, 'T2': T2, 'Flow': Flow})
    data.to_excel(filename, index=False)
    print(f"Синтетические данные сохранены в {filename}")

def load_and_preprocess_data(filename):
    """Загрузка данных и их предварительная обработка."""
    data = pd.read_excel(filename)
    print("Первые строки данных:")
    print(data.head())
    X = data[['P1', 'P2', 'T1', 'T2']].values
    y = data['Flow'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    """Масштабирование данных."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_model(input_shape):
    """Создание архитектуры модели."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_training_and_predictions(history, y_actual, y_predicted):
    """Построение графиков обучения и фактических vs предсказанных значений."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # График обучения
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid()

    # График фактических vs предсказанных значений
    axs[1].scatter(y_actual, y_predicted, alpha=0.7)
    axs[1].set_xlabel('Actual Flow')
    axs[1].set_ylabel('Predicted Flow')
    axs[1].set_title('Actual vs Predicted Flow')
    axs[1].grid()

    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, save_path=None):
    """Обучение и оценка модели."""
    print("Обучение модели...")
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

    y_predicted = model.predict(X_test).flatten()
    plot_training_and_predictions(history, y_test, y_predicted)

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    if save_path:
        model.save(save_path)
        print(f"Модель сохранена в '{save_path}'")

def main(args):
    if args.generate or not os.path.exists(args.file):
        if not os.path.exists(args.file):
            print(f"Файл {args.file} не найден. Генерация новых данных.")
        generate_data(filename=args.file, num_samples=args.samples, seed=args.seed)
    else:
        print(f"Используем существующий файл {args.file}.")

    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.file)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    model = build_model(input_shape=X_train_scaled.shape[1])
    train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, save_path=args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Модель прогнозирования расхода по перепаду давления и температуре.")
    parser.add_argument('--generate', action='store_true', help="Генерировать новые данные.")
    parser.add_argument('--file', type=str, default='synthetic_data.xlsx', help="Файл с данными. По умолчанию: 'synthetic_data.xlsx'.")
    parser.add_argument('--samples', type=int, default=1000, help="Количество сэмплов для генерации. По умолчанию: 1000.")
    parser.add_argument('--seed', type=int, default=42, help="Значение для инициализации генератора случайных чисел. По умолчанию: 42.")
    parser.add_argument('--save_model', type=str, default=None, help="Имя файла для сохранения обученной модели.")
    args = parser.parse_args()
    main(args)

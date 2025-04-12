from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from stock_data_fetcher import fetch_stock_data, preprocess_data

def train_model():
    df = fetch_stock_data()
    if df is None:
        return None

    X, y, _, _ = preprocess_data(df)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X[:-365], y[:-365], epochs=50, batch_size=32, verbose=0)
    return model

def predict_next_day(model):
    df = fetch_stock_data()
    if df is None:
        return "Error fetching data"

    X, _, scaler, _ = preprocess_data(df)
    latest_input = X[-1].reshape(1, -1)

    predicted_price = model.predict(latest_input)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]
    return f"Next day's predicted closing price: ${predicted_price:.2f}"

def plot_predicted_vs_actual(model):
    import matplotlib.pyplot as plt

    df = fetch_stock_data()
    if df is None:
        return "Error fetching data"

    X, y, scaler, df_full = preprocess_data(df)
    y_pred = model.predict(X)

    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    y_predicted = scaler.inverse_transform(y_pred.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(df_full.index[-100:], y_actual[-100:], label="Actual Price", color='black')
    plt.plot(df_full.index[-100:], y_predicted[-100:], label="Predicted Price", color='blue')
    plt.title("Predicted vs. Actual Stock Prices")
    plt.legend()
    plt.savefig("predicted_vs_actual.png")
    return "predicted_vs_actual.png"

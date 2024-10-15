from flask import Flask, request, jsonify
from crypto import get_stock_data
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

@app.route('/results', methods=['POST'])
def results():
    results = {}
    data = request.json

    # استلام العملة، الفترة الزمنية
    symbol = data.get('symbol')  # العملة المطلوبة
    period = data.get('period', '1d')  # الفترة الزمنية الافتراضية هي يوم واحد

    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    try:
        # جلب البيانات التاريخية بناءً على المدخلات
        stock_data = get_stock_data(symbol, period)
        
        if 'error' in stock_data:
            return jsonify({"error": stock_data['error']}), 400

        close_prices = stock_data['Close']
        if close_prices.isnull().any():
            close_prices = close_prices.dropna()

        if len(close_prices) < 10:
            return jsonify({"error": "Insufficient data"}), 400
        
        train_data = close_prices

        # تدريب نموذج ARIMA
        model_arima = ARIMA(train_data, order=(5, 1, 0))
        fitted_model_arima = model_arima.fit()
        forecast_arima = fitted_model_arima.forecast(steps=1)[0]
        
        # تدريب نموذج XGBoost
        X = np.arange(len(train_data)).reshape(-1, 1)
        y = train_data.values
        model_xgboost = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model_xgboost.fit(X, y)
        next_index = np.array([[len(train_data)]])
        forecast_xgboost = model_xgboost.predict(next_index)[0]

        # جمع التوقعات السابقة لعمل بيانات زمنية لنموذج LSTM
        arima_forecasts = []
        xgboost_forecasts = []
        for i in range(10, len(train_data)):
            # توقعات ARIMA
            model_arima = ARIMA(train_data[:i], order=(5, 1, 0))
            fitted_model_arima = model_arima.fit()
            arima_forecasts.append(fitted_model_arima.forecast(steps=1)[0])
            
            # توقعات XGBoost
            X_train = np.arange(i).reshape(-1, 1)
            y_train = train_data[:i].values
            model_xgboost.fit(X_train, y_train)
            next_idx = np.array([[i]])
            xgboost_forecasts.append(model_xgboost.predict(next_idx)[0])

        # تحويل التوقعات السابقة إلى مصفوفة لتمثيل المدخلات
        arima_forecasts = np.array(arima_forecasts)
        xgboost_forecasts = np.array(xgboost_forecasts)

        # مصفوفة المدخلات للنموذج التجميعي (التوقعات السابقة من ARIMA و XGBoost)
        X_stacking = np.column_stack((arima_forecasts, xgboost_forecasts))

        # الهدف (القيم الفعلية) - نستخدم القيم الحقيقية كهدف لتدريب LSTM
        y_stacking = train_data[10:].values

        # مقياس البيانات إلى نطاق [0, 1] قبل استخدامها في LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_stacking_scaled = scaler.fit_transform(X_stacking)
        y_stacking_scaled = scaler.transform(y_stacking.reshape(-1, 1))

        # تحضير البيانات لـ LSTM
        X_lstm, y_lstm = [], []
        for i in range(10, len(X_stacking_scaled)):
            X_lstm.append(X_stacking_scaled[i-10:i])
            y_lstm.append(y_stacking_scaled[i])
        
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        # تدريب نموذج LSTM التجميعي
        lstm_model = tf.keras.Sequential()
        lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
        lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
        lstm_model.add(tf.keras.layers.Dense(units=25))
        lstm_model.add(tf.keras.layers.Dense(units=1))

        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_lstm, y_lstm, epochs=1, batch_size=1, verbose=2)

        # توقع باستخدام LSTM بناءً على التوقعات الحالية من ARIMA و XGBoost
        current_input = np.array([[forecast_arima, forecast_xgboost]])
        current_input_scaled = scaler.transform(current_input)
        current_input_lstm = np.reshape(current_input_scaled, (1, current_input_scaled.shape[0], current_input_scaled.shape[1]))

        forecast_lstm = lstm_model.predict(current_input_lstm)
        forecast_lstm = scaler.inverse_transform(forecast_lstm)[0][0]

        # إضافة النتائج إلى الاستجابة
        results["current_price"] = float(close_prices.iloc[-1])
        results["forecast_arima"] = float(forecast_arima)
        results["forecast_xgboost"] = float(forecast_xgboost)
        results["forecast_lstm"] = float(forecast_lstm)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

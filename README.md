Stock Predictor


Stock adv-ai-ser  

This project is an early  model in a final year project I’m working on. Since I’m using the spiral development method, I’ll update this consistently, though I’ve got more advanced, completed models in the vault.  

 About the Project  
This stock predictor pulls data from Alpha Vantage, applies technical indicators like EMA, MACD, and Stochastic Oscillator, then runs that data through a neural network built with TensorFlow and Keras to predict the next day’s closing price. It also has a Gradio UI to make it easier to use.  

 Setup & Installations  
I originally built this in Google Colab, which made setting everything up simpler. Moving it to a local setup, especially on Windows, caused some issues with imports — that’s on me. I’ve cleaned things up, but here’s what you need to install:  

```bash
pip install gradio alpha_vantage pandas numpy matplotlib tensorflow keras ta scikit-learn
```

If you run into errors with TensorFlow or NumPy like i did, try this:  

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

This keeps everything compatible with TensorFlow.  

---

 How to Run  

Clone the repo:
   ```bash
   git clone https://github.com/yourusername/stock-adv-ai-ser.git
   cd stock-adv-ai-ser
   ```

Make sure your folder structure looks like this:
   ```
   /stock_predictor_app
       ├── stock_data_fetcher.py
       ├── stock_predictor.py
       └── stock_app.py
   ```

run the app:
   ```bash
   python stock_app.py
   ```

It should open the Gradio interface in your browser.  

---

Features  
- Real-time data fetching from Alpha Vantage  
- Technical indicators: 10-day EMA, 50-day EMA, MACD, Stochastic  
- Neural network prediction with TensorFlow/Keras  
- Gradio UI for easier interaction  
- TradingView widget for live charts  

---

Improvements & Future Plans  
- Better error handling for API limits and missing data  
- Use ANN
- -Add questionairee
- in general update to my more complete recent model 
- Support for multiple stocks, not just TSLA  
- Hyperparameter tuning for better accuracy  

---

Feel free to fork this, experiment, and improve on it. If you end up with something better or more interesting, I’d love to hear about it.  


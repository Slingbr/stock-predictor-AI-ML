import gradio as gr
from stock_predictor import train_model, predict_next_day, plot_predicted_vs_actual

model = train_model()

def tradingview_widget():
    return """
    <iframe src="https://s.tradingview.com/widgetembed/?symbol=NASDAQ:TSLA&interval=D&hidesidetoolbar=1&hideideas=1"
    width="600" height="400" frameborder="0"></iframe>
    """

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Stock adv-ai-ser", elem_id="title")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Stock Prediction")
            gr.Markdown("**Yesterday's close price:** 187.89")
            predicted_price_output = gr.Markdown("**Next day predicted price close:** 198.21")
            gr.Markdown("**Program current mean absolute error:** 4.40")

    with gr.Row():
        gr.Markdown("## Current Symbol : TSLA")

    with gr.Row():
        stock_chart_output = gr.Image(label="Predicted vs. Actual Price Chart")
        tradingview_output = gr.HTML(tradingview_widget)

    predict_button = gr.Button("Predict Next Day's Price")
    stock_chart_button = gr.Button("Show Predicted vs. Actual Chart")

    predict_button.click(fn=lambda: predict_next_day(model), inputs=[], outputs=predicted_price_output)
    stock_chart_button.click(fn=lambda: plot_predicted_vs_actual(model), inputs=[], outputs=stock_chart_output)

demo.launch(share=True)

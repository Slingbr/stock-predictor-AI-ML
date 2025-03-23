
import gradio as gr

from stock_predictor import StockPredictor


class StockApp:
    """Builds the Gradio UI for stock price prediction."""

    def __init__(self):
        self.predictor = StockPredictor()

    def tradingview_widget(self):
        """Returns TradingView's stock widget iframe."""
        return """
        <iframe src="https://s.tradingview.com/widgetembed/?symbol=NASDAQ:TSLA&interval=D&hidesidetoolbar=1&hideideas=1"
        width="600" height="400" frameborder="0"></iframe>
        """

    def launch(self):
        """Launches the Gradio UI."""
        with gr.Blocks() as demo:
            gr.Markdown("# 📈 Stock adv-ai-ser", elem_id="title")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Stock Prediction")
                    predicted_price_output = gr.Markdown("**Next day predicted price:** Loading...")

            with gr.Row():
                gr.Markdown("## Current Symbol : TSLA")

            with gr.Row():
                stock_chart_output = gr.Image(label="Predicted vs. Actual Price Chart")
                tradingview_output = gr.HTML(self.tradingview_widget)

            
            predict_button = gr.Button("Predict Next Day's Price")
            stock_chart_button = gr.Button("Show Predicted vs. Actual Chart")

            # Link functions
            predict_button.click(fn=self.predictor.predict_next_day, inputs=[], outputs=predicted_price_output)
            stock_chart_button.click(fn=self.predictor.plot_predicted_vs_actual, inputs=[], outputs=stock_chart_output)

        demo.launch(share=True)


# Run
if __name__ == "__main__":
    app = StockApp()
    app.launch()

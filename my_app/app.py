from shiny import App, render, ui
from matplotlib import pyplot as plt


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"

    @output
    @render.plot
    def viz():
        fig, ax = plt.subplots()
        a = input.n()
        ax.plot([0, 0], [0, 1 - a])
        ax.plot([1, 1], [0, a])
        return fig


app_ui = ui.page_fluid(
    ui.input_slider("n", "p", 0, 1, 0.5),
    ui.h2("Функция вероятности"),
    #ui.output_text_verbatim("txt"),
    ui.output_plot("viz"),
)

    

app = App(app_ui, server)
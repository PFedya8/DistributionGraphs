import numpy as np
from shiny import App, render, ui
from matplotlib import pyplot as plt



app_ui = ui.page_fluid(
    ui.h2("Graphs"),
    ui.input_slider("n", "N", 0, 100, 20),
    ui.output_text_verbatim("txt"),
    ui.output_plot("plot"),
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"
        




app = App(app_ui, server)
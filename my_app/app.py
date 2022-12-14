from shiny import *
import numpy as np
import math


import pandas as pd
from matplotlib import pyplot as plt


app_ui = ui.page_fluid(
    # First page
    # For latex
    ui.head_content(
        ui.tags.script(
            src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
            # src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        ui.tags.script(
            "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
        ),
    ),
    # Title
    
    
    ui.navset_tab(
        ui.nav(
            "Коши",
            ui.input_slider("couchy1", "\(x_0\)", 0, 10, 0),
            ui.input_numeric("couchy2", "\(\gamma\)", value= 1),

            ui.h4("Параметры:"),
            ui.h5("\(C(x_0, \gamma) \)"),
            ui.h5("\(x_0 \ - коэффициент \ сдвига \)"),
            ui.h5("\(\gamma > 0 \ - коэффициент \ масштаба \)"),

            ui.h3("Плотность вероятности:"),
            ui.h5("\( \mathbb{P}(\\xi) = \\frac{1}{\pi \gamma[1 + (\\frac{x-x_0}{\gamma})^2]}\)"),
            ui.output_plot("couchy_distr1"),

            ui.h3("Функция распределения"),
            ui.h5("\( \mathbb{F}(x) = \\frac{1}{\pi} \\arctan({\\frac{x - x_0}{\gamma}}) + \\frac{1}{2} \)"),
            ui.output_plot("couchy_distr2"),

            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: не \ существует\)"),
                ui.tags.li("\(Медиана: x_0 \)"),
                ui.tags.li("\(Мода: x_0 \)"),
                ui.tags.li("\(Дисперсия: не \ существует \)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: не \ существует\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: не \ существует\)"),
            ),
        ),
    ),
    
)

def server(input, output, session):
    @output
    @render.plot
    def couchy_distr1():
        fig, ax = plt.subplots()
        x_0 = input.couchy1()
        gamma = input.couchy2()

        plt.grid()
        x = np.linspace(-5, 5, 100)
        help_x = ((x - x_0) / gamma) ** 2
        y = 1 / (math.pi * gamma * (1 + help_x))
        ax.plot(x, y)
        return fig
    
    @output
    @render.plot
    def couchy_distr2():
        fig, ax = plt.subplots()
        x_0 = input.couchy1()
        gamma = input.couchy2()

        plt.grid()
        x = np.linspace(-5, 5, 100)
        help_x = np.zeros(100)
        for i in range(100):
            help_x[i] = math.atan((x[i] - x_0) / gamma)
        y = 1 / math.pi * help_x + 1 / 2
        ax.plot(x, y)
        return fig

    

app = App(app_ui, server)
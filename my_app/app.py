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
            "Парето",
            ui.input_slider("pareto1", "\(k\)", 0, 10, 5),
            ui.input_slider("pareto2", "\(x_m\)", 0, 10, 5),

            ui.h4("Параметры:"),
            ui.h5("\(P(k, x_m) \)"),
            ui.h5("\(x_m \ - коэффициент \ масштаба \)"),

            ui.h3("Плотность вероятности:"),
            ui.h5("\( \mathbb{P}(\\xi) = \\frac{k x_m}{x^{k+1}}, \ x \geq x_m \)"),
            ui.output_plot("pareto_distr1"),

            ui.h3("Функция распределения"),
            ui.h5("\( \mathbb{F}(x) = 1 - (\\frac{x_m}{x})^k \)"),
            ui.output_plot("pareto_distr2"),

            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\frac{kx_m}{k-1}, \ если \ k > 1\)"),
                ui.tags.li("\(Медиана: x_m \sqrt[k]{2}\)"),
                ui.tags.li("\(Мода: x_m\)"),
                ui.tags.li("\(Дисперсия: (\\frac{x_m}{k-1})^2 \\frac{k}{k-2}, \ при \ k \ > \ 2 \)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\frac{2(1+k)}{k-3} \sqrt{\\frac{k-2}{k}}, \ при \ k \ > \ 3 \)"),
                ui.tags.li("\(Коэфициент \ эксцесса: \\frac{6(k^3 + k^2 - 6k -2)}{k(k-3)(k-4)}, \ при \ k \ > \ 4 \)"),
            ),
        ),
    ),
    
)

def server(input, output, session):
    @output
    @render.plot
    def pareto_distr1():
        fig, ax = plt.subplots()
        k = input.pareto1()
        x_m = input.pareto2()

        plt.grid()
        x = np.linspace(x_m, 20, 100)
        y = k * (x_m ** k) / (x ** (k + 1))
        ax.plot(x, y)
        ax.plot([x_m, x_m], [0, k])
        return fig
    
    @output
    @render.plot
    def pareto_distr2():
        fig, ax = plt.subplots()
        k = input.pareto1()
        x_m = input.pareto2()

        plt.grid()
        x = np.linspace(x_m, 20, 100)
        y = 1 - (x_m / x) ** k
        ax.plot(x, y)
        ax.plot([x_m, x_m], [0, 1])
        return fig

    

app = App(app_ui, server)
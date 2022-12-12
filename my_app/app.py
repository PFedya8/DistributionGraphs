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
            "Нормальное распределение (гауссовское)",
            ui.input_slider("normal1", "p", 0, 1, 0.5),
            ui.input_slider("normal2", "\(\mu\)", -5, 5, 0),
            ui.input_slider("normal3", "\(\sigma\)", 0, 10, 5),

            ui.h4("Параметры:"),
            ui.h5("\(N(\mu, \sigma^2) \)"),
            ui.h5("\(\mu \ - коэффициент \ сдвига \)"),
            ui.h5("\(\sigma > 0 \ - коэффициент \ масштаба \)"),

            ui.h3("Плотность вероятности:"),
            ui.h5("\( \mathbb{P}(\\xi) = \\frac{1}{\sigma \sqrt{2 \pi}} e^{-\\frac{(x-\mu)^2}{2\sigma^2}} \)"),
            ui.output_plot("normal_distribution1"),

            ui.h3("Функция распределения"),
            ui.h5("\( \mathbb{F}(\\xi) = 1 - q^{n+1}\)"),
            ui.output_plot("normal_distribution2"),

            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\frac{q}{p} \)"),
                ui.tags.li("\(Дисперсия: \\frac{q}{p^2} \)"),
                ui.tags.li("\(Мода: 0\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\frac{2-p}{\\sqrt{1-p}} \)"),
                ui.tags.li("\(Коэфициент \ эксцесса: 6 + \\frac{p^2}{1-p} \)"),
            ),

            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Плотность: P(\{k\}) = (1 - p)^{k-1}p \)"),
            ),
        ),
    ),
    
)


def server(input, output, session):
    @output
    @render.plot
    def normal_distribution1():
        fig, ax = plt.subplots()
        p = input.normal1()
        mu = input.normal2()
        sigma = input.normal3()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(-5, 5, 100)
        y = np.zeros(100)
        for i in range(0, 100):
            y[i] = 1 / (sigma * (2 * math.pi) ** 1/2) * math.exp(-(x[i] - mu) ** 2)
        ax.plot(x, y)
        
        return fig
    
    @output
    @render.plot
    def normal_distribution2():
        fig, ax = plt.subplots()
        p = input.normal1()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(0, 10, 100)
        y = 1 - (1 - p) ** (x + 1)
        ax.plot(x, y)
        
        return fig

    

app = App(app_ui, server)
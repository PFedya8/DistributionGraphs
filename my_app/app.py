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
            ui.h5("\( \mathbb{F}(x) = \\frac{1}{2}[1+erf(\\frac{x-\mu}{\sqrt{2\sigma^2}})] \)"),
            ui.h5("\( erf \ - \ функция \ ошибок\)"),
            ui.h5("\( erf x\ = \ \\frac{2}{\sqrt{\pi}} \int\limits_0^x e^{-t^2} \mathrm d t\)"),
            ui.output_plot("normal_distribution2"),

            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \mu \)"),
                ui.tags.li("\(Дисперсия: \mu \)"),
                ui.tags.li("\(Мода: \mu\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: 0 \)"),
                ui.tags.li("\(Коэфициент \ эксцесса: 0 \)"),
            ),
        ),
    ),
    
)

def rectangle_method(f, a, b, N):
    ans = 0
    h = (b - a) / N
    mid = (2 * a + h) / 2
    for i in range(N):
        ans += f(mid)
        mid += h
    
    ans *= h
    return ans

def f(t):
    return math.exp(-(t ** 2))

def server(input, output, session):
    @output
    @render.plot
    def normal_distribution1():
        fig, ax = plt.subplots()
        mu = input.normal2()
        sigma = input.normal3()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(-5, 5, 100)
        y = np.zeros(100)
        for i in range(0, 100):
            y[i] = 1 / (sigma * (2 * math.pi) ** (1/2)) * math.exp(-(x[i] - mu) ** 2)
        ax.plot(x, y)
        
        return fig
    
    @output
    @render.plot
    def normal_distribution2():
        fig, ax = plt.subplots()
        mu = input.normal2()
        sigma = input.normal3()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(-5, 5, 100)
        x_i = np.zeros(100)
        for i in range(0, 100):
            x_i[i] = (x[i] - mu) / (sigma * (2 ** (1/2)))
        y = np.zeros(100)
        for i in range(0, 100):
            y[i] = 1 / 2 * (1 + 2 / (math.pi ** (1/2)) * rectangle_method(f, 0, x_i[i], 100))
        ax.plot(x, y)
        return fig

    

app = App(app_ui, server)
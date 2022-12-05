from shiny import *
import numpy as np


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
            "Отрицательное биномиальное распределение",
            ui.input_slider("slide1", "p", 0, 1, 0.5),
            ui.input_slider("slide2", "r", 0, 10, 5),

            ui.h4("Параметры:"),
            ui.h5("\( NB(r, p) \ - \ количество \ неудач \ до \ r-го \ успеха \)"),
            ui.h5("\( p - вероятность \ упеха \)"),

            ui.h3("Функция вероятности:"),
            ui.h5("\( \mathbb{P}(\\xi = k) = C^{k}_{k+r-1} p^r q^k,\ \ k \in \{0, 1, 2, ...\}\)"),
            ui.output_plot("prob_neg_bin"),

            ui.h3("Функция распределения"),
            ui.h5("\( F(k; r, p) = I_p(k + 1, r) \)"),

            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\frac{rq}{p} \)"),
                ui.tags.li("\(Дисперсия: \\frac{rq}{p^2} \)"),
                ui.tags.li("\(Мода: [\\frac{(r-1)q}{p}] \ если \ r > 1, \ 0 \ если \ r \leq \ 1\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\frac{2-p}{\\sqrt{rq}} \)"),
                ui.tags.li("\(Коэфициент \ эксцесса: \\frac{6}{r} + \\frac{p^2}{rq} \)"),
            ),

            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Плотность: P(\{k\}) = (1 - p)^{k-1}p \)"),
            ),
        ),
    ),
    
)

def fac(n):
    factorial = 1
    i = 1
    while (i <= n):
        factorial *= i
        i += 1
    return factorial


def combinations(n, k):
    return fac(n) / (fac(k) * fac(n - k))


def server(input, output, session):
    @output
    @render.plot
    def prob_neg_bin():
        fig, ax = plt.subplots()
        p = input.slide1()
        r = input.slide2()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = [0] * 10
        y = [0] * 10
        for i in range (0 , 10):
            x[i] = i
        x_1 = 0
        factorial = 1
        for i in range (1, 10):
            x_1 = combinations(x[i] + r - 1, x[i])
            y[i] = x_1 * (p ** r) * ((1 - p) ** x[i])
            ax.plot([x[i], x[i]], [0, y[i]], 'blue')
        ax.plot(x, y, 'o')
        ax.plot(x, y, 'blue')
        
        return fig
    
    

    

app = App(app_ui, server)
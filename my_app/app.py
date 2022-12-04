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
            "Геометрическое распределение",
            ui.input_slider("slide1", "p", 0, 1, 0.5),

            ui.h4("Параметры:"),
            ui.h5("\(n \geq 0 — число \ неудач \ до \ первого \ успеха\)"),
            ui.h5("\( p - вероятность \ упеха \)"),

            ui.h3("Функция вероятности:"),
            ui.h5("\( \mathbb{P}(\\xi) = q^n p, \ \ n \in \{0, 1, 2, ...\}\)"),
            ui.output_plot("probability3"),

            ui.h3("Функция распределения"),
            ui.h5("\( \mathbb{F}(\\xi) = 1 - q^{n+1}\)"),
            ui.output_plot("probability4"),

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
    def probability3():
        fig, ax = plt.subplots()
        p = input.slide1()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(0, 10, 100)
        y = (1 - p) ** x * p
        ax.plot(x, y)
        
        return fig
    
    @output
    @render.plot
    def probability4():
        fig, ax = plt.subplots()
        p = input.slide1()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(0, 10, 100)
        y = 1 - (1 - p) ** (x + 1)
        ax.plot(x, y)
        
        return fig

    

app = App(app_ui, server)
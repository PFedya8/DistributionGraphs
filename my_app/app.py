from shiny import *
import tabulate
import numpy as np


import pandas as pd
from matplotlib import pyplot as plt


app_ui = ui.page_fluid(
    # First page
    # For latex
    ui.head_content(
        ui.tags.script(
            src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        ),
        ui.tags.script(
            "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
        ),
    ),
    # Title
    
    
    ui.navset_tab(
        
        ui.nav(
            "Распределение Бернулли",
            ui.input_slider("slide1", "p", 0, 1, 0.5),
            ui.h2("Функция вероятности"),
            ui.output_plot("probability1"),
            ui.h2("Функция распределения"),
            ui.output_plot("distribution_function"),
            ui.output_table("result"),
            
                        
            ui.h2("Характеристики:"),
            ui.p("\(Математическое \ ожиданние: p\)"),
            ui.p("\(Дисперсия: p(1-p)\)"),
            ui.p("\(Мода: отсутствует\)"),
            ui.p("\(Коэфициент \ ассиметрии: отсутствует\)"),
            ui.p("\(Коэффциент \ эксцесса: отсутствует\)"),
            
            
        
            
            ui.p(
                """
                \( Математическое \ ожиданние: p \newline\
                nlkh
                \)
                """
            )
        ),
        ui.nav(
            "Биноминальное распределение",
            ui.input_slider("slide2", "p", 0, 1, 0.8),
            ui.output_plot("probability2"),
        ),
    ),
    
)



def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"slide1*2 is {input.slide1() * 2}"

    @output
    @render.plot
    def probability1(): #first plot
        fig, ax = plt.subplots()
        a = input.slide1()
        
        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        ax.plot([0, 0], [0, 1 - a], color="red")
        ax.plot([1, 1], [0, a])
        return fig

    @output
    @render.plot
    def distribution_function():
        fig, ax = plt.subplots()
        a = input.slide1()
        
        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        ax.plot([-1, 0], [0, 0], color="red")
        ax.plot([0, 1], [1 - a, 1 - a])
        ax.plot([1, 2], [1, 1])
        return fig


    @output
    @render.plot
    def probability2():
        fig, ax = plt.subplots()
        b = input.slide2()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        ax.plot([1, 0], [0, 1 - b], color="red")
        
        return fig
    



   

    

app = App(app_ui, server)
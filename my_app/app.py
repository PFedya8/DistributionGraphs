from shiny import *
import numpy as np

import scipy.stats as stats
from scipy.stats import hypergeom
from scipy.stats import poisson
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
            
            "Распределение Бернулли",
            
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

         
        ui.nav(
            "Биноминальное распределение",
            ui.input_slider("slide2", "p", 0, 1, 0.8),
            # input n as integer
            ui.input_numeric("n", "n", value= 10),
            ui.h3("Функция вероятности"),
            ui.output_plot("prob_binom"),
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: np\)"),
                ui.tags.li("\(Дисперсия: npq\)"),
                ui.tags.li("\(Медиана: {[np] -1, [np], [np] + 1}\)"),
                ui.tags.li("\(Мода:\ [(n+1)p]\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\frac{q-p}{\\sqrt{npq}}\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: \\frac{1-6pq}{npq}\)"),
                
            ),
            
            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(I_{1-p}(n - [k], 1 + [k])\)"),
            ),
            
        ),
        
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
    
        ui.nav(
            "Гипергеометрическое распределение",
            ui.input_numeric("population_size", "Population size", value= 20),
            ui.input_numeric("desired_items", "Total number of desired items", value= 7),
            ui.input_numeric("sample_size", "Number_of_draws", value= 12),
            ui.h3("Функция вероятности"),
            ui.output_plot("prob_hyper"),
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\frac{nD}{N}\)"),
                ui.tags.li("\(Дисперсия: \\frac{n(D/N)(1 - D/N)(N-n)}{(N-1)}\)"),
                ui.tags.li("\(Медиана: отсутствукт\)"),
                ui.tags.li("\(Мода:\ [\\frac{(D+1)(n + 1)}{N+2}]\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\frac{(N-2D)(N-1)^{0.5}(N - 2n)}{[nD(N-D)(N-n)]^{0.5}(N-2)}\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: [\\frac{N^2(N-1)}{n(N-2)(N-3)(N-n)}] \\times [\\frac{N(N+1)-6N(N-n)}{D(N-D)} + \\frac{3n(N-n)(N+6)}{N^2} -6]\)"),
            ),
        ), 
       
        
        
        ui.nav(
            "Распределение Пуассона",
            ui.input_numeric("lambd", "lambda", value= 10),
            ui.input_slider("sizer_pois", "k", 1, 500, 100),
            ui.output_plot("prob_pois"),
        ),
        
        
    ),
    
)



def server(input, output, session):
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
    
    # plot for binomial distribution
    @output
    @render.plot
    def prob_binom():
        fig = plt.subplots()
        p = input.slide2()
        n = input.n()
        binomial = data = stats.binom.rvs(n=n, p=p, size=1000)
        plt.hist(binomial, bins= n, density=False)
        plt.show()
        
        return fig


    @output
    @render.plot
    def prob_hyper():
        fig = plt.figure()
        n = input.population_size()
        a = input.desired_items()
        k = input.sample_size()
     
        x = np.arange(0, n + 1)
        y = stats.hypergeom.pmf(x, n, a, k)
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'bo')
        ax.vlines(x, 0, y, lw=2)
        ax.set_xlabel('Number of desired items in the sample')        
        plt.show()
        
    
    
    
    @output
    @render.plot
    def prob_pois():
        # poisson distribution
        fig = plt.figure()
        lam =  input.lambd()
        size = input.sizer_pois()
        x = np.arange(0, size)
        y = stats.poisson.pmf(x, lam)
        
        plt.plot(x, y, 'bo')
        for i in range(len(x)):
            plt.plot(x, y)
        plt.vlines(x, 0, y, lw=2)
        
        
        
        plt.show()

    

app = App(app_ui, server)

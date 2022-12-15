from shiny import *
import numpy as np
import seaborn as sb


import scipy.stats as stats
from scipy.stats import binom
from scipy.stats import hypergeom
from scipy.stats import poisson
import pandas as pd
import math as m
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
        # Распределение Бернулли
        ui.nav(
            "Распределение Бернулли",
            
            ui.input_slider("slide1", "p", 0, 1, 0.5),
            ui.h3("Функция вероятности"),
            ui.output_plot("probability1"),
            
            ui.h3("Функция распределения"),
            ui.output_plot("distribution_function"),
            ui.output_table("result"),
            
                        
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: p\)"),
                ui.tags.li("\(Дисперсия: p(1-p)\)"),
                ui.tags.li("\(Мода: отсутствует\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: отсутствует\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: отсутствует\)"),
                
            ),
            
            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(P(\{0\}) = 1 - p, \ P(\{1\}) = p\)"),
            ),
            
        ),
        # Биноминальное распределение
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
        # Геометрическое распределение
        ui.nav(
            "Геометрическое распределение",
            ui.input_slider("slide_geom", "p", 0, 1, 0.5),

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
        # Гипергеометрическое распределение
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
       # Распределение Пуассона
        ui.nav(
            "Распределение Пуассона",
            ui.input_numeric("lambd", "lambda", value= 10),
            ui.input_slider("sizer_pois", "k", 1, 500, 100),
            ui.h3("Функция вероятности"),
            ui.output_plot("prob_pois"),
            ui.h3("Функция распределения"),
            ui.output_plot("dist_pois"),
            
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\lambda\)"),
                ui.tags.li("\(Дисперсия: \\lambda\)"),
                ui.tags.li("\(Мода: [\\lambda] \)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: \\lambda^{-0,5}\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: \\lambda^{-1}\)"),
                
            ),
            
            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Плотность \ вероятности: f(x) = \\frac{\\lambda^k}{k!}e^{-\\lambda}\)"),
                ui.tags.li("\(Функция \ распределения: F(x) = \\frac{Г(k+1, \\lambda)}{k!} \)"),
            ),
            
            
        ),
        # Равномерное непрерывное распределение
        ui.nav(
            "Равномерное непрерывное распределение",
            
            ui.input_slider("slide_uniform1", "a", 0, 100, 20),
            # next slider have to start from the value of the previous slider
            ui.input_slider("slide_uniform2", "b", 0, 100, 80),
            
            
            ui.h3("Функция распределения"),
            ui.output_plot("dist_uniform"),
            
            ui.h3("Функция плотности"),
            ui.output_plot("prob_uniform"),
            
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\frac{a+b}{2}\)"),
                ui.tags.li("\(Дисперсия: \\frac{(b-a)^2}{12}\)"),
                ui.tags.li("\(Мода: \\forall число из [a,b] \)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: 0\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: -\\frac{6}{5} \)"),
            ),
            
            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Плотность \ вероятности: f(x) = \\frac{1}{b-a} I \{x \in [a,b]\}\)"),
                ui.tags.li("\(Функция \ распределения: F(x) = \\frac{x-a}{b-a} I \{x \in [a,b]\} + I\{x \geq b\}\)"),
            ),
        ),
        # Потенциальное распределение
        ui.nav(
            "Показательное распределение",
            ui.input_slider("slide_pot1", "\(\\lambda\)", min= 0, max= 30, value= 1.0, step= 0.5),
            ui.input_slider("size_pot", "Graph size", min= 1, max= 100, value= 10, step= 1),
            ui.h3("Функция распределения"),
            ui.output_plot("dist_pot"),
            ui.h3("Функция плотности"),
            ui.input_slider("size_pot2", "Graph size", min= 1, max= 100, value= 10, step= 1),
            
            ui.output_plot("prob_pot"),
            
            ui.h3("Характеристики:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Математическое \ ожиданние: \\lambda^{-1}\)"),
                ui.tags.li("\(Дисперсия: \\lambda^{-2}\)"),
                ui.tags.li("\(Мода: 0\)"),
                ui.tags.li("\(Коэфициент \ ассиметрии: 2\)"),
                ui.tags.li("\(Коэфициент \ эксцесса: 6 \)"),
            ),
            
            ui.h3("Формулы:"),
            ui.tags.ul(
                {"style":"list-style-type:circle;font-size: 20px"},
                ui.tags.li("\(Плотность \ вероятности: f(x) = \\lambda e^{-\\lambda x} I \{x \gt 0\}\)"),
                ui.tags.li("\(Функция \ распределения: F(x) = 1 - e^{-\\lambda x} \)"),
            ),
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
    def probability3():
        fig, ax = plt.subplots()
        p = input.slide_geom()

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
        p = input.slide_geom()

        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        x = np.linspace(0, 10, 100)
        y = 1 - (1 - p) ** (x + 1)
        ax.plot(x, y)
        
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

    # plot for binomial distribution
    @output
    @render.plot
    def prob_binom():
        p = input.slide2()
        n = input.n()
        '''
        binomial = data = stats.binom.rvs(n=n, p=p, size=1000)
        plt.hist(binomial, bins= n, density=False)
        plt.show()
        '''
        binom.rvs(size=10,n=n,p=p)

        data_binom = binom.rvs(n=n,p=p,loc=0,size=1000)
        ax = sb.distplot(data_binom,
                kde=True,
                color='blue',
                hist_kws={"linewidth": 25,'alpha':1})
        ax.set(xlabel='Binomial', ylabel='Frequency')
        


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
        
        
    
    
    @output
    @render.plot
    def dist_pois():
        # graph of poisson distribution function
        
        fig = plt.figure()
        lam =  input.lambd()
        size = input.sizer_pois()
        
        x = np.arange(0, size)
        y = stats.poisson.cdf(x, lam)
        
        # plt.plot(x, y, 'bo')
        for i in range(len(x)):
            plt.plot(x, y)
        
        plt.show()
        return fig
    
    @output
    @render.plot
    def dist_uniform():
        fig = plt.figure()
        a = input.slide_uniform1()
        b = input.slide_uniform2()
        
        if (b < a):
            # swap a and b
            a, b = b, a
        x = np.linspace(0, 100, 1000)
        # print(x)
        y = np.zeros(len(x))
        for i in range(len(x)):
            if (x[i] < a):
                plt.plot([x[i], a], [0, 0], color="red")
            elif (x[i] < a):
                plt.plot([x[i], x[i]], [0, 0], color="blue")
            elif (a <= x[i] < b):
                # y[i] = (x[i] - a) / (b - a)
                plt.plot([a, b], [0, 1])
            elif (x[i] > b): 
                plt.plot([x[i], 100], [1, 1], color="red")
        
        # plt.plot(x, y)
        # plt.show()
        
        
        return fig
    
    
    @output
    @render.plot
    def prob_uniform():
        fig = plt.figure()
        a = input.slide_uniform1()
        b = input.slide_uniform2()
        
        if (b < a):
            # we need to swap a and b
            a, b = b, a
            

        
        x = np.linspace(0, 100, 1000)
        # y = np.zeros(len(x))
        
        for i in range(len(x)):
            if (x[i] < a ):
                plt.plot([x[i], a] , [0, 0], color="red")
            elif (x[i] > b):
                plt.plot([b, x[i]], [0, 0], color="red")
            elif (a <= x[i] < b):
                plt.plot([a, b], [1 / (b - a), 1 / (b - a)], color="red")
            
            # lines -- from (a, 0) to (a, 1 / (b - a))
            plt.plot([a, a], [0, 1 / (b - a)], color="blue", linestyle="--")
            # lines -- from (b, 0) to (b, 1 / (b - a))
            plt.plot([b, b], [0, 1 / (b - a)], color="blue", linestyle="--")
        
        return fig
    
    @output
    @render.plot
    def dist_pot():
        fig = plt.figure()
        a = input.slide_pot1()
        sizer = input.size_pot()
        
        x = np.linspace(0, 100, 1000)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = 1 - m.exp(-a * x[i]) 
        
        plt.plot(x, y, color="red")
        plt.xlim(0, sizer)
        return fig
    

    @output
    @render.plot
    def prob_pot():
        fig = plt.figure()
        a = input.slide_pot1()
        sizer = input.size_pot2()
        
        x = np.linspace(0, 100, 1000)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = a * m.exp(-a * x[i])
        
        plt.plot(x, y, color="blue")
        plt.xlim(0, sizer)
        return fig
    
        
   
     

    

app = App(app_ui, server)
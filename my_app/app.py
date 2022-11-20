from shiny import App, render, ui
from matplotlib import pyplot as plt


app_ui = ui.page_fluid(
    # First page
    ui.navset_tab(
        
        ui.nav(
            "First page",
            ui.output_text("txt"),
            ui.h1("Бернулли"),
            ui.input_slider("slide1", "p", 0, 1, 0.5),
            ui.h2("Функция вероятности"),
            ui.output_plot("plot1"), 

        ),
        ui.nav(
            "Second page",
            ui.h1("Биноминальное распределение"),
            ui.input_slider("slide2", "p", 0, 1, 0.8),
            ui.output_plot("plot2"),
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
    def plot1(): #first plot
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
    def plot2():
        fig, ax = plt.subplots()
        b = input.slide2()
        print(b)
        plt.xlabel("x")
        plt.ylabel("p")
        plt.grid()
        ax.plot([1, 0], [0, 1 - b], color="red")
        
        return fig
    



   

    

app = App(app_ui, server)
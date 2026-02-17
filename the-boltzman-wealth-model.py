"""
Practice for Agent-Based Modeling using Mesa Lib.

Implented based on the lecture :
    https://mesa.readthedocs.io/latest/tutorials/0_first_model.html
    https://mesa.readthedocs.io/latest/tutorials/1_adding_space.html

Started : 2026-02-09
"""



# lib
import mesa
import solara
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import title

# Check Mesa version for visualization compatibility
if mesa.__version__.startswith(("3.0", "3.1", "3.2")):
    print(
        f"⚠️  Mesa {mesa.__version__} detected. Visualization features require Mesa 3.3+"
    )
    print("To upgrade: pip install --upgrade mesa")

from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component, make_space_component
from mesa.visualization.components import AgentPortrayalStyle


# Helper function
# compute Gini coefficient
def compute_gini(model):
    agent_wealth = [ agent.wealth for agent in model.agents]
    x = sorted(agent_wealth)
    n = model.num_agents

    B = sum(xi * (n-i) for i, xi in enumerate(x)) / (n* sum(x))
    return 1+ (1/n) - 2 * B


def data_visualize(model):
    # visualize the Gini coefficient
    gini = model.datacollector.get_model_vars_dataframe()
    g = sns.lineplot(data=gini)
    g.set(title="Gini Coefficient over Time", ylabel="Gini Coefficient")
    plt.show()

    # all_wealth = []
    #
    # for _ in range(1000):
    #     test_agent = MoneyModel(n=100)
    #
    #     for _ in range(300):
    #         test_agent.step()
    #
    #     for i in test_agent.agents:
    #         all_wealth.append(i.wealth)
    #
    #
    # g = sns.histplot(all_wealth, discrete=True)
    # g.set(
    #     title="Wealth distribution",
    #     xlabel="Wealth",
    #     ylabel="number of agents")
    #
    # # wealth의 최소값과 최대값 사이의 모든 정수를 틱으로 설정합니다.
    # g.set_xticks(range(min(all_wealth), max(all_wealth) + 1))
    # plt.show()



    # for _ in range(10000):
    #     model.step()
    #
    # agent_counts = np.zeros((model.grid.width, model.grid.height))
    # agent_wealth_sum = np.zeros((model.grid.width, model.grid.height))
    # sum=0
    # for cell in model.grid.all_cells:
    #     agent_counts[cell.coordinate] = len(cell.agents)
    #     agent_wealth_sum[cell.coordinate] = np.sum([agent.wealth for agent in cell.agents])
    #     print(np.sum(agent_wealth_sum))
    #
    #
    # g = sns.heatmap(agent_wealth_sum, cmap='viridis', cbar=False, square=True, annot=True)
    # g.figure.set_size_inches(5,5)
    # g.set(title="Number of agents on each cell of the grid")
    #
    # plt.show()

def data_agent_property(model):
    agent_wealth = model.datacollector.get_agent_vars_dataframe()
    # print(agent_wealth.head())


    # last_step = agent_wealth.index.get_level_values("Step").max()  # Get the last step
    # end_wealth = agent_wealth.xs(last_step, level="Step")[
    #     "Wealth"
    # ]  # Get the wealth of each agent at the last step
    # # Create a histogram of wealth at the last step
    # g = sns.histplot(end_wealth, discrete=True)
    # g.set(
    #     title="Distribution of wealth at the end of simulation",
    #     xlabel="Wealth",
    #     ylabel="number of agents",
    # )

    g = sns.histplot(agent_wealth["Wealth"], discrete=True)
    g.set(title="Wealth distribution", xlabel="Welath", ylabel="number of agents")
    plt.show()


def agent_portrayal(agent):
    return AgentPortrayalStyle(color="tab:orange", size=50)

# setting for solara

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "width": 10,
    "height": 10,
}



class MoneyAgent(CellAgent):
    """

    """

    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.wealth = 1

    def move(self):
        self.cell = self.cell.neighborhood.select_random_cell()

    def give_money(self):
        cellmates=[a for a in self.cell.agents if a is not self]

        if cellmates:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

        # if self.wealth>0 and cellmates:
        #     other_agent = self.random.choice(cellmates)
        #     other_agent.wealth += 1
        #     self.wealth -= 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

    # def exchange(self):
    #     if self.wealth > 0:
    #         other_agent = self.random.choice(self.model.agents)
    #         if other_agent.wealth is not None:
    #             self.wealth -= 1
    #             other_agent.wealth += 1



class MoneyModel(mesa.Model):
    """

    """

    def __init__(self, n=10, width=10, height=10, rng=None, seed=None ):
        super().__init__(seed=seed, rng=rng)
        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)

        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth"}
        )


        # Create agents
        agents = MoneyAgent.create_agents(
        self,
        self.num_agents,
        self.random.choices(self.grid.all_cells.cells, k=self.num_agents),)

    def run_for(self, n):
        for _ in range(n):
            self.step()

    def step(self) -> None:
        self.datacollector.collect(self)
        self.agents.do("step") # Calls the step method for all agents

money_model = MoneyModel(n=50, width=10, height=10)
#
# renderer = (
#     SpaceRenderer(model=money_model, backend="matplotlib")
#     .setup_agents(agent_portrayal)
#     .render()
# )

space_component = make_space_component(
    agent_portrayal=agent_portrayal,
    backend="matplotlib",   # 또는 "canvas" 등
)

GiniPlot = make_plot_component("Gini", page=1)

app = SolaraViz(
    MoneyModel,
    components=[space_component, GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)

#def main():

    #model = MoneyModel(100, 10,10, 10)
    # model.run_for(20)
    # data_visualize(model)
    # data_agent_property(model)



    # This is required to render the visualization in the Jupyter notebook
    # page
    # return page # Return the SolaraViz page object


#if __name__ == '__main__':
#    solara.display(app)
#    solara.run()

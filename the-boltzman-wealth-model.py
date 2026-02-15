"""
Practice for Agent-Based Modeling using Mesa Lib.

Implented based on the lecture :
    https://mesa.readthedocs.io/latest/tutorials/0_first_model.html
    https://mesa.readthedocs.io/latest/tutorials/1_adding_space.html

Started : 2026-02-09
"""



# lib
import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.pyplot import title
from networkx.algorithms.tree import to_prufer_sequence



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

        if self.wealth>0 and cellmates:
            other_agent = self.random.choice(cellmates)
            other_agent.wealth += 1
            self.wealth -= 1

    def exchange(self):
        if self.wealth > 0:
            other_agent = self.random.choice(self.model.agents)
            if other_agent.wealth is not None:
                self.wealth -= 1
                other_agent.wealth += 1



class MoneyModel(mesa.Model):
    """

    """

    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)

        agents = MoneyAgent.create_agents(
        self,
        self.num_agents,
        self.random.choices(self.grid.all_cells.cells, k=self.num_agents),)

    def step(self) -> None:
        self.agents.shuffle_do("move")
        self.agents.do("give_money")


def main():

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

    model = MoneyModel(1000, 10,10)

    for _ in range(10000):
        model.step()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    agent_wealth_sum = np.zeros((model.grid.width, model.grid.height))
    sum=0
    for cell in model.grid.all_cells:
        agent_counts[cell.coordinate] = len(cell.agents)
        agent_wealth_sum[cell.coordinate] = np.sum([agent.wealth for agent in cell.agents])
        print(np.sum(agent_wealth_sum))


    g = sns.heatmap(agent_wealth_sum, cmap='viridis', cbar=False, square=True, annot=True)
    g.figure.set_size_inches(5,5)
    g.set(title="Number of agents on each cell of the grid")

    plt.show()
if __name__ == '__main__':
    main()
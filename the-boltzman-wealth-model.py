"""
Pratice for Agent-Based Modeling using Mesa Lib.

Implented based on the lecture : https://mesa.readthedocs.io/latest/tutorials/0_first_model.html

Started : 2026-02-09

""" # 불필요한 임포트 제거

# lib

import mesa
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


class MoneyAgent(mesa.Agent):
    """

    """

    def __init__(self, model, unique_id): # unique_id 추가
        super().__init__(model)

        self.wealth = 1

    def say_wealth(self):
        print(f'my wealth is {self.wealth}')

    def say_hi(self):
        print(f'Hi! I am agent {self.unique_id}')

    def exchange(self): # 에이전트의 특정 행동 로직
        if self.wealth > 0:
            # 모델의 스케줄러에서 모든 에이전트 목록을 가져옵니다.
            all_agents = self.model.schedule.agents
            
            # 자기 자신을 제외한 다른 에이전트 목록을 만듭니다.
            other_agents = [agent for agent in all_agents if agent != self]

            if other_agents: # 다른 에이전트가 존재할 경우에만 교환을 시도합니다.
                other_agent = self.random.choice(other_agents)
                # MoneyAgent의 wealth는 항상 정수이므로 'is not None' 검사는 불필요합니다.
                self.wealth -= 1
                other_agent.wealth += 1

    def step(self): # 에이전트의 step 메서드 (스케줄러에 의해 호출됨)
        self.exchange()


class MoneyModel(mesa.Model):
    """

    """

    def __init__(self, n=10, seed=None):
        super().__init__(seed=seed)
        # 스케줄러를 초기화합니다. RandomActivation은 에이전트들을 무작위 순서로 활성화합니다.
        self.schedule = mesa.time.RandomActivation(self)
        self.num_agents = n

        # 반복문을 통해 MoneyAgent 인스턴스를 생성하고 스케줄러에 추가합니다.
        for i in range(self.num_agents):
            a = MoneyAgent(self, i) # 모델과 unique_id 전달
            self.schedule.add(a)

    def step(self) -> None:
        # 모델의 step 메서드는 스케줄러의 step 메서드를 호출합니다.
        # 스케줄러는 등록된 모든 에이전트의 step() 메서드를 호출합니다.
        self.schedule.step()


def main():

    all_wealth = []

    for _ in range(1000):
        test_agent = MoneyModel(n=100)

        for _ in range(300):
            test_agent.step()

        # 모델의 스케줄러를 통해 에이전트에 접근합니다.
        for i in test_agent.schedule.agents:
            all_wealth.append(i.wealth)

    g = sns.histplot(all_wealth, discrete=True)
    g.set(
        title="Wealth distribution",
        xlabel="Wealth",
        ylabel="number of agents")

    # wealth의 최소값과 최대값 사이의 모든 정수를 틱으로 설정합니다.
    g.set_xticks(range(min(all_wealth), max(all_wealth) + 1))
    plt.show()

if __name__ == '__main__':
    main()
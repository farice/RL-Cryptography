'''
Created on 2017/12/07

@author: Faris Sbahi
'''
from pybrainSG.rl.experiments.episodicSG import EpisodicExperimentSG
from pybrainSG.rl.agents.multiAgent import MultiAgent
from pybrainSG.rl.learners.valuebased.ceq import *
from pybrainSG.rl.agents.ceqa import *
from pybrainSG.rl.examples.tasks.cryptogame import CryptoGame, CryptoGameTask
import numpy as np
import warnings

def main():
    warnings.simplefilter("ignore")
    ma=MultiAgent()
    for i in range(CryptoGame.numberofAgents):
        learner= CEQ_Lin(
                        num_features=(CryptoGame.numberofAgents*2),
                        num_actions=np.ones(CryptoGame.numberofAgents,dtype=np.int8)*len(CryptoGame.availableActions),
                        num_agents=CryptoGame.numberofAgents,
                        indexOfAgent=i)
        agent= CEQ_Agent(
                            learner,
                            num_features=np.ones(CryptoGame.numberofAgents,dtype=np.int8)*(CryptoGame.numberofAgents*2),
                            num_actions=np.ones(CryptoGame.numberofAgents,dtype=np.int8)*len(CryptoGame.availableActions),
                            num_agents=CryptoGame.numberofAgents,
                            index=i)
        ma.addAgent(agent)
    task=CryptoGameTask(gameType="EG1")
    exp=EpisodicExperimentSG(task,ma)
    print("Rewards for agents at the end of episode:")
    for i in range(40000):
        rewards=exp.doEpisodes(number=1)
        print(str(rewards[0][-1]))

'''
Created on 2017/12/06

@author: Faris Sbahi
'''

from pybrain.rl.environments import Environment
from pybrainSG.rl.environments.episodicSG import EpisodicTaskSG
import numpy as np
import copy

class CryptoGameTask(EpisodicTaskSG):
    '''' '''
    maximumTurn=50
    def __init__(self,gameType="GG1"):
        '''
        gameType: indicates game type an experiment perform:
        [EG1:] Encryption Game
        '''
        EpisodicTaskSG.__init__(self, CryptoGame(gameType))

    def reset(self):
        EpisodicTaskSG.reset(self)
        self.isGameFinished=False
        self.currentTurn=0

    def isFinished(self):
        if self.currentTurn > self.maximumTurn:
            return True
        return self.env.isReachGoal

    def getReward(self):
        self.currentTurn+=1
#         print "rew:"+str(self.env.getJointReward())
        return self.env.getJointReward()

class CryptoGame(Environment):
    availableActions=[0,1,2,3,4,5,6,7] # permutation cycles
    sizeofGlidWorld=3
    numberofGoals=3
    numberofAgents=3

    def __init__(self,gameType="EG1"):
        Environment.__init__(self)
        assert (gameType == "EG1") or (gameType == "EG2") or (gameType == "EG3"), "gameType should be either EG1, EG2, or EG3."
        self.gameType=gameType
        print("Game type: " + str(self.gameType))

    def getSensors(self):
        #State: locations of all agents
        state=[np.r_[self.agents[0],self.agents[1]],
               np.r_[self.agents[0],self.agents[1]],
               np.r_[self.agents[0],self.agents[1]]]
        print ("state:" + str(state))
        return state

    def performAction(self, action):
        tempPos=[]
        self.prevAgents=copy.deepcopy(self.agents)
#         print "act:" + str(action)
        self.isColide=False
        for i in range(CryptoGame.numberofAgents):
            tempPos.append(self.__move__(copy.deepcopy(self.agents[i]), action[i]))
        if not self.__isColideWithEachOther(tempPos):
            self.agents=tempPos


    def __move__(self,position, forward):
        if forward == 0:#Move North
            position[1]+=1
        elif forward == 1:#Move west
            position[0]-=1
        elif forward == 2:#Move south
            position[1]-=1
        elif forward == 3:#Move east
            position[0]+=1
        else:
            assert False, "Unexpected action"

        if position[0] >= CryptoGame.sizeofGlidWorld:
            position[0]=CryptoGame.sizeofGlidWorld-1
        if position[0] < 0:
            position[0]=0
        if position[1] >= CryptoGame.sizeofGlidWorld:
            position[1]=CryptoGame.sizeofGlidWorld-1
        if position[1] < 0:
            position[1]=0
        return position

    def __isColideWithEachOther(self,tempPos):
        if (tempPos[0][0] == tempPos[1][0]) and (tempPos[0][1] == tempPos[1][1]):
            if (tempPos[0][0] != self.goals[0][0]) or (tempPos[0][1] != self.goals[0][1]):
                self.isColide=True
                return True
            else:
                return False
        else:
            return False

    def __isReachGoal(self):
        #return boolean list, that determine if each agent reach each goal.
        irGoal=[False,False]
        if (self.agents[0][0] == self.goals[0][0]) and (self.agents[0][1] == self.goals[0][1]):#For the first agent.
            irGoal[0]=True
            self.isReachGoal=True
        if (self.agents[1][0] == self.goals[1][0]) and (self.agents[1][1] == self.goals[1][1]):#For the second agent.
            irGoal[1]=True
            self.isReachGoal=True
        return irGoal

    def reset(self):
        self.agents=[np.array([0,0]),
                     np.array([2,0])]
        self.prevAgents=[np.array([0,0]),
                     np.array([2,0])]

        self.isReachGoal=False
        self.goals=[np.array([2,2]),
                    np.array([0,2])]
        self.goals=[np.array([1,2]),
                    np.array([1,2])]

    def getJointReward(self):
        jointRew=[0,0]
        irGoal=self.__isReachGoal()
        if irGoal[0]:
            jointRew[0]=100
        if irGoal[1]:
            jointRew[1]=100
        if self.isColide:
            jointRew[0]-=1
            jointRew[1]-=1

        return np.array(jointRew)

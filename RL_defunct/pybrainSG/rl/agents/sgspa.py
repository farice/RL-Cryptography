'''
Created on 2016/03/10

@author: takuya-hv2
'''
from pybrainSG.rl.agents.loggingSG import LoggingAgentSG
from pybrain.utilities import drawIndex
from pybrainSG.rl.valuebased.indexablevaluebased import IndexableValueBasedLearner
from scipy import array
import numpy as np

#Implmenting now
class SGSP_Agent(LoggingAgentSG):
    """ 
    Agent based on SPSG RL algorithms put on: 
    pybrainSG.rl.valuebased.spsg
    """
    init_exploration = 0.005   # aka epsilon
    exploration_decay = 0.9999 # per episode        
        
    # flags for exploration strategies
    epsilonGreedy = True
    learning = True
    
    def __init__(self, learner, num_actions, numAgents, index, **kwargs):
        assert isinstance(learner, IndexableValueBasedLearner), "learner should be indexable."
        self.learner = learner
        LoggingAgentSG.__init__(self, np.ones(numAgents)*learner.num_features, num_actions, numAgents, index, **kwargs)
        self.learner._behaviorPolicy = self._actionProbs
        self.reset()
        self.agentProperties["requireOtherAgentsState"]=False
        self.agentProperties["requireJointAction"]=True
        self.agentProperties["requireJointReward"]=True
        for prop in self.learner.getProperty().keys():
            if learner.getProperty()[prop]:
                assert self.getProperty()[prop], "learners property should same to that of agents."
        
    def _actionProbs(self, state):
        if not self.epsilonGreedy:
            return self.learner._softmaxPolicy(state)
        elif self.epsilonGreedy:
            return (self.learner._softmaxPolicy(state) * (1 - self._expl_proportion) 
                    + self._expl_proportion / float(self.learner.num_actions[self.indexOfAgent]))
    
    def getAction(self):
        self.lastaction = drawIndex(self._actionProbs(self.lastobs), True)
        if self.learning and not self.learner.batchMode and self._oaro is not None:
            self.learner._updateWeights(*(self._oaro + [self.lastaction]))
            self._oaro = None
#         print "Agent " + str(self.indexOfAgent) + ": " + str(self.lastaction)
        return array([self.lastaction])
        
    def integrateObservation(self, obs):
        if self.learning and not self.learner.batchMode and self.lastobs is not None:
            if self.learner.passNextAction:
                self._oaro = [self.lastobs, self.lastaction, self.lastreward, obs]
            else:
                self.learner._updateWeights(self.lastobs, self.lastaction, self.lastreward, obs)
        LoggingAgentSG.integrateObservation(self, obs)        
        
    def reset(self):
        LoggingAgentSG.reset(self)
        self._expl_proportion = self.init_exploration
        self.learner.reset()    
        self._oaro = None
        self.newEpisode()
        
    def newEpisode(self):
        if self.logging:
            for i in range(self.numAgents):
                self.history[i].newSequence()
        if self.learning and not self.learner.batchMode:
            self.learner.newEpisode()
        else:
            self._expl_proportion *= self.exploration_decay      
            self.learner.newEpisode()
                        
    def learn(self):
        if not self.learning:
            return
        if not self.learner.batchMode:
            print('Learning is done online, and already finished.')
            return
        for seq in self.history[self.indexOfAgent]:
            for obs, action, reward in seq:
                if self.laststate is not None:
                    self.learner._updateWeights(self.lastobs, self.lastaction, self.lastreward, obs)
                self.lastobs = obs
                self.lastaction = action[0]
                self.lastreward = reward
            self.learner.newEpisode()
    
    def setIndexOfAgent(self,index):
        """ indexing agent and its learner.
            :key index: index of agent
            :type index: integer
        """ 
        super(SGSP_Agent, self).setIndexOfAgent(index)
        self.learner.setIndexOfAgent(index)

import sys
import math
sys.path.append('/Users/Kikos/Desktop/helicopter')
sys.path.append('/Users/Kikos/Desktop/helicopter/site-packages')
from eonn import eonnIRL
from eonn.genome import Genome
from eonn.organism import Pool
from helicopter.helicopter import Helicopter, XcellTempest
from mdp import test
from mdp import train
import numpy
from environment import Environment
from functions import extract_sa 
from functions import Evaluator
from expert import Expert

class Apprentice():
    def __init__(self,env,theta):
        try:
         index = int(env)
         params = train[index]
         self.heli=Environment(params) # helicopter simulator for learning genomes
        except:
         print 'Usage: experiment [0-9]'
         exit()
        self.learnedGenome = 1 # Current genome attached to the apprentice assign genome when it first appears
        self.apprenticeEnvironment = env # the environment at which the apprentice will operate is deternined at instantiation
        if sum(numpy.absolute(theta))>0: # make sure initial reward vector is >0
            self.theta  = theta
        else:
            exit()
        
    def features_from_state(self,state):
      currentFeatures = [0,0,0,0]
      for i in range(4):
        currentFeatures[i] =sum(v**2 for v in state[i*3:i*3+3] ) #feature definitions -  (position,velocity and rotation features from states)
      return currentFeatures
    
    def error_from_observation(self,observation):
        """ Return the error of the current state. """
        if not self.heli.terminal:
            state,q,a = extract_sa(observation) # Function from evaluator
            state = state+q
            feat = self.features_from_state(state[:-1])
            error = numpy.dot(self.theta,feat)
        else:
            error = sum([v**2 for v in self.heli.LIMITS[:9]] + [1.0 - self.heli.LIMITS[9]**2])
            error *= (self.heli.max_steps - self.heli.steps)
        return error
    
    def percieved_eval(self,policy): #policy evaluation returns policy 1/log(error) i.e max is best
      """ Helicopter alternative evaluation function for IRL. """
      observation, sum_error = self.heli.reset()
      while not self.heli.terminal:
        action = policy.propagate(observation,1)
        observation, err = self.heli.update(action) 
        error = self.error_from_observation(observation)
        sum_error+=error
      return 1/math.log(sum_error)
 
    def learn_genomeIRL(self): #input the reward function
      pool = Pool.spawn(Genome.open('policies/generic.net'), 20)
      # Set evolutionary parameters
      eonnIRL.keep = 15 ; eonnIRL.mutate_prob = 0.4 ; eonnIRL.mutate_frac = 0.1;eonnIRL.mutate_std = 0.8;eonnIRL.mutate_repl = 0.15
      # Evolve population
      pool = eonnIRL.optimize(pool, self.percieved_eval,2) # These are imported functions from EONNIRL
      champion = max(pool)
      # Print results
      print '\nerror:', math.exp(1 / self.percieved_eval(champion.policy))
      #print '\ngenome:\n%s' % champion.genome
      return champion.policy
    
    def policy_observation(self,policy): #policy observation returns policy feature sum
      """ Helicopter alternative evaluation function for IRL. """
      observation, sum_error = self.heli.reset()
      featureSum = self.features_from_state(observation)
      while not self.heli.terminal:
        action = policy.propagate(observation,1)
        observation, err = self.heli.update(action) # Maybe I should return q here as well. so that we have the same info to start with.
        state,q,a = extract_sa(observation) # Function from evaluator
        state = state+q
        feat = self.features_from_state(state[:-1])
        featureSum=[sum(x) for x in zip(*[featureSum,feat])]
      return featureSum
       
       
    def observe_multiple(self,times,policy): #env is the policy to be executed by the expert (and the environment during the expert demonstration)
        featureExp = [0,0,0,0]
        for i in range(times):
            featureSum = self.policy_observation(policy)
            featureExp = feature_exp_step(featureExp,featureSum,i+1)
        return featureExp


def real_eval(heli,policy): #policy evaluation returns policy 1/log(error) i.e max is best
    """ Helicopter alternative evaluation function for IRL. """
    observation, sum_error = heli.reset()
    while not heli.terminal:
        action = policy.propagate(observation,1)
        observation, err = heli.update(action) 
        sum_error+=err
    return 1/math.log(sum_error)
 
def feature_exp_step(featureExp,currentFeatures,counter):
    #calculates running average from state features constructed using featuresFromState:
    for i in range(4):
        featureExp[i] = currentFeatures[i]/(counter+1) + featureExp[i]*counter/(counter+1)
    return featureExp

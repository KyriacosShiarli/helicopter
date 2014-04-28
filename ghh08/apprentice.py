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

class Apprentice():
    def __init__(self, env,steps=600):
      def init_env():
        """ Initialize helicopter environment. """
        try:
         index = int(env)
         params = test[index]
         Environment(params,steps)
        except:
         print 'Usage: experiment [0-9]'
        exit()
      heli = init_env()
      
      learnedGenome = 1
    
    def hover(policy):
      """ Helicopter evaluation function. """
      state, sum_error = heli.reset()
      while not heli.terminal:
        action = policy.propagate(state, 1)
        state, error = heli.update(action)
        sum_error += error
      return 1 / math.log(sum_error)
    
    def featuresFromState(state):
      currentFeatures = [0,0,0,0]
      for i in range(4):
        currentFeatures[i] =sum(v**2 for v in state[i*3:i*3+3] ) #feature definitions -  (position,velocity and rotation features from states)
      return currentFeatures
    
    
    def hoverIRL(policy,thet):
      """ Helicopter alternative evaluation function. """
      state, sum_error = heli.reset()
      while not heli.terminal:
        action = policy.propagate(state, 1)
        state, error = heli.update(action) # Maybe I should return q here as well. so that we have the same info to start with.
        st,q,a = extract_sa(state)
        st = st+q
        feat = featuresFromState(st[:-1])
        errorTest = numpy.dot(theta,feat)
        if errorTest==error:
          sum_error += errorTest
        else:
          sum_error+=error
        print"-----------"
        print errorTest
        print error
        print "-----------"
      return 1 / math.log(sum_error)
    
    
    
    def learnGenomeIRL(theta): #input the reward function
      pool = Pool.spawn(Genome.open('policies/generic.net'), 20)
      # Set evolutionary parameters
      eonnIRL.keep = 15
      eonnIRL.mutate_prob = 0.4
      eonnIRL.mutate_frac = 0.1
      eonnIRL.mutate_std = 0.8
      eonnIRL.mutate_repl = 0.15
      # Evolve population
      pool = eonnIRL.optimize(pool, hoverIRL,theta)
      champion = max(pool)
      # Print results
      print '\nerror:', math.exp(1 / hover(champion.policy))
      print '\nerror:', math.exp(1 / hoverIRL(champion.policy,theta))
      print '\ngenome:\n%s' % champion.genome
    
      

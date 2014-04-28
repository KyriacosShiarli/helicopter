import sys
import math
sys.path.append('/Users/Kikos/Desktop/helicopter/site-packages')
sys.path.append('/Users/Kikos/Desktop/helicopter/ghh08')
from mdp import train
from mdp import test
from expert import Expert
from environment import Environment


from eonn.genome import Genome
from eonn.organism import Organism


def init_env():
  """ Initialize helicopter environment. """
  try:
    index = int(sys.argv[1])
    params = train[index]
    return Environment(params)
  except:
    print 'Usage: experiment [0-9]'
    exit()
    
def featureExpStep(featureExp,currentFeatures,counter):
        #calculates running average from state features constructed using featuresFromState:
        for i in range(3):
          featureExp[i] = currentFeatures[i]/(counter+1) + featureExp[i]*counter/(counter+1)
          
def featuresFromState(state):
        currentFeatures = [0,0,0]
        for i in range(3):
          currentFeatures[i] = math.sqrt(state[i]**2 + state[i+1]**2 + state[i+2]**2) #feature definitions -  (position,velocity and rotation features from states)
        return currentFeatures
              
def observeExpert(): #Feature sum for one run of the optimal policy
  """ Main function, runs the experiment. """
  expert = Expert(int(sys.argv[1])) # initialise and expert with a certain policy form the pre-trained ones
  env = init_env() # initialise an environment
  featureSum=[0,0,0] # feature expectations is a 1x3 vector
  counter = 1 #counter to calculate average
  #runs of the policy
  expert.start() 
  state, reward = env.reset()
  while not env.terminal:
      action = expert.step(state, reward)
      state, reward = env.update(action)
      feat = featuresFromState(state)
      featureSum=[sum(x) for x in zip(*[featureSum,feat])]  
      counter+=1
  expert.end(reward)
  return featureSum    
      

def main():
  sums = getFeatureSum()
  print sums
  
      
if __name__ == '__main__':
  main()

    
    
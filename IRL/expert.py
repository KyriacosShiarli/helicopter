from eonn.genome import Genome
from eonn.organism import Organism


POLICIES = [Genome.open('policies/mdp%i.net' % i) for i in range(10)]



class Expert:
  """ Model-Free agent, chooses best policy from a set of pre-trained policies. """
  def __init__(self,policyNumber):
    """ Initialize expert. """
    self.episode = 0
    #self.org = [Genome.open('policies/mdp%i.net' %i)]
    self.pool = POLICIES[policyNumber]
    self.org = Organism(POLICIES[policyNumber])

  def start(self):
    """ Start a new episode """
    self.reward = 0
    self.steps = 0

  def step(self, state, reward):
    """ Choose an action based on the current state. """
    self.steps += 1
    self.reward += reward
    # DEBUG
    action = self.org.policy.propagate(state, 1)
    return action

  def end(self, reward):
    """ Ends the current episode """
    self.reward += reward
    self.org.evals.append(self.reward)
    print '%i %i %.2f' % (self.episode, self.steps, self.reward)   # ---- Prints episode steps and reward steps is usually 6000. 

      


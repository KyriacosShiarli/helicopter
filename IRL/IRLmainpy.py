import sys,os
# relative paths required
dir = os.path.dirname(__file__)
prev = os.path.dirname(dir)

two = prev+'/site-packages/'
sys.path.append(prev)
sys.path.append(dir)
sys.path.append(two)
from eonn.genome import Genome
from eonn.organism import Organism
from apprentice import *
import numpy as np
import math



POLICIES = [Genome.open('policies/mdp%i.net' % i) for i in range(10)]
pol = Organism(POLICIES[2])


def normalise_vector(vector):
  return np.divide(vector,np.linalg.norm(vector))

def modify_mu(mu_apprentice,mu_modified,mu_expert):
  factor = np.dot(mu_apprentice[-1]-mu_modified[-1],mu_expert-mu_apprentice[-1])/np.dot(mu_apprentice[-1]-mu_modified[-1],mu_apprentice[-1]-mu_modified[-1])
  mu_new = mu_modified[-1] + factor * (mu_apprentice[-1]-mu_modified[-1])
  mu_modified.append(mu_new)
  
def threshold(theta):
  for i in range(len(theta)):
    if theta[i]<0.:
      theta[i]=0 

def error_print(policy): # not done yet just keeping things tidy
  print_dyn( '\n percieved error for policy %s \n' % math.exp(1/student.percieved_eval(policy)))
  print_dyn( '\n real error for policy %s \n' % math.exp(1/real_eval(student.heli,policy)))

def print_dyn(what): # dynamic printing function
  sys.stdout.write(what)
  sys.stdout.flush()
  
def print_progress(i):
  print_dyn( "--------------------------------------------------------------\n")
  print_dyn( "ITERATION: %s \n" % i)
  print_dyn( "errors for expert:\n")
  error_print(pol.policy)
  print "errors for apprentice:"
  error_print(trainedPolicy)
  print_dyn(str(['apprentice FE %s'% mu_app +'and expert FE %s \n' % mu_expert]))
  print_dyn("delta theta %s \n" %delta_theta)
  print_dyn( "old theta %s\n" %old)
  print_dyn( "new theta %s\n" %student.theta)
  print_dyn( "--------------------------------------------------------------\n")

fac=1
#start with an initial theta to give a first policy
thetaInitial = [10,0,0.5,1] 
# initialise agent with that
student = Apprentice(sys.argv[1],thetaInitial)
#get feature expectation from expert in that environment
mu_expert = student.observe_multiple(5,pol.policy)
# learn best policy given that reward function
for i in range(10):
  trainedPolicy = student.learn_genomeIRL()
  mu_app = student.policy_observation(trainedPolicy) # all feature sums normalised
  
  delta_theta = normalise_vector(np.subtract(mu_app,mu_expert)) #other way round because we talk of errors

  old = student.theta
  student.theta = np.add(student.theta,delta_theta)
  threshold(student.theta)
  print_progress(i)
# get the feature expectation using that trained optimal policy
#mu_app.append(normalise_vector(student.policy_observation(trainedPolicy)))
'''
mu_mod.append(mu_app[0])
#calculate your first theta
student.theta = abs(mu_expert - mu_app[0])
print 'elloooo',student.theta
tee = np.linalg.norm(student.theta)

for i in range(20):
  trainedPolicy=student.learn_genomeIRL()
  mu_app.append(normalise_vector(student.policy_observation(trainedPolicy)))
  modify_mu(mu_app,mu_mod,mu_expert)
  student.theta = abs(mu_expert-mu_app[0])
  print 'elloooo',student.theta
  tee = np.linalg.norm(student.theta)
'''
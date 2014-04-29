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
  
def theta_descent():
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
    print_dyn( "new theta %s\n" %new)
    print_dyn( "--------------------------------------------------------------\n")
  fac=1
  #get feature expectation from expert in that environment
  mu_expert = normalise_vector(student.observe_multiple(1,pol.policy))
  # learn best policy given that reward function
  for i in range(40):
    trainedPolicy = student.learn_genomeIRL()
    mu_app = normalise_vector(student.policy_observation(trainedPolicy))
    delta_theta = np.subtract(mu_app,mu_expert)
    old = student.theta
    new =normalise_vector(np.add(student.theta,delta_theta))
    threshold(new)
    print_progress(i)
    student.theta=new

def projection():

  mu_app = []
  mu_mod = []
  # initialise agent with that
  
  #get feature expectation from expert in that environment
  mu_expert = normalise_vector(student.observe_multiple(5,pol.policy))
  # learn best policy given that reward function
  for i in range(20):
    trainedPolicy = student.learn_genomeIRL()
    mu_app.append(normalise_vector(student.policy_observation(trainedPolicy)))
    if i ==0:
      mu_mod.append(mu_app[0])
      old = student.theta
      student.theta = abs(np.subtract(mu_mod[i],mu_expert))
    else:  
      modify_mu(mu_app,mu_mod,mu_expert)
      old = student.theta
      student.theta = normalise_vector(abs(np.subtract(mu_mod[i],mu_expert)))
    #threshold(student.theta)
    print_dyn( "--------------------------------------------------------------\n")
    print_dyn( "ITERATION: %s \n" % i)
    print_dyn( "errors for expert:\n")
    error_print(pol.policy)
    print "errors for apprentice:"
    error_print(trainedPolicy)
    print_dyn(str(['apprentice FE %s'% mu_app[i] +'and expert FE %s \n' % mu_expert]))
    print_dyn( "old theta %s\n" %old)
    print_dyn( "new theta %s\n" %student.theta)
    print_dyn( "--------------------------------------------------------------\n")

thetaInitial = [2,0.5,2,0.5]
student = Apprentice(sys.argv[1],thetaInitial)
error = student.percieved_eval(pol.policy)
theta_descent()
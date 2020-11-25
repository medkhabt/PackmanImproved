import random, copy
from datetime import datetime 
from abc import abstractmethod
from pickle import FALSE
from pyexpat import model
#from Tools.demo import vector
#from _hashlib import new 


#===============================================================================

def update( xyEnvironment, objects, agents,   width , height ):
    xyEnvironment.objects  = objects
    xyEnvironment.agents=agents
    xyEnvironment.width=width
    xyEnvironment.height=height
    

def distance2(location1 , location2):  
    return (location1[0]-location2[0])**2 + (location1[1]-location2[1])**2
    
def vector_add(head, location):
    return (head[0]+location[0], head[1]+location[1])
 
#==============================================================================
   

PRINT_STEP = False
STEP_PER_STEP=False

class Object:
  
    def __repr__(self): #Equivalent a toString
        return '<%s>' % getattr(self, '__name__', self.__class__.__name__)

    def is_alive(self):
        """Objects that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def display(self, canvas, x, y, width, height):
        """Display an image of this Object on the canvas."""
        pass
#===============================================================================

class Agent(Object):

    def __init__(self):
        def program(percept):
            return input('Percept=%s; action? ' % percept)
        self.program = program
        self.alive = True
#===============================================================================


def TraceAgent(agent):
    old_program = agent.program
    def new_program(percept):
        action = old_program(percept)
        if PRINT_STEP: print ('%s perceives %s and does %s' % (agent, percept, action))
        return action
    agent.program = new_program
    return agent


#===============================================================================

class RandomAgent(Agent):
    def __init__(self, actions):
        Agent.__init__(self)
        self.bump = False
        self.heading = headings[0]
        random.seed(datetime.now()) # random.seed(), fix l'enchainement du random numbers. 
        #here we have a variant argument, which means we have different random numbers each
        #time. 
        
        #Overriding the program of the Agent Class, here it seems like an 
        #attribute but still it is a function so technically it's a method. 
        self.program = lambda percept: random.choice(actions)


#===============================================================================


class Environment:
   
    def __init__(self,):
        self.objects = []; 
        self.agents = []

    object_classes = [] ## List of classes that can go into environment

    @abstractmethod
    def percept(self, agent):
        "Return the percept that the agent sees at this point. Override this." 
        pass

    @abstractmethod
    def execute_action(self, agent, action):
        "Change the world to reflect this action. Override this."
        pass

    def default_location(self, thing):
        "Default location to place a new object with unspecified location." 
        return None

    def exogenous_change(self):
        "If there is spontaneous change in the world, override this."
        pass

    def is_done(self):
        "By default, we're done when we can't find a live agent."
        for agent in self.agents:
            if agent.is_alive(): return False
        return True

    def step(self): 
      
        if not self.is_done():
            actions = [agent.program(self.percept(agent))
                       for agent in self.agents]
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
                self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            if self.is_done(): return
            self.step()

    def add_object(self, object, location=None):
        object.location = location or self.default_location(object)
        self.objects.append(object)
        if isinstance(object, Agent):
            object.performance = 0
            self.agents.append(object)
            return self
    
#===============================================================================

class Wall(Object): pass

#===============================================================================


class XYEnvironment(Environment):

    def __init__(self, width=10, height=10):
        update(self, objects=[], agents=[], width=width+2, height=height+2)
        
    
    def objects_at(self, location):
        "Return all objects exactly at a given location."
        return [obj for obj in self.objects if obj.location == location]

    def objects_near(self, location, radius):
        "Return all objects within radius of location."
        radius2 = radius * radius
        return [obj for obj in self.objects
                if distance2(location, obj.location) <= radius2]

    def percept(self, agent):
        "By default, agent perceives objects within radius r."
        return [self.object_percept(obj, agent)
                for obj in self.objects_near(agent)]

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'TurnRight':
            agent.heading = turn_heading(agent.heading, -1)
        elif action == 'TurnLeft':
            agent.heading = turn_heading(agent.heading, +1)
        elif action == 'Forward':
            self.move_to(agent, vector_add(agent.heading, agent.location))
        elif action == 'Grab':
            objs = [obj for obj in self.objects_at(agent.location)
                    if obj.is_grabable(agent)]
            if objs:
                agent.holding.append(objs[0])
        elif action == 'Release':
            if agent.holding:
                agent.holding.pop()
                
        if agent.bump:
            agent.performance -= 100
        

    def object_percept(self, obj, agent): #??? Should go to object?
        "Return the percept for this object."
        return obj.__class__.__name__

    def default_location(self, thing):
#         random.seed(datetime.now()) 
        return (random.choice(range(self.width-2))+1, random.choice(range(self.height-2))+1)

    @abstractmethod 
    def move_to(self, object, destination):
        "Move an object to a new location."
        pass
    
    def add_object(self, object, location= None):
        Environment.add_object(self, object, location)
        object.holding = []
        object.held = None
 
    def add_walls(self):
        "Put walls around the entire perimeter of the grid."
        for x in range(self.width):
            self.add_object(Wall(), (x, 0))
            self.add_object(Wall(), (x, self.height-1 ))
        for y in range(self.height):
            self.add_object(Wall(), (0, y))
            self.add_object(Wall(), (self.width-1 , y))

#--------------------------------------------------------------------------------
headings=[(1, 0), (0, 1), (-1, 0), (0, -1)]

def turn_heading(heading, inc):
    "Return the heading to the left (inc=+1) or right (inc=-1) in headings."
    return headings[(headings.index(heading) + inc) % len(headings)]  
 
#--------------------------------------------------------------------------------

class Dirt(Object): pass

#===============================================================================

 
def RandomVacuumAgent():
    "Randomly choose one of the actions from the vaccum environment."
    return RandomAgent(['TurnRight', 'TurnLeft', 'Forward', 'Suck' ])

#===============================================================================
 
class ReflexVacuumAgent(Agent):
    "A reflex agent for the  Vacuum Cleaner Robot"

    def __init__(self): 
        Agent.__init__(self)
        self.bump=False
        self.heading=headings[0]
        #**************************************
        def program( percept ):
            status = percept[0]
            bump= percept[1]
            if status == 'Dirty' :
                return 'Suck'
            else:
                return random.choice(['TurnRight', 'TurnLeft', 'Forward', 'Suck' ])
        self.program = program




class ModelBasedVacuumAgent(Agent): 
    """ your code here """  
    def __init__(self): 
        Agent.__init__(self)
        self.bump=False
        self.heading=headings[0]
        self.model = []
        #******************************8
        def program( percept  ):
            location = percept[0] 
            status = percept[1]
            if status == 'Bump':
                obstacleLocation = vector_add(self.heading, location)
                model.append(obstacleLocation)
            if location == 'Dirty':
                return 'Suck'
            while(True):
                action = random.choice(['TurnRight', 'TurnLeft', 'Forward'])
                if action == 'Forward':
                    newLocation = vector_add(self.heading, location)
            if not newLocation in self.model:
                return action
            else : return action
        self.program = program

class VacuumEnvironment(XYEnvironment):
   
    def __init__(self, width=10, height=10):
        XYEnvironment.__init__(self, width, height)
        self.add_walls() 
        
    object_classes = [Wall, Dirt, RandomVacuumAgent, 
                      ReflexVacuumAgent, ModelBasedVacuumAgent] 

    def percept(self, agent):
        status = 'Clean'
        if (self.find_at(Dirt, agent.location)):
            status= 'Dirty'
        bump = 'None'   
        if(agent.bump) :
            bump = 'Bump'
        return (status, bump)

    def execute_action(self, agent, action):
        if action == 'Suck':
            if self.find_at(Dirt , agent.location):
                self.clean_location(agent.location)
                agent.performance += 1000
        else : 
            agent.performance -= 1
            XYEnvironment.execute_action(self, agent, action)
        
    def step(self):
        XYEnvironment.step(self)
        if PRINT_STEP: printEnv(self)
        if STEP_PER_STEP: 
            input("Tape a key to continue ...")

    def clean_location(self, location): 
        for d in  self.objects_at(location):
            if isinstance(d, Dirt):
                self.objects.remove(d)
                d.location = None
                

    def find_at(self, clasz, location):
        for o in XYEnvironment.objects_at(self, location):
            if isinstance(o, clasz):
                return True
            
        return False       
        
    def move_to(self, thing, destination):
        "Move an object to a new location."
        l=XYEnvironment.objects_at(self, destination)
        if isinstance(thing, Agent):
            if  repr(Wall()) in [ repr(e) for e in l]: 
                thing.bump=True
            else: thing.location = destination
        else  : thing.location = destination
#===============================================================================

def printEnv(xy_env): 
    print ("--------" * xy_env.width )
    for row in range(xy_env.height):
        for r in range(3):
            print ("|" +  cells(xy_env, row, r,  0) )
        print ("--------" * xy_env.width )
    
def  cells(xy_env, row, r, col ) :
    if col == xy_env.width : return ""
    segment = "       "
    for o in xy_env.objects:
        if o.location == (row, col):
            segment = get_view(o, r)
            break 
    for o in xy_env.agents:
        if o.location == (row, col):
            segment = get_view(o, r)
            break 
    return segment + "|" +  cells(xy_env, row, r, col+1);
     



def get_view (thing , i):
    views = [
        ["       ", "   *   ", "       "],#Dirty (Clean : "       ", "       ", "       ")
        ["*******", "*******", "*******"],#wall 
        ["   ~   ", " ~ A ~ ", "   ~   "],#Collision
        ["       ", "   A   ", "   v   "],#down 
        ["       ", "   A > ", "       "],#rigth
        ["   ^   ", "   A   ", "       "],#up               enum Oriontation {UP, RIGTH, DOWN, LEFT, NONE] 
        ["       ", " < A   ", "       "],#left
        ["   ^   ", " < A > ", "   v   "]#Error
    ]
    if isinstance(thing, Dirt):
        return views[0][i]
    if isinstance(thing, Wall):
        return views[1][i]
    if isinstance(thing, Agent): 
        if getattr(thing, 'bump')  :
            return views[2][i]
        return views[headings.index(thing.heading)+3][i]
    #TODO : error case
    
#===============================================================================


class PreProgramedVacuumAgent(Agent):
    "An agent that chooses an action at random, ignoring all percepts."
    def __init__(self, actions):
        Agent.__init__(self)
        self.bump = False
        self.heading = headings[0]
        self.action_index=-1
        def program (percept):       
            self.action_index=self.action_index+1                                                                                                                                
            return actions[(self.action_index  % len(actions))]  

        self.program = program
 


# ==============================================================================
# ==============================================================================

def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    envs = [EnvFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(envs))) 
            for A in AgentFactories]

def test_agent(AgentFactory, steps, envs):
    "Return the mean score of running an agent in each of the envs, for steps"
    total = 0
    for env in envs: 
        agent = AgentFactory()
        env.add_object(agent)
        env.run(steps)
        total += agent.performance
    return float(total)/len(envs)
# ==============================================================================


def vacuumEnv4x4(): 
    env = VacuumEnvironment(4 ,4 )
    #env.add_object(TraceAgent(PreProgramedVacuumAgent(['TurnRight', 'TurnRight', 'Forward', 'Forward', 'Forward', 'TurnRight','TurnRight'])))
    
    env.add_object(Wall(), (2,2))
    env.add_object(Wall(), (2,3)) 
    env.add_object(Wall(), (3,3))
    
    
    
    env.add_object(Dirt(), (1,1))
    env.add_object(Dirt(), (1, 2))
    env.add_object(Dirt(), (3,4))
    env.add_object(Dirt())
    env.add_object(Dirt())
    env.add_object(Dirt())
    
    return env

# ==============================================================================
# 
env =vacuumEnv4x4()
#env.add_object(TraceAgent(RandomVacuumAgent()),  (1,1))
#env.add_object(TraceAgent(ReflexVacuumAgent()),  (1,1))
env.add_object(TraceAgent(ModelBasedVacuumAgent()),  (1,1))
# # 
PRINT_STEP = True
printEnv(env)
# #  
STEP_PER_STEP = True  
env.run()
printEnv(env)
# # ==============================================================================
PRINT_STEP = False

#  
#

    
    
    
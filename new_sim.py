import pymunk, random, math, numpy as np
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame, sys
import matplotlib.pyplot as plt

xres,yres = 1400,800

# numerically stable sigmoid
def sigmoid(center, spread, x):
    arg = spread*(x - center)
    if arg < 0:
        r = math.exp(arg)/(1 + math.exp(arg))
    else:
        r = 1/(1 + math.exp(-arg))
    return r

# converts to pygame coordinates
def convCoords(point):
    return int(point[0]), int(yres - point[1])

class Motor():
    def __init__(self, SPACE, BETA, TIME_STEP, LENGTH,
                 ENERGY, STD, DIR_PROB,
                 IMP_SENS, IMP_CENTER, ct):
        # define constants
        # probability parameters
        self.ENERGY = ENERGY
        self.STD = STD
        self.DIR_PROB = DIR_PROB
        self.IMP_SENS = IMP_SENS
        self.IMP_CENTER = IMP_CENTER
        # simulation constants for optimization
        self.TIME_STEP = TIME_STEP
        self.SQRT_DT = math.sqrt(TIME_STEP)
        # body attributes
        self.RADIUS = 1/math.sqrt(math.pi)
        self.MASS = math.pi*(self.RADIUS)**2
        self.FORCE_MAG = -math.sqrt(2*self.MASS*self.ENERGY)*math.log(BETA)
        TERM_VEL = math.sqrt(2*self.ENERGY/self.MASS)
        self.force_vector = 0,0
        # create pymunk body object at initial position with
        # a random velocity less in magnitude than the terminal velocity
        self.body = pymunk.Body()
        self.shape = pymunk.Circle(self.body, self.RADIUS)
        init_xpos = random.uniform(-0.9*LENGTH/2, 0.9*LENGTH/2)
        self.body.position = init_xpos, 0
        init_xvel = random.uniform(-TERM_VEL, TERM_VEL)
        self.body.velocity = init_xvel, 0
        self.momentum = self.MASS*self.body.velocity.x
        self.direction = np.sign(self.body.velocity.x)
        self.shape.density = 1
        self.shape.elasticity = 1
        # so that motors don't collide with each other
        self.shape.filter = pymunk.ShapeFilter(group=1)
        self.shape.collision_type = ct
        SPACE.add(self.body, self.shape)
        # separate handler that changes the direction of the applied force
        # if the direction of the velocity has changed,
        # change the dir of the force
    
    # applies the expected force plus some noise to the motor every time step
    # the force is always in the direction of the velocity
    def apply_force(self):
        noise = np.random.normal(0, self.STD)
        self.force_vector = self.direction*(self.FORCE_MAG + noise*self.SQRT_DT),0
        self.body.apply_force_at_local_point(self.force_vector, (0, 0))
        
    # collision handler: switches the direction of the applied force 
    # can happen in two ways
    # 1. every time step there is a random probability of the switching force
    # direction. This probability is assigned to each motor and it is allowed
    # to mutate going into the next cell simulation
    # 2. the larger the external impulse, the more likely to switch directions
    def change_dir(self):
        new_momentum = self.MASS*self.body.velocity.x
        impulse = new_momentum - self.momentum
        self.momentum = new_momentum
        external_impulse = np.abs(impulse - self.force_vector[0]*self.TIME_STEP)
        switch_prob = sigmoid(self.IMP_CENTER, self.IMP_SENS, external_impulse)
        rand_number = np.random.random()
        if (rand_number < switch_prob or rand_number < self.DIR_PROB*self.TIME_STEP):
            self.direction = self.direction * (-1)
            
    def step(self):
        self.apply_force()
        self.change_dir()
        
    def draw(self, screen):
        posx = self.body.position.x/2 + xres/2
        posy = yres/2
        color = (254,0,0)
        coords = convCoords((posx - self.RADIUS/2, posy + self.RADIUS/2))
        shape = pygame.Rect(coords, (self.RADIUS,self.RADIUS))
        pygame.draw.ellipse(screen, color, shape)
    
class Wall:
    def __init__(self, SPACE, init_x, init_y, height, thick, ct):
        self.height = height
        self.thick = thick
        self.body = pymunk.Body()
        self.body.position = init_x, init_y
        self.shape = pymunk.Segment(self.body
                                    , (0, -self.height / 2)
                                    , (0, self.height / 2)
                                    , self.thick / 2)
        self.shape.density = 0.2
        self.shape.elasticity = 0.1
        self.shape.collision_type = ct
        SPACE.add(self.body, self.shape)
        
    def draw(self, screen):
        posx = self.body.position.x/2 + xres/2
        posy = yres/2
        angle = self.body.angle
        xpos = self.height * math.sin(angle)
        ypos= self.height * math.cos(angle)
        color = (0,0,0)
        pygame.draw.line(screen, color,
                         (posx - xpos,posy - ypos),
                         (posx + xpos, posy + ypos), 2)
        
class Spring:
    def __init__(self, SPACE, body1, body2, length, young, damp):
        self.BODY1 = body1
        self.BODY2 = body2
        joint = pymunk.DampedSpring(self.BODY1, self.BODY2, (0, 0), (0, 0),
                                    length, young, damp)
        self.joint = joint
        SPACE.add(joint)

class Cell:
    
    def make_motor(self, *args):
        return Motor(self.SPACE, self.BETA,
                     self.TIME_STEP, self.LENGTH,
                     *args)
    
    def make_spring(self, *args):
        return Spring(self.SPACE, *args)
    
    def make_wall(self, *args):
        return Wall(self.SPACE, *args)
    
    # global_args = (duration, frames per second, damping coeff, energy budget)
    # sim_args = (k1, k2, motor args)
    def __init__(self, *args, **kwargs):
        np.random.seed()
        self.LENGTH = 1000
        self.WALL_SEP = 20
        self.args = args
        self.kwargs = kwargs
        settings_keys = ('DURATION','FPS','BETA','BUDGET')
        try:
            for key in settings_keys:
                assert(key in kwargs)
                setattr(self, key, kwargs[key])
        except:
            raise
        self.TIME_STEP = 1/self.FPS
        self.ERROR = False
        
        self.distance = None
        self.motors = None
        
        # cell attributes
        self.n_motors = 0
        self.motor_density = self.n_motors/self.LENGTH
        self.pol = None
        self.avg_k = None
        # motor attributes
        self.m_avg_E = None
        self.m_avg_std = None
        self.m_avg_dirp = None
        self.m_avg_sens = None
        self.m_avg_cent = None
        
    def initialize(self, K1, K2, K, m_dens, m_std,
                     dir_prob, imp_sens):
        self.SPACE = pymunk.Space()
        self.SPACE.damping = self.BETA
        self.K1 = K1
        self.K2 = K2
        self.K = K
        n_motors = int(m_dens*self.LENGTH)
        E_per_motor = self.BUDGET/n_motors
        term_impulse = math.sqrt(2*E_per_motor)
        self.motors = np.asarray(
            [self.make_motor(E_per_motor, m_std, dir_prob,
                             imp_sens, term_impulse, i) 
                             for i in range(n_motors)], dtype='O')
    
    # where all mutations take place
    def inherit(self, parent = None):
        if parent is None:
            parent = self
        self.SPACE = pymunk.Space()
        self.SPACE.damping = self.BETA
        total_E = 0
        i = 0
        self.K1 = parent.K1*(0.5 + np.random.beta(12,12))
        self.K2 = parent.K2*(0.5 + np.random.beta(12,12))
        self.K = parent.K
        motors = []
        while(True):
            parent_motor = np.random.choice(parent.motors)
            mutations = (0.5 + np.random.beta(12,12,5))
            old_params = np.asarray([parent_motor.ENERGY,parent_motor.STD, parent_motor.DIR_PROB,
                        parent_motor.IMP_SENS, parent_motor.IMP_CENTER])
            new_params = old_params*mutations
            total_E += new_params[0]
            if total_E > self.BUDGET:
                break
            child_motor = self.make_motor(*new_params, i + 4)
            motors.append(child_motor)
            i+=1
        self.motors = np.asarray(motors, dtype = 'O')
    
    # calculates useful quantities like polarization, average spring constant,
    # number of motors, motor density, average motor energy, average parameters
    # for motor sigmoid, etc
    def calculate(self):
        self.pol = self.K2/self.K1
        self.avg_k = (self.K2 + self.K1)/2
        self.n_motors = self.motors.size
        self.motor_density = self.n_motors/self.LENGTH
        self.m_avg_E = np.mean(
            [motor.ENERGY for motor in self.motors])
        self.m_avg_std = np.mean(
            [motor.STD for motor in self.motors])
        self.m_avg_dirp = np.mean(
            [motor.DIR_PROB for motor in self.motors])
        self.m_avg_sens = np.mean(
            [motor.IMP_SENS for motor in self.motors])
        self.m_avg_cent = np.mean(
            [motor.IMP_CENTER for motor in self.motors])
            
    def run(self):
        if self.motors is None:
            raise Exception("Motors not initialized")
        elapsed_secs = 0
        frame = 0
        # store commonly used values for optimization
        duration = self.DURATION
        # initialize all objects
        in_wall1_pos = -self.LENGTH/2
        out_wall1_pos = in_wall1_pos - self.WALL_SEP
        y_pos = 0
        spring_damp = 0
        inner_wall1 = self.make_wall(in_wall1_pos, y_pos, 20, 5, 1)
        inner_wall2 = self.make_wall(-in_wall1_pos, y_pos, 20, 5, 2)
        outter_wall1 = self.make_wall(out_wall1_pos, y_pos, 40, 10, 3)
        outter_wall2 = self.make_wall(-out_wall1_pos, y_pos, 40, 10, 4)   
        self.make_spring(inner_wall1.body, outter_wall1.body,
                         self.WALL_SEP, self.K1, spring_damp)
        self.make_spring(inner_wall2.body, outter_wall2.body,
                         self.WALL_SEP, self.K2, spring_damp)
        self.make_spring(outter_wall1.body, outter_wall2.body,
                         self.LENGTH + 2*self.WALL_SEP, self.K, spring_damp)
        # do the simulation
        while(elapsed_secs < duration):
            [motor.step() for motor in self.motors]
            if(elapsed_secs%1 == 0):
                # check for error
                if (inner_wall1.body.velocity.y != 0 or
                    inner_wall2.body.velocity.y != 0):
                    # if there is a numerical error,
                    # make the cell very unlikely to reproduce
                    self.distance = -10
                    self.ERROR = True
                    return
                markerx = (outter_wall1.body.position.x + 
                           outter_wall2.body.position.x) / 2
                self.distance = markerx/self.LENGTH
            elapsed_secs = frame/self.FPS
            frame += 1
            self.SPACE.step(self.TIME_STEP)
        self.calculate()

    def visual_run(self):
        pygame.init()
        screen = pygame.display.set_mode((xres,yres))
        clock = pygame.time.Clock()
        
        elapsed_secs = 0
        frame = 0
        # store commonly used values for optimization
        duration = self.DURATION
        # initialize all objects
        in_wall1_pos = -self.LENGTH/2
        out_wall1_pos = in_wall1_pos - self.WALL_SEP
        y_pos = 0
        spring_damp = 0
        inner_wall1 = self.make_wall(in_wall1_pos, y_pos, 20, 5, 1)
        inner_wall2 = self.make_wall(-in_wall1_pos, y_pos, 20, 5, 2)
        outter_wall1 = self.make_wall(out_wall1_pos, y_pos, 40, 10, 3)
        outter_wall2 = self.make_wall(-out_wall1_pos, y_pos, 40, 10, 4)   
        self.make_spring(inner_wall1.body, outter_wall1.body,
                         self.WALL_SEP, self.K1, spring_damp)
        self.make_spring(inner_wall2.body, outter_wall2.body,
                         self.WALL_SEP, self.K2, spring_damp)
        self.make_spring(outter_wall1.body, outter_wall2.body,
                         self.LENGTH + 2*self.WALL_SEP, self.K, spring_damp)
        # do the simulation
        while(elapsed_secs < duration):
            # quits the simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            screen.fill((250,250,250))
            [motor.step() for motor in self.motors]
            [motor.draw(screen) for motor in self.motors]
            inner_wall1.draw(screen)
            inner_wall2.draw(screen)
            outter_wall1.draw(screen)
            outter_wall2.draw(screen)
            if(elapsed_secs%1 == 0):
                # check for error
                if (inner_wall1.body.velocity.y != 0 or
                    inner_wall2.body.velocity.y != 0):
                    # if there is a numerical error,
                    # make the cell very unlikely to reproduce
                    self.distance = -10
                    self.ERROR = True
                    return
                markerx = (outter_wall1.body.position.x + 
                           outter_wall2.body.position.x) / 2
                self.distance = markerx/self.LENGTH
            elapsed_secs = frame/self.FPS
            frame += 1
            self.SPACE.step(self.TIME_STEP)
            pygame.display.update()
            clock.tick(self.FPS)
        self.calculate()
        pygame.quit()
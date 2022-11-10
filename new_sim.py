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
    def __init__(self, BETA, TIME_STEP, LENGTH,
                 ENERGY, COV, DIR_PROB,
                 IMP_SENS, IMP_CENTER, ct, coll_hand = None):
        # define constants
        # probability parameters
        self.ENERGY = ENERGY
        self.COV = COV
        self.DIR_PROB = DIR_PROB
        self.IMP_SENS = IMP_SENS
        self.IMP_CENTER = IMP_CENTER
        # simulation constants for optimization
        self.TIME_STEP = TIME_STEP
        self.SQRT_DT = math.sqrt(TIME_STEP)
        # body attributes
        self.RADIUS = 1
        self.MASS = math.pi*(self.RADIUS)**2
        self.FORCE_MAG = -math.sqrt(2*self.MASS*self.ENERGY)*math.log(BETA)
        TERM_VEL = math.sqrt(2*self.ENERGY/self.MASS)
        self.force_vector = 0,0
        # create pymunk body object at initial position with
        # a random velocity less in magnitude than the terminal velocity
        self.body = pymunk.Body()
        self.shape = pymunk.Circle(self.body, self.RADIUS)
        init_xpos = random.uniform(-0.8*LENGTH/2, 0.8*LENGTH/2)
        self.body.position = init_xpos, 0
        init_xvel = random.choice([-TERM_VEL, TERM_VEL])
        self.body.velocity = init_xvel, 0
        self.momentum = self.MASS*self.body.velocity.x
        self.direction = np.sign(self.body.velocity.x)
        self.shape.density = 1
        self.shape.elasticity = 1
        # so that motors don't collide with each other
        self.shape.filter = pymunk.ShapeFilter(group=1)
        self.shape.collision_type = ct
        # choose the collision handler
        if coll_hand is None:
            self.switched = False
            self.COV1 = COV
            self.set_coll_handler('velswitch')
        else: 
            self.set_coll_handler(coll_hand)
        # separate handler that changes the direction of the applied force
        # if the direction of the velocity has changed,
        # change the dir of the force
    
    # applies the expected force plus some noise to the motor every time step
    # the force is always in the direction of the velocity
    def apply_force(self):
        noise = np.random.normal(0, self.FORCE_MAG*self.COV)
        #! this is not step-size independent
        self.force_vector = self.direction*(self.FORCE_MAG + noise),0
        self.body.apply_force_at_local_point(self.force_vector, (0, 0))
        
    # sigmoid activation as a function of impulse, lots of issues
    # collision handler: switches the direction of the applied force 
    # can happen in two ways
    # 1. every time step there is a random probability of the switching force
    # direction. This probability is assigned to each motor and it is allowed
    # to mutate going into the next cell simulation
    # 2. the larger the external impulse, the more likely to switch directions
    def change_dir_sigmoid(self):
        new_momentum = self.MASS*self.body.velocity.x
        impulse = new_momentum - self.momentum
        self.momentum = new_momentum
        external_impulse = np.abs(impulse - self.force_vector[0]*self.TIME_STEP)
        switch_prob = sigmoid(self.IMP_CENTER, self.IMP_SENS, external_impulse)
        rand_number = np.random.random()
        if (rand_number < switch_prob or rand_number < self.DIR_PROB*self.TIME_STEP):
            self.direction = self.direction * (-1)
    
    # collision handler: switches direction of the applied force
    # can happen in two ways   
    # 1. every time step there is a random probability of the switching force
    # direction. This probability is assigned to each motor and it is allowed
    # to mutate going into the next cell simulation
    # 2. if the direction of velocity changes, the direction of the force is switched
    def change_dir_velswitch(self):
        direction = np.sign(self.body.velocity.x)
        if self.switched is True: 
            # if force and velocity in same direction, the switch is complete
            # turn noise back on
            if (direction == self.direction):
                    self.switched = False
                    self.COV = self.COV1
        else:
            if (direction == self.direction * (-1)):
                self.direction = self.direction * (-1)
            # random chance to flip force direction 
            # probability of switching in 1 second = dir_prob
            elif (random.random() < self.DIR_PROB*self.TIME_STEP):
                self.direction = self.direction * (-1)
                self.switched = True
                # turn off noise to guarantee that the switch happens
                self.COV = 0
    
    def set_coll_handler(self, option):
        if option == 'sigmoid':
            self.change_dir = self.change_dir_sigmoid
        elif option == 'velswitch':
            self.change_dir = self.change_dir_velswitch
        else:
            raise Exception("Invalid option")
            
    def step(self):
        self.apply_force()
        self.change_dir()
        
    def draw(self, screen):
        posx = self.body.position.x/2 + xres/2
        posy = yres/2
        color = (254,0,0)
        draw_radius = 4*self.RADIUS
        coords = convCoords((posx - draw_radius, posy + draw_radius))
        shape = pygame.Rect(coords, (2*draw_radius, 2*draw_radius))
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
        self.shape.density = 0.15
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
                         (posx + xpos, posy + ypos), 3)
        
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
        return Motor(self.BETA,
                     self.TIME_STEP, self.LENGTH,
                     *args)
    
    def make_spring(self, *args):
        return Spring(self.SPACE, *args)
    
    def make_wall(self, *args):
        return Wall(self.SPACE, *args)
    
    # global_args = (duration, frames per second, damping coeff, energy budget)
    # sim_args = (k1, k2, motor args)
    def __init__(self, kwargs):
        np.random.seed()
        self.LENGTH = 1000
        self.WALL_SEP = 20
        self.DURATION = kwargs['duration']
        self.FPS = kwargs['FPS']
        self.BETA = kwargs['beta']
        self.BUDGET = kwargs['budget']
        self.TIME_STEP = 1/self.FPS
        self.ERROR = False
        
        self.distance = None
        self.motors = None
        
    def initialize(self, kwargs):
        self.K1 = kwargs['k1']
        self.K2 = kwargs['k2']
        self.K = kwargs['K']
        n_motors = int(kwargs['m_dens']*self.LENGTH)
        E_per_motor = self.BUDGET/n_motors
        # places the sigmoid center at the 
        term_impulse = 3.14*math.sqrt(2*E_per_motor/3.14)
        self.motors = np.asarray(
            [self.make_motor(E_per_motor, kwargs['m_cov'], kwargs['dir_prob'],
                             kwargs['imp_sens'], term_impulse, i) 
                             for i in range(n_motors)], dtype='O')
    
    # where all mutations take place
    def inherit(self, parent = None):
        if parent is None:
            parent = self
        total_E = 0
        i = 0
        self.K1 = parent.K1*(0.5 + np.random.beta(12,12))
        self.K2 = parent.K2*(0.5 + np.random.beta(12,12))
        self.K = parent.K
        motors = []
        while(True):
            parent_motor = np.random.choice(parent.motors)
            mutations = (0.5 + np.random.beta(12,12,5))
            old_params = np.asarray([parent_motor.ENERGY,parent_motor.COV, parent_motor.DIR_PROB,
                        parent_motor.IMP_SENS, parent_motor.IMP_CENTER])
            new_params = old_params*mutations
            total_E += new_params[0]
            if total_E > self.BUDGET:
                break
            child_motor = self.make_motor(*new_params, i + 4)
            motors.append(child_motor)
            i+=1
        self.motors = np.asarray(motors, dtype = 'O')
            
    def space_init(self):
        self.SPACE = pymunk.Space()
        self.SPACE.damping = self.BETA
        if self.motors is None:
            raise Exception("Motors not initialized")
        for motor in self.motors:
            self.SPACE.add(motor.body,motor.shape)
        # initialize all objects
        in_wall1_pos = -self.LENGTH/2
        out_wall1_pos = in_wall1_pos - self.WALL_SEP
        y_pos = 0
        spring_damp = 0
        wall_height = 20
        wall_width = 5
        inner_wall1 = self.make_wall(in_wall1_pos, y_pos, wall_height, wall_width, 1)
        inner_wall2 = self.make_wall(-in_wall1_pos, y_pos, wall_height, wall_width, 2)
        outer_wall1 = self.make_wall(out_wall1_pos, y_pos, 2*wall_height, 2*wall_width, 3)
        outer_wall2 = self.make_wall(-out_wall1_pos, y_pos, 2*wall_height, 2*wall_width, 4)   
        self.make_spring(inner_wall1.body, outer_wall1.body,
                         self.WALL_SEP, self.K1, spring_damp)
        self.make_spring(inner_wall2.body, outer_wall2.body,
                         self.WALL_SEP, self.K2, spring_damp)
        self.make_spring(outer_wall1.body, outer_wall2.body,
                         self.LENGTH + 2*self.WALL_SEP, self.K, spring_damp)
        return inner_wall1, inner_wall2, outer_wall1, outer_wall2

    def run(self):
        elapsed_secs = 0
        frame = 0
        # store commonly used values for optimization
        duration = self.DURATION
        inner_wall1, inner_wall2, outer_wall1, outer_wall2 = self.space_init()
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
                markerx = (outer_wall1.body.position.x + 
                           outer_wall2.body.position.x) / 2
                self.distance = markerx/self.LENGTH
            elapsed_secs = frame/self.FPS
            frame += 1
            self.SPACE.step(self.TIME_STEP)

    def detailed_run(self):
        elapsed_secs = 0
        frame = 0
        # store commonly used values for optimization
        duration = self.DURATION
        inner_wall1, inner_wall2, outer_wall1, outer_wall2 = self.space_init()
        # do the simulation
        inwall1_pos = []
        inwall2_pos = []
        steps = int(duration/self.TIME_STEP)
        mots_pos = np.zeros((self.motors.size, steps))
        mots_mom = np.zeros((self.motors.size, steps))
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
            markerx = (outer_wall1.body.position.x + 
                        outer_wall2.body.position.x) / 2
            self.distance = markerx/self.LENGTH
            inwall1_pos.append(inner_wall1.body.position.x)
            inwall2_pos.append(inner_wall2.body.position.x)
            for i,motor in enumerate(self.motors):
                mots_pos[i,frame-1] = motor.body.position.x
            for i,motor in enumerate(self.motors):
                mots_mom[i,frame-1] = motor.momentum
            elapsed_secs = frame/self.FPS
            frame += 1
            self.SPACE.step(self.TIME_STEP)
        return inwall1_pos, inwall2_pos, mots_pos, mots_mom


    def visual_run(self):
        pygame.init()
        screen = pygame.display.set_mode((xres,yres))
        clock = pygame.time.Clock()
        
        elapsed_secs = 0
        frame = 0
        # store commonly used values for optimization
        duration = self.DURATION
        # initialize all objects
        inner_wall1, inner_wall2, outer_wall1, outer_wall2 = self.space_init()
        # do the simulation
        while(elapsed_secs < duration):
            # quits the simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            screen.fill((250,250,250))
            [motor.step() for motor in self.motors]
            inner_wall1.draw(screen)
            inner_wall2.draw(screen)
            outer_wall1.draw(screen)
            outer_wall2.draw(screen)
            [motor.draw(screen) for motor in self.motors]
            if(elapsed_secs%1 == 0):
                # check for error
                if (inner_wall1.body.velocity.y != 0 or
                    inner_wall2.body.velocity.y != 0):
                    # if there is a numerical error,
                    # make the cell very unlikely to reproduce
                    self.distance = -10
                    self.ERROR = True
                    return
                markerx = (outer_wall1.body.position.x + 
                           outer_wall2.body.position.x) / 2
                self.distance = markerx/self.LENGTH
            elapsed_secs = frame/self.FPS
            frame += 1
            self.SPACE.step(self.TIME_STEP)
            pygame.display.update()
            clock.tick(self.FPS)
        pygame.quit()
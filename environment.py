from random import randint
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from abc import abstractmethod
import torch
import math
from transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

tof_half_fov_deg = 10  # 0.5 fov of single mr beam
noise_per_dist_meter = 0.02  # additive noise (after averaging) 
max_range = 3.5

class VelocityHandler:
    
    def __init__(self, linear_step, angular_step, max_linear, max_angular):

        self.linear_step = linear_step
        self.angular_step = angular_step
        self.max_linear = max_linear
        self.max_angular = max_angular

        self.parameters = {"class": self.__class__, "linear_step": linear_step, "angular_step": angular_step,
        "max_linear": max_linear, "max_angular": max_angular}

        self.arrange()

    
    def arrange(self):

        self.linear_bins = int(-((-self.max_linear)//self.linear_step))
        self.angular_bins = int(-((-2*self.max_angular)//self.angular_step))
        self.linear_step = self.max_linear/self.linear_bins
        self.angular_step = 2*self.max_angular/self.angular_bins

    
    def category_to_velocity(self, action :int):

        # action must be max total_bins
        linear_ind = action//(self.angular_bins+1)
        angular_ind = action%(self.angular_bins+1)

        linear_vel = self.linear_step * linear_ind
        angular_vel = self.angular_step * angular_ind - self.max_angular

        return linear_vel, angular_vel
    
    @property
    def total_bins(self):

        return (self.linear_bins+1)*(self.angular_bins+1)
        
    
'''    @property
    def action_zero(self):

        return int(self.max_angular//self.angular_step)'''


class SimpleVelocityHandler:
    
    def __init__(self, linear_vel=0.5, angular_vel=0.5):

        self.linear_vel = linear_vel
        self.angular_vel = angular_vel

        self.parameters = {"class": self.__class__, "linear_vel": linear_vel, "angular_vel": angular_vel}
    
    def category_to_velocity(self, action :int):

        # action must be max total_bins
        linear_vel = self.linear_vel
        if action==0:
            angular_vel = -self.angular_vel
        elif action==1:
            angular_vel = 0
        else:
            angular_vel = self.angular_vel

        return linear_vel, angular_vel
    
    @property
    def total_bins(self):

        return 3

class SimpleTransformer:
    
    def __init__(self, x=0.05, y=0.03):

        self.x = x
        self.y = y

        self.parameters = {"class": self.__class__, "x": x, "y": y}

    
    def category_to_velocity(self, action :int):

        # action must be max total_bins
        x = self.x
        if action==0:
            y = -self.y
        elif action==1:
            y = 0
        else:
            y = self.y

        return x, y
    
    @property
    def total_bins(self):

        return 3

class SimpleTransformer2:
    
    def __init__(self, x=0.05, y=0.03):

        self.x = x
        self.y = y

        self.parameters = {"class": self.__class__, "x": x, "y": y}
    
    def category_to_velocity(self, action :int):

        # action must be max total_bins
        if action==0:
            x = self.x/2
            y = -self.y
        elif action==1:
            x = self.x
            y = 0
        else:
            x = self.x/2
            y = self.y

        return x, y
    
    @property
    def total_bins(self):

        return 3

class Env:
    def __init__(self, velocity_handler, rewarder, model_name="turtlebot3_waffle", num_of_bins = 360, random_init=False, verbose=False):
        
        # assert 360%num_of_bins == 0, f"num_of_bins ({num_of_bins}) should devide 360"

        self.velocity_handler = velocity_handler
        self.rewarder = rewarder
        self.rewarder.convention({"ranges":[0,num_of_bins], 
        "position":[num_of_bins,num_of_bins+2], 
        "orientation":[num_of_bins+2,num_of_bins+4], 
        "velocity":[num_of_bins+4,num_of_bins+6]})
        self.model_name = model_name
        self.random_init = random_init
        self.verbose = verbose
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.num_of_bins = num_of_bins
        self.eval_=False

        self.parameters = {"class": self.__class__, 
        "velocity_handler": self.velocity_handler.parameters, 
        "rewarder": self.rewarder.parameters,
        "num_of_bins": self.num_of_bins}

        self.create_states()
    
    def preprocess_ranges(self, ranges):

        '''ranges = torch.Tensor(ranges)
        if self.num_of_bins==360:
            return ranges
        ranges = torch.reshape(ranges, (self.num_of_bins, -1))
        ranges = torch.mean(ranges, -1)#.values
        #clip the ranges (there are inf)
        return torch.clip(torch.FloatTensor(ranges), 0.0, 10.0)'''
        new_ranges = []
        for i in range(self.num_of_bins):
            start_angle = i*360/self.num_of_bins - tof_half_fov_deg
            end_angle = start_angle + 2*tof_half_fov_deg
            start_angle = math.floor(start_angle)
            if start_angle<0:
                start_angle += 360
            end_angle = math.ceil(end_angle)
            if end_angle>=360:
                end_angle -= 360
            
            current_ranges = []
            ind = start_angle - 1
            while ind != end_angle:
                ind += 1
                if ind>=360:
                    ind -= 360
                if ranges[ind]<max_range:
                    current_ranges.append(ranges[ind])
            new_ranges.append(max_range if len(current_ranges)==0 else sum(current_ranges)/len(current_ranges))
        new_ranges = torch.Tensor(new_ranges)
        new_ranges += noise_per_dist_meter*new_ranges*torch.normal(mean=torch.zeros(self.num_of_bins), std=torch.ones(self.num_of_bins))
        return new_ranges
    
    def create_states(self):
        
        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = -1.67
        state.pose.position.y = 1.46
        state.pose.position.z = 0
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = -0.894
        state.pose.orientation.w = 0.448

        self.init_states = [state]

        if self.random_init:
            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 0.0
            state.pose.position.y = -2.1
            state.pose.position.z = 0
            state.pose.orientation.x = 0
            state.pose.orientation.y = 0
            state.pose.orientation.z = 0
            state.pose.orientation.w = 0

            self.init_states.append(state)
            #TODO add more...

            '''state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -0.7
            state.pose.position.y = 3.6
            state.pose.position.z = 0
            state.pose.orientation.x = 0
            state.pose.orientation.y = 0
            state.pose.orientation.z = 0
            state.pose.orientation.w = 0

            self.init_states.append(state)'''
        
        self.num_states = len(self.init_states)
        
        
    
    def initial(self):

        ind = randint(0,self.num_states-1) if self.random_init else 0
        init_state = self.init_states[ind]

        rospy.wait_for_service('/gazebo/set_model_state')

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( init_state )
    
    def eval(self):
        self.eval_=True
        self.rewarder.eval()
    
    def train(self):
        self.eval_=False
        self.rewarder.train()
    
    def get_state(self, model, laser) -> torch.Tensor:

        #extract features
        position = model.pose[-1].position
        position = torch.Tensor([position.x, position.y])
        orientation = model.pose[-1].orientation
        orientation = torch.Tensor([orientation.z, orientation.w])
        linear = model.twist[-1].linear
        linear_vel = (linear.x**2 + linear.y**2)**0.5
        angular_vel = model.twist[-1].angular.z
        #linear, angular = (0, 0) if self.last_action is None else self.env.velocity_handler.category_to_velocity(self.last_action)
        vel = torch.Tensor([linear_vel, angular_vel])
        
        ranges = self.preprocess_ranges(laser.ranges)

        state = torch.cat((ranges, position, orientation, vel), dim=-1)
        return_state = torch.cat((ranges, vel), dim=-1)
        is_done, reward = self.rewarder(state)
        rospy.loginfo(f"position={position}")
        ## Add log here !!!  XXX
        return return_state, is_done, reward
    
    def step(self, action :int):

        twist = Twist()

        control_linear_vel, control_angular_vel = self.velocity_handler.category_to_velocity(action)

        twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

        self.pub.publish(twist)

        if self.verbose:
            rospy.loginfo("linear: " + str((int)(100*control_linear_vel)/100) + ", anglular: " + str((int)(100*control_angular_vel)/100))
    
    @property
    def total_bins(self):

        return self.velocity_handler.total_bins
    
    @property
    def total_observations(self):

        return self.num_of_bins+2

class SimpleEnv (Env):

    def get_state(self, model, laser) -> torch.Tensor:
        
        position = model.pose[-1].position
        self.position = (position.x, position.y)
        orientation = model.pose[-1].orientation
        orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.z = euler_from_quaternion(orientation)

        #extract features
        position = model.pose[-1].position
        position = torch.Tensor([position.x, position.y])
        orientation = model.pose[-1].orientation
        orientation = torch.Tensor([orientation.z, orientation.w])
        linear = model.twist[-1].linear
        linear_vel = (linear.x**2 + linear.y**2)**0.5
        angular_vel = model.twist[-1].angular.z
        vel = torch.Tensor([linear_vel, angular_vel])
        
        ranges = self.preprocess_ranges(laser.ranges)
        r_min = np.min(np.array(ranges))
        r_right = np.min(np.array(ranges[-5:-2]))
        state = torch.cat((ranges, position, orientation, vel), dim=-1)
        rospy.loginfo("position=({:.2f},{:.2f}), min range = {:.2f}, right range = {:.2f}".format(position.tolist()[0],position.tolist()[1],r_min, r_right))

        is_done, reward = self.rewarder(state)

        return ranges, is_done, reward
    
    def get_pos_z(self, x, y):

        z = self.z + math.atan(y/x)
        if z<-math.pi:
            z += 2*math.pi
        elif z>math.pi:
            z -= 2*math.pi
        
        dist = (x**2+y**2)**0.5
        x_, y_ = math.cos(z)*dist, math.sin(z)*dist
        curr_x, curr_y = self.position
        pos_x, pos_y = (curr_x+x_, curr_y+y_)
        
        return pos_x, pos_y, z
    
    def step(self, action :int):

        x, y = self.velocity_handler.category_to_velocity(action)

        pos_x, pos_y, z = self.get_pos_z(x, y)
        # # Add noise:
        pos_x += np.random.normal(0, 0.01)
        pos_y += np.random.normal(0, 0.01)
        z += np.random.normal(0, np.deg2rad(2))

        ori = quaternion_from_euler(0,0,z)

        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = pos_x
        state.pose.position.y = pos_y
        state.pose.position.z = 0
        state.pose.orientation.x = ori[0]
        state.pose.orientation.y = ori[1]
        state.pose.orientation.z = ori[2]
        state.pose.orientation.w = ori[3]

        rospy.wait_for_service('/gazebo/set_model_state')

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state )

        if self.verbose:
            rospy.loginfo("x: " + str(x) + ", y: " + str(y))
    
    @property
    def total_observations(self):

        return self.num_of_bins

class SimpleEnvTrain (SimpleEnv):

    def create_states(self):
        
        
        angle = 0 #degrees
        angle = angle/180*3.1415
        ori1 = quaternion_from_euler(0,0,angle)
        ori2 = quaternion_from_euler(0,0,angle-3.14)

        # 0-1 line
        # 2-5 small square
        # 6-9 big square
        # 10-13 meshushe
        # 14-17 circle

        ############### line
        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = -1
        state.pose.position.y = 0.1
        state.pose.position.z = 0
        state.pose.orientation.x = ori1[0]
        state.pose.orientation.y = ori1[1]
        state.pose.orientation.z = ori1[2]
        state.pose.orientation.w = ori1[3]

        self.init_states = [state]

        if self.random_init:

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -1
            state.pose.position.y = -1.4
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)
            ###############

            ############### small square
            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.2
            state.pose.position.y = -16.6
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.2
            state.pose.position.y = -14.9
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.2
            state.pose.position.y = -13.3
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.2
            state.pose.position.y = -18.2
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)
            ###############

            ############### big square
            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = -17
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = -13.2
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = -11.6
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = -18.5
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)
            ###############

            ############### meshushe
            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = 9.5
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = 19.15
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = 20.7
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = -15
            state.pose.position.y = 8.28
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)
            ###############

            ############### circle
            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 13
            state.pose.position.y = 14.18
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.5
            state.pose.position.y = 19.67
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 12.5
            state.pose.position.y = 20.7
            state.pose.position.z = 0
            state.pose.orientation.x = ori1[0]
            state.pose.orientation.y = ori1[1]
            state.pose.orientation.z = ori1[2]
            state.pose.orientation.w = ori1[3]

            self.init_states.append(state)

            state = ModelState()
            state.model_name = self.model_name
            state.pose.position.x = 13
            state.pose.position.y = 12.8
            state.pose.position.z = 0
            state.pose.orientation.x = ori2[0]
            state.pose.orientation.y = ori2[1]
            state.pose.orientation.z = ori2[2]
            state.pose.orientation.w = ori2[3]

            self.init_states.append(state)
            ###############
        
        self.num_states = len(self.init_states)
    
    def initial(self, ind=None):

        # 0-1 line
        # 2-5 small square
        # 6-9 big square
        # 10-13 meshushe
        # 14-17 circle
        if ind is None:
            ind = randint(0,self.num_states-1) if self.random_init else 0
        
        place = None
        if 0<=ind<=1:
            place = 1
        elif 2<=ind<=5:
            place = 2
        elif 6<=ind<=9:
            place = 3
        elif 10<=ind<=13:
            place = 4
        else:
            place = 5
        # ind = randint(14,17) if self.random_init else 0
        init_state = self.init_states[ind]

        rospy.wait_for_service('/gazebo/set_model_state')

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( init_state )
        return place


class Rewarder(object):
    
    #TODO check size of state, according to bins and stuff!!!

    def __init__(self):

        self.stand_exp = 0
        self.num_exp = 0
        self.eval_=False
        
        self.parameters = {"class": self.__class__}
    
    def convention(self, dictionary):

        self.borders = dictionary
    

    def eval(self):
        self.eval_=True

    
    def train(self):
        self.eval_=False
        
    
    def __call__(self, state):
        
        self.num_exp += 1
        is_done, reward = self.get_stats(state)
        if is_done:
            self.stand_exp = 0
            self.num_exp = 0

        return is_done, reward
    
    @abstractmethod
    def get_stats(self, state):
        
        is_done, reward = True, 0

        return is_done, reward

class done:
    def __init__(self, is_done, reason):
        self.is_done = is_done
        self.reason = reason
    def __bool__(self):
        return bool(self.is_done)
    def __repr__(self):
        return str(self.is_done)+': '+str(self.reason)



class SimpleRightHand(Rewarder):

    def __init__(self, init_dest=5):
        self.current_dest = init_dest
        self.out = 0
        self.in_a_row = 0
        super().__init__()
    
    def get_stats(self, state):

        state = state.tolist()

        ranges = state[self.borders["ranges"][0]: self.borders["ranges"][1]]
        l = len(ranges)

        min_threshold = 0.35
        max_threshold = 1#0.65
        max_out = 4
        dest = 3 #in a row

        '''tmp = -(65*l//360)
        right = ranges[-115*l//360:tmp]'''
        right = ranges[-5:-2] # for -5, -4, -3
        
        feedback = 10
        reason = 'none'
        if self.eval_:
            if min(ranges)<min_threshold or min(right)>max_threshold:
                reward = -2
                is_done = True
                reason = 'out of range'
            else:
                reward = 1
                is_done = False
                reason = 'in range'
        
        else:
            if min(ranges)<min_threshold:
                reward = -feedback
                is_done = True
                self.in_a_row = 0
                # print(min(ranges), "min")
                reason = 'under range'
                return done(is_done, reason), reward
            
            if min(right)>max_threshold:
                if self.out>=max_out:
                    self.out = 0
                    reward = -feedback
                    is_done = True
                    self.in_a_row = 0
                    # print(min(right), "max")
                    reason = 'right out & max_out'
                    return done(is_done, reason), reward
                else:
                    self.out += 1
                    reason = f'out = {self.out}'
            
            if self.num_exp>=self.current_dest:
                reward = feedback
                is_done = True
                self.in_a_row += 1
                if self.in_a_row>=dest:
                    self.in_a_row = 0
                    reason = 'reached destination {} in a row: {}'.format(self.in_a_row, self.current_dest)
                    self.current_dest += 10
                    rospy.loginfo(f"Did it!!!!!!!!!!!! Destination is {self.current_dest}")
                else:
                    reason = 'reached destination {} NOT in a row: {}'.format(self.in_a_row, self.current_dest)
                    rospy.loginfo("Did it.")
                return done(is_done, reason), reward
        
            reward = 1
            reason = 'in range'
            is_done = False
            return done(is_done, reason), reward
        
        return done(is_done, reason), reward


class SimpleLine(Rewarder):

    def __init__(self, line_points):
        super().__init__()
        self.line_points = line_points
        self.line_points.append(self.line_points[0])
        self.out = 0
    
    def calc_min_dist(self, position):

        x0, y0 = position
        arr = []
        for i in range(len(self.line_points)-1):
            x1, y1 = self.line_points[i]
            x2, y2 = self.line_points[i+1]
            arr.append(abs(x0*(y2-y1)-y0*(x2-x1)+y1*x2-y2*x1)/(((y2-y1)**2+(x2-x1)**2)**0.5))
        return min(arr)

    
    def get_stats(self, state):

        state = state.tolist()

        ranges = state[self.borders["ranges"][0]: self.borders["ranges"][1]]
        position = state[self.borders["position"][0]: self.borders["position"][1]]
        dist = self.calc_min_dist(position)
        l = len(ranges)

        min_threshold = 0.35
        max_threshold = 1#0.65
        max_out = 1

        tmp = -(75*l//360)
        right = ranges[-105*l//360:tmp]
        
        if self.eval_:
            if min(ranges)<min_threshold or min(ranges)>max_threshold:
                reward = -2
                is_done = True
            
            else:
                reward = 1
                is_done = False
        
        else:
            if min(ranges)<min_threshold:
                reward = -20
                is_done = True
                return is_done, reward
            
            if min(right)>max_threshold:
                if self.out>=max_out:
                    self.out = 0
                    reward = -20
                    is_done = True
                    return is_done, reward
                else:
                    self.out += 1
        
            reward = 2-dist
            is_done = False
            return is_done, reward
        
        return is_done, reward

class SimpleLine2(SimpleLine):
    
    def get_stats(self, state):

        state = state.tolist()

        ranges = state[self.borders["ranges"][0]: self.borders["ranges"][1]]
        position = state[self.borders["position"][0]: self.borders["position"][1]]
        dist = self.calc_min_dist(position)
        l = len(ranges)
        
        max_threshold = 0.1
        max_out = 1

        if dist>max_threshold:
            if self.out>=max_out:
                self.out = 0
                reward = -2
                is_done = True
                return is_done, reward
            else:
                self.out += 1
    
        reward = 1-dist
        is_done = False
        
        return is_done, reward
    

class RightHand(Rewarder):

    #TODO use borders!!!
    def get_stats(self, state):

        #TODO improve
        state = state.tolist()

        ranges = state[:-2]
        vel = state[-2:]
        linear_vel, angular_vel = vel[0], vel[1]

        min_threshold = 0.25
        max_threshold = 0.65#0.55

        '''front = ranges[:30]
        front.extend(ranges[-30:])'''
        right = ranges[-95:-85]


        if linear_vel<=0.2 and -0.2<angular_vel<0.2:
            self.stand_exp += 1
        else:
            self.stand_exp = 0
        
        if min(ranges)<min_threshold or min(right)>max_threshold or self.stand_exp>2:
            reward = -30
            is_done = True
        
        else:
            reward = 0
            is_done = False

            #reward += 2-abs(min(right)-(min_threshold+max_threshold)//2)
            reward += (linear_vel+1)#*3

            '''#going front
            if min(front)>max_threshold:
                reward += linear_vel*5
                if angular_vel<0:
                    reward += angular_vel
            #turning left
            else:
                reward += (angular_vel+1)*5 - linear_vel'''
        
        return is_done, reward

class Travel(Rewarder):

    #TODO use borders!!!
    def get_stats(self, state):

        state = state.tolist()

        ranges = state[:-2]
        vel = state[-2:]
        linear_vel, angular_vel = vel[0], vel[1]

        min_threshold = 0.25
        max_threshold = 0.65#0.55

        if linear_vel<=0.2:
            self.stand_exp += 1
        else:
            self.stand_exp = 0

        if min(ranges)<min_threshold or self.stand_exp>2:
            reward = -30
            is_done = True
        else:
            reward = 2 + linear_vel
            is_done = False

        return is_done, reward

class Right(Rewarder):

    #TODO use borders!!!
    def get_stats(self, state):

        state = state.tolist()

        ranges = state[:-2]
        vel = state[-2:]
        linear_vel, angular_vel = vel[0], vel[1]

        min_threshold = 0.25
        max_threshold = 0.65#0.55

        reward = 5*linear_vel-angular_vel
        is_done = self.num_exp%15==0

        return is_done, reward

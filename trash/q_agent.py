#!/usr/bin/env python
'''
simple script to use LIDAR info to travel with right hand, using Q-learning
https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/
'''

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import sys, select, os
import time
import numpy as np
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

class Qlearning:
    def __init__(self, lr=0.2, discount=0.95, bins=4, bin_size=5, low=0.25, high=1.0, angle=92, min_threshold=0.25, max_threshold=0.55, node="turtlebot3_RightHand", **kwargs):

        self.lr = lr
        self.discount = discount
        self.bins = bins
        self.bin_size = bin_size
        self.low = low
        self.high = high
        self.angle = angle
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        assert lr>0, "The learning rate " + str(lr) + " must be above 0"
        assert 0<discount<1, "The discount " + str(discount) + " must be in (0,1)"
        assert angle%bins==0, "Angle " + str(angle) + " must be divided by bins " + str(bins)
        assert high>low, "High " + str(high) + " must be above low " + str(low)
        assert max_threshold>min_threshold, "Max threshold " + str(max_threshold) + " must be above min threshold " + str(min_threshold)

        DISCRETE_SIZE = np.array([bin_size]*bins)
        DISCRETE_HIGH = np.array([high]*bins)
        self.DISCRETE_LOW = np.array([low]*bins)

        self.discrete_win_size = (DISCRETE_HIGH-self.DISCRETE_LOW)/DISCRETE_SIZE
        
        self.q_table = np.random.uniform(low=-2, high=0, size=(list(DISCRETE_SIZE) + [3]))

        rospy.init_node(node)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    
    def get_discrete_state(self, state):
    
        #getting top right 90 bins
        state = np.array(state[-self.angle:])
        
        #average to N large bins
        angle_size = self.angle//self.bins
        state = np.array([np.mean(state[angle_size*i: angle_size*(i+1)]) for i in range(self.bins)])
        
        discrete_state = (state - self.DISCRETE_LOW)/self.discrete_win_size
        discrete_state = np.clip(discrete_state, 0, self.bin_size-1)

        return tuple(discrete_state.astype(np.int))

    
    def get_stats(self, state):
    
        reward, done = None, None
        
        if min(state[-self.angle:])<self.min_threshold or state[-self.angle]>self.max_threshold:
            reward = -2
            done = True
        
        else:
            reward = 0
            done = False
        
        return done, reward


    def initial(self):

        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle'
        state_msg.pose.position.x = -1.67
        state_msg.pose.position.y = 1.46
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = -0.894
        state_msg.pose.orientation.w = 0.448
        '''state_msg.pose.position.x = 0.0
        state_msg.pose.position.y = -2.1
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0'''

        state_msg.twist.linear.x = 0.2

        rospy.wait_for_service('/gazebo/set_model_state')

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    
    def train(self, every_epoch=25, from_episode=None, checkpoint='saves'):

        self.checkpoint = checkpoint
        if from_episode is None:
            self.episode_num = 1
        else:
            self.q_table = np.load(self.checkpoint+'/table_'+str(from_episode)+'.npy')
            self.episode_num = int(from_episode)+1

        self.every_epoch=every_epoch

        self.initial()
        self.last_state = None

        rospy.Subscriber('scan', LaserScan, self.move)
        rospy.spin()
    
    def test(self, from_episode=None):

        if not from_episode is None:
            self.q_table = np.load('saves/table_'+str(from_episode)+'.npy')

        self.initial()
        self.last_state = None

        rospy.Subscriber('scan', LaserScan, self.test_move)
        rospy.spin()
    
    def test_move(self, data):
        
        ranges = data.ranges
        done, _ = self.get_stats(ranges)
        
        if done:
            rospy.loginfo("Failed. Try again:")

            #start new episode, initial place
            self.initial()

            return None

        state = self.get_discrete_state(ranges)
        
        new_action = np.argmax(self.q_table[state])
        self.env_step(new_action)
    
    def env_step(self, action):

        twist = Twist()

        control_linear_vel = None
        control_angular_vel = None

        if action==0:
            control_linear_vel = 0.2
            control_angular_vel = 0.5

        elif action==1:
            control_linear_vel = 0.2
            control_angular_vel = -0.5

        else:
            control_linear_vel = 0.2
            control_angular_vel = 0.0

        twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

        self.pub.publish(twist)

        rospy.loginfo("linear: " + str(control_linear_vel) + ", anglular: " + str(control_angular_vel))

    
    def move(self, data):
        
        ranges = data.ranges
        done, reward = self.get_stats(ranges)

        state = self.get_discrete_state(ranges)

        
        if not self.last_state is None:

            current_q = np.max(self.q_table[state])
            
            last_action = np.argmax(self.q_table[self.last_state])
            last_q = self.q_table[self.last_state + (last_action,)]
            
            new_q = (1 - self.lr) * last_q + self.lr * (reward + self.discount * current_q)
            if done:
                self.q_table[self.last_state + (last_action,)] = reward
            else:
                self.q_table[self.last_state + (last_action,)] = new_q
        
        
        if done:
            self.last_state = None
            rospy.loginfo("Ended episode "+str(self.episode_num))

            #start new episode, initial place
            self.initial()
            if (self.episode_num)%self.every_epoch==0:
                np.save(self.checkpoint+'/table_'+str(self.episode_num), self.q_table)
            self.episode_num+=1

            return None
        
        new_action = np.argmax(self.q_table[state])
        self.last_state = state
        self.env_step(new_action)


if __name__ == '__main__':

    agent = Qlearning(lr=0.3)

    #agent.train(from_episode=325, checkpoint="exp3")
    agent.test(from_episode=400)
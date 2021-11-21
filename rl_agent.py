#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
import os
import numpy as np
from gazebo_msgs.msg import ModelStates
from data import Experience, ReplayMemory
import torch
import torch.nn as nn
import message_filters
import os
import pickle
import random
import threading
#from tf2_py.transformations import euler_from_quaternion
# a2c, ppo, sac  ,stable baseline 3
# roslaunch tof2lidar lidar2tof.launch

eps_gamma = 0.97

#TODO: waypoints for objection, better reward (max dist), 16 bins instead of 18, loss graph, velocity changes
# objection function better, add 0.1 entropy loss

class Agent_rl:
    def __init__(self, policy, env, node="turtlebot3_rl", device="cpu", **kwargs):

        rospy.init_node(node)
        self.parameters = {}
        
        self.policy = policy
        self.env = env
        self.parameters["policy"] = policy.params
        self.parameters["env"] = env.parameters
        if isinstance(device, str):
            self.device = torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        rospy.loginfo(f"Using device: {self.device}")

    
    def check_parameters(self):

        assert "properties.pkl" in os.listdir(self.folder), f"properties.pkl does not exist in directory {self.folder}."
        a_file = open(f"{self.folder}properties.pkl", "rb")
        parameters = pickle.load(a_file)

        for k in parameters:
            assert k in self.parameters, "different parameters."
            assert self.parameters[k] == parameters[k], "different parameters."


    def train(self, trainer, buffer_size=50, batch_size=10, gamma=0.99, eps=0, gradient_steps=1, off_policy=True, save_every=None, folder=None, from_episode=None, **kw):

        self.memory = ReplayMemory(
            buffer_size = buffer_size, 
            batch_size = batch_size, 
            gamma = gamma, 
            off_policy = off_policy)
        self.trainer = trainer
        
        self.episode_num = 1

        # for exploration, 0 for fully deterministic
        self.eps = eps

        self.gradient_steps = gradient_steps

        self.last_state = None
        self.last_action = None
        self.episode_steps = 0
        self.episode_reward = 0.0

        self.scores = None

        if folder is None:
            folder = "saves"

        # load checkpoint
        directory = os.path.dirname(__file__)  # directory of script
        folder = f"{directory}/{folder}"
        rospy.loginfo(f"Saving on {folder}")
        os.umask(0)
        if not os.path.exists(folder):
            os.makedirs(folder)

        folder += "/"
        self.folder = folder

        if from_episode==-1:
            models_list = [(f, f.split("_")[-1]) for f in os.listdir(folder) if f.startswith("model_")]
            if len(models_list)>0:
                model, episode = max(models_list, key = lambda f: int(f[1]))
                self.check_parameters()
                self.policy.load_state_dict(torch.load(f"{folder}{model}"))
                rospy.loginfo("Loaded from "+model)
                self.episode_num = int(episode) + 1

                if off_policy and os.path.exists(f"{folder}memory.pkl"):
                    self.memory.load_from_data(f"{folder}memory.pkl")
                
                if os.path.exists(f"{folder}eps"):
                    with open(f"{folder}eps", 'rb') as f:
                        self.eps = float(np.load(f))
                
                if os.path.exists(f"{folder}entropy"):
                    with open(f"{folder}entropy", 'rb') as f:
                        self.trainer.entropy_coef = float(np.load(f))
            
        elif not from_episode is None:
            model = f"model_{from_episode}"
            assert model in os.listdir(folder), f"{model} does not exist in directory {folder}."
            self.check_parameters()
            self.policy.load_state_dict(torch.load(f"{folder}{model}"))
            rospy.loginfo("Loaded from "+model)
            self.episode_num = from_episode + 1
            if off_policy and os.path.exists(f"{folder}memory.pkl"):
                self.memory.load_from_data(f"{folder}memory.pkl")
            
            if os.path.exists(f"{folder}eps"):
                with open(f"{folder}eps", 'rb') as f:
                    self.eps = float(np.load(f))
            
            if os.path.exists(f"{folder}entropy"):
                with open(f"{folder}entropy", 'rb') as f:
                    self.trainer.entropy_coef = float(np.load(f))
        
        # if not loaded, save properties
        if self.episode_num==1:
            a_file = open(f"{folder}properties.pkl", "wb")
            pickle.dump(self.parameters, a_file)
            a_file.close()

        # delete last data
        for name in ["reward", "loss", "loss_e", "loss_p", "loss_v", "loss", "distance", "place"]:
            data = []
            if not os.path.exists(f"{folder}{name}"):
                continue
            with open(f"{folder}{name}", 'rb') as f:
                try:
                    while 1:
                        data.append(np.load(f))
                except:
                    data = np.concatenate(data, axis=None)
            with open(f"{folder}{name}", 'wb') as f:
                np.save(f,data[:self.episode_num-1])
        
        self.save_every = save_every

        self.data = {}
        self.data["reward"] = []
        self.data["distance"] = []
        self.data["loss"] = []
        self.data["loss_e"] = []
        self.data["loss_p"] = []
        self.data["loss_v"] = []
        self.data["place"] = []

        place = self.env.initial()
        self.data["place"].append(place)
        
        self.lock_cb = threading.Lock()

        model_sub = message_filters.Subscriber('gazebo/model_states', ModelStates)
        laser_sub = message_filters.Subscriber('scan', LaserScan) #every 0.2 sec
        ts = message_filters.ApproximateTimeSynchronizer([model_sub, laser_sub], queue_size=1, slop=0.05, allow_headerless=True)
        #ts = message_filters.TimeSynchronizer([info, laser], 1)
        ts.registerCallback(self.training_loop)
        rospy.spin()
    
    def eval(self, folder=None, from_episode=None):

        self.episode_num = 1
        self.eps = 0 # for no exploration

        self.last_state = None
        self.last_action = None

        if folder is None:
            folder = "saves/"
        else:
            folder += "/"
        assert os.path.exists(folder), f"folder {folder} does not exist."
        
        self.folder = folder
        
        if from_episode==-1:
            models_list = [(f, f.split("_")[-1]) for f in os.listdir(folder) if f.startswith("model_")]
            if len(models_list)>0:
                model, episode = max(models_list, key = lambda f: int(f[1]))
                self.check_parameters()
                self.policy.load_state_dict(torch.load(f"{folder}{model}"))
                rospy.loginfo("Loaded from "+model)
        elif not from_episode is None:
            model = f"model_{from_episode}"
            assert model in os.listdir(folder), f"{model} does not exist in directory {folder}."
            self.check_parameters()
            self.policy.load_state_dict(torch.load(f"{folder}{model}"))
            rospy.loginfo("Loaded from "+model)
        
        self.env.eval()
        self.env.initial()

        
        model_sub = message_filters.Subscriber('gazebo/model_states', ModelStates)
        laser_sub = message_filters.Subscriber('scan', LaserScan) #every 0.2 sec
        ts = message_filters.ApproximateTimeSynchronizer([model_sub, laser_sub], queue_size=1, slop=0.05, allow_headerless=True)
        #ts = message_filters.TimeSynchronizer([info, laser], 1)
        ts.registerCallback(self.test)
        rospy.spin()
    
    def test(self, model_info, laser_info):
        
        curr_state, is_done, _ = self.env.get_state(model_info, laser_info)
        
        if not is_done:

            self.last_state = curr_state
        
            self.step()
        
        else:

            self.last_state = None
            self.last_action = None
            rospy.loginfo("Ended episode "+str(self.episode_num))

            #start new episode, initial place
            self.env.initial()
            self.episode_num+=1
    
    
    def step(self):
        
        state = torch.unsqueeze(self.last_state, 0).to(self.device)
        
        selected_action, self.scores = self.policy.predict(state)
        if random.random() < self.eps:
            # Exploration
            selected_action = random.choice(np.arange(self.env.total_bins))
            self.eps *= eps_gamma
        self.last_action = selected_action
        self.env.step(selected_action)

        '''with torch.no_grad():
            state = torch.unsqueeze(self.last_state, 0).to(self.device)
            actions_prob = torch.squeeze(self.policy(state))
            actions_prob = nn.functional.softmax(actions_prob, dim=0)
        
        selected_action = actions_prob.multinomial(num_samples=1).item()
        self.last_action = selected_action
        self.env.step(selected_action)'''


    def training_loop(self, model_info, laser_info):
        # TODO: add lock so that an inturruptin ls doesnt disrupt sequence ?
        # '.initial()' doesnt execute ?
        if self.lock_cb.acquire(blocking=False):
            try:
                curr_state, is_done, reward = self.env.get_state(model_info, laser_info)

                # if not initial state
                if not self.last_state is None:
                    experience = Experience(
                            state=self.last_state,
                            next_state=curr_state,
                            scores=self.scores,
                            action=self.last_action,
                            reward=reward,
                            qval=0,
                            is_done=bool(is_done),
                        )
                    self.memory.push_experience(experience)
                    self.episode_steps += 1
                    self.episode_reward += reward
                
                # if not final state
                if not bool(is_done):
                    self.last_state = curr_state
                
                    self.step()
                
                # if final state
                else:
                    print('---- reached final state ----')
                    print(is_done)
                    self.memory.push_episode()
                    # for graphs
                    self.data["reward"].append(self.episode_reward)
                    self.data["distance"].append(self.episode_steps)

                    rospy.loginfo(f"Ended episode {self.episode_num}")
                    

                    # train
                    try:
                        losses = []
                        losses_e, losses_p, losses_v = [], [], []
                        for batch in self.memory.get_batch(self.gradient_steps if self.gradient_steps>0 else self.episode_steps):
                            loss, loss_dict = self.trainer.train(batch)
                            losses.append(loss)
                            losses_e.append(loss_dict["loss_e"] if "loss_e" in loss_dict else -1)
                            losses_p.append(loss_dict["loss_p"] if "loss_p" in loss_dict else -1)
                            losses_v.append(loss_dict["loss_v"] if "loss_v" in loss_dict else -1)
                        
                        def mean(l):
                            return sum(l)/len(l) if len(l)>0 else -1

                        rospy.loginfo(f"total_loss = {mean(losses)}")
                        self.data["loss"].append(mean(losses))
                        self.data["loss_e"].append(mean(losses_e))
                        self.data["loss_p"].append(mean(losses_p))
                        self.data["loss_v"].append(mean(losses_v))
                    
                    except Exception as e:
                        rospy.loginfo(e)
                        self.data["loss"].append(-1)
                        self.data["loss_e"].append(-1)
                        self.data["loss_p"].append(-1)
                        self.data["loss_v"].append(-1)

                    # save
                    if not self.save_every is None and self.episode_num % self.save_every == 0:
                        torch.save(self.policy.state_dict(), f"{self.folder}model_{self.episode_num}")
                        with open(f"{self.folder}reward", 'ab') as f:
                            np.save(f, self.data["reward"])
                        with open(f"{self.folder}loss", 'ab') as f:
                            np.save(f, self.data["loss"])
                        with open(f"{self.folder}loss_e", 'ab') as f:
                            np.save(f, self.data["loss_e"])
                        with open(f"{self.folder}loss_p", 'ab') as f:
                            np.save(f, self.data["loss_p"])
                        with open(f"{self.folder}loss_v", 'ab') as f:
                            np.save(f, self.data["loss_v"])
                        with open(f"{self.folder}distance", 'ab') as f:
                            np.save(f, self.data["distance"])
                        with open(f"{self.folder}place", 'ab') as f:
                            np.save(f, self.data["place"])
                        self.data["reward"] = []
                        self.data["distance"] = []
                        self.data["loss"] = []
                        self.data["loss_e"] = []
                        self.data["loss_p"] = []
                        self.data["loss_v"] = []
                        self.data["place"] = []
                        self.memory.save_data(f"{self.folder}memory.pkl")
                        with open(f"{self.folder}eps", 'wb') as f:
                            np.save(f, self.eps)
                        with open(f"{self.folder}entropy", 'wb') as f:
                            np.save(f, self.trainer.entropy_coef)
                        rospy.loginfo("saved!!!")

                    #start new episode, initial place
                    self.episode_steps = 0
                    self.episode_reward = 0.0
                    self.last_state = None
                    self.last_action = None

                    place = self.env.initial()
                    self.data["place"].append(place)
                    self.episode_num+=1
                    self.lock_cb.release()
                    return None
            except Exception as e:
                self.lock_cb.release()
                raise e

            self.lock_cb.release()

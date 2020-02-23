# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 02:03:30 2020

@author: Kosmas Pinitas
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import gym
import time
from keras.models import Model#, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
import multiprocessing
import tensorflow as tf
from agent import Agent
import os
path = os.path.abspath(os.getcwd())
from helper import preprocessing
import numpy as np
import random
import h5py
# global variables for A3C

env_name = "Assault-v0"
NUM_ACTIONS = NUM_ACTIONS = gym.make(env_name).action_space.n-1 # 6 distinct actions
TEST_MODE = True
LOAD = True if TEST_MODE else False
EPISODES=100
SLEEP_TIME = 600 
VIDEO_IN_TEST_MODE = True

class A3C:
    def __init__(self, action_size,load = False, gamma=0.99,learning_rate = 2.5e-4):
        # env params
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.load=load
        self.wait_at_start =  30
        
        #discount factor
        self.gamma = gamma
        
        # optimizer params
        self.learning_rate = learning_rate
        
        #number of asynchronous agents
        self.n_agents = multiprocessing.cpu_count()

        # create model for actor and critic network or load an existing model
        self.actor, self.critic = self.build_model()
        if self.load:
            self.load_model()
             
        
        # custom oprimizer
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

       
        #run session in order to save scalars into summary logs (tensorboard)
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(path+'\\summary\\assault', self.sess.graph)

    def train(self):
        #initializein asynchronous agents
        agents = [Agent(self.action_size, self.state_size, [self.actor, self.critic], self.sess, self.optimizer,
                        self.gamma, [self.summary_op, self.summary_placeholders,
                        self.update_ops, self.summary_writer],EPISODES,env_name) for _ in range(self.n_agents)]
        
    
        #start threads
        for agent in agents:
            time.sleep(1)
            agent.start()
        
        #save model
        while True:
            time.sleep(SLEEP_TIME)
            self.save_model()
    
    
    def test(self, video = False):
        env = gym.make(env_name)
        if video:
            env = gym.wrappers.Monitor(env=env,directory=path+"\\videos\\assault",force=True)
        done=False
        score = 0
        curr_frame = env.reset()
        
        next_frame = curr_frame

        
        #  do nothing at the start  
        for _ in range(random.randint(1, self.wait_at_start)):
            curr_frame = next_frame
            next_frame, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame. So just append 3 more times the  initial frames
        state = preprocessing(next_frame, curr_frame)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        
        while not done:
            
            env.render()
            
            act = self.actor.predict(np.float32(history / 255.))[0]
            
            actId = np.argmax(act)
            
            permitted_action = actId+1

            next_frame, reward, done, info = env.step(permitted_action)
            next_state = preprocessing(next_frame, curr_frame)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)


            score += reward
            reward = np.clip(reward, -1., 1.)
            history=next_history
        return score
  
    
    def build_model(self):
        #state preprocessing
        model_input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(model_input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        dense = Dense(256, activation='relu')(conv)
        # actor and critic outputs
        policy = Dense(self.action_size, activation='softmax')(dense)
        value = Dense(1, activation='linear')(dense)
        
    
        actor = Model(inputs=model_input, outputs=policy)
        critic = Model(inputs=model_input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic


    
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        # actor loss function 
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)
        #adding entropy in current loss function (loss for policy approximation )
        loss = actor_loss + 0.01*entropy
        optimizer = RMSprop(lr=self.learning_rate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output
        #loss for V approximation
        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.learning_rate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def load_model(self):
        self.actor.load_weights("actor.h5")
        self.critic.load_weights("critic.h5")

    def save_model(self):
        self.actor.save_weights("actor.h5")
        self.critic.save_weights('critic.h5')

    # set up logs (tensorboard)
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)                                     
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op



if __name__ == "__main__":
    global_agent = A3C(action_size=NUM_ACTIONS,load =LOAD)
    if not TEST_MODE:
        global_agent.train()
    else:
        score = global_agent.test(video=VIDEO_IN_TEST_MODE)

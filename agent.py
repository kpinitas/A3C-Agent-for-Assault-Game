# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 02:32:10 2020

@author: Kosmas Pinitas
"""
  
import gym
import random
import threading
import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from helper import preprocessing


global episode_counter
episode_counter = 0 #globa; episode counter



class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, gamma, summary_ops, episodes=10000, env_name='Assault-v0'):
        threading.Thread.__init__(self)

        #coppy params of global model for each agent thread
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.gamma = gamma
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = self.build_localmodel()

        self.avg_p_max = 0
        self.avg_loss = 0
        self.episodes=episodes
        self.env_name=env_name

        self.t_max = 20
        self.t = 0

    def run(self):
        global episode_counter

        env = gym.make(self.env_name)

        step = 0

        while episode_counter < self.episodes:
            done = False
            dead = False
            score = 0
            curr_frame = env.reset()
            next_frame = curr_frame

            # on start do nothing
            for _ in range(random.randint(1, 30)):
                curr_frame = next_frame
                next_frame, _, _, info = env.step(1)
            
            start_life = info['ale.lives']
            
            #append the same frame 3 more times
            state = preprocessing(next_frame, curr_frame)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1
                curr_frame = next_frame
                # get action for the current history and go one step in environment
                action, policy = self.get_action(history)
                
                permitted_action = action+1

                if dead:
                    action = 0
                    permitted_action = action+1
                    dead = False
                
                next_frame, reward, done, info = env.step(permitted_action)
                
                # append the nrw frame after action
                next_state = preprocessing(next_frame, curr_frame)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                #calculate the max prob
                self.avg_p_max += np.amax(self.actor.predict(np.float32(history / 255.)))

                # if they shoot me I die but the episode continues 
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                
                
                #keep the score
                score += reward
                #we don't want big reward values in order to reduce noise
                reward = np.clip(reward, -1., 1.)

                # save a sample (s, a, r, s')
                self.memory(history, action, reward)

                # If the agent dies just reset  the history
                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_localmodel()
                    self.t = 0

                # In the end print the score over episodes
                if done:
                    episode_counter += 1
                    print("episode:", episode_counter, "  score:", score, "  step:", step)

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    #update log files
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode_counter + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0


    # The agent uses samples in order to evaluate the policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1] / 255.))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


    # update policy network and value network every episode
    def train_model(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]
        
        #state normalization
        states = np.float32(states / 255.)
        
        #predictied value for states
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))
        
        #calculate advantages
        advantages = discounted_rewards - values
        
        #optimize networks (actor-critic network)
        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    #same as in aec
    def build_localmodel(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        dense = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(dense)
        value = Dense(1, activation='linear')(dense)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()
        
        #set local model weights as global's
        actor.set_weights(self.actor.get_weights())
        critic.set_weights(self.critic.get_weights())

        actor.summary()
        critic.summary()

        return actor, critic
    
    #set local model weights as global's
    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    #take a random action from policy distribution
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def memory(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


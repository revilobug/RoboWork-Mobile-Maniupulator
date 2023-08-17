import rospy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


class EpisodeBuffer:
    def __init__(self, timesteps_per_batch=1024):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.vals = []
        self.rewards = []
        self.non_terminal = [] 
	self.max_size = timesteps_per_batch

    def __len__(self):
        return len(self.states)
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.non_terminal.append(done)   

    def get_data(self):
	# return as numpy array
	return np.array(self.states), np.array(self.actions), np.array(self.logprobs), np.array(self.vals), np.array(self.rewards), np.array(self.non_terminal)   

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.vals[:]
        del self.rewards[:]
        del self.non_terminal[:]
   
    def is_full(self):
	if self.__len__() == self.max_size:
	    rospy.loginfo("Buffer is full, please clear the buffer.")
            return True
	else:
	    return False

class Actor(Model):
    """Actor Network"""
    def __init__(self, action_dim):
	super(Actor, self).__init__()
	self._layer1 = Dense(units=256, name="actor_l1", kernel_initializer='lecun_normal', activation='relu')
        self._layer2 = Dense(units=256, name="actor_l2", kernel_initializer='lecun_normal', activation='relu')
	self._out = Dense(units=action_dim, name="actor_out", activation='tanh')

    def call(self, input):
        features = self._layer1(input)
        features = self._layer2(features)
	return self._out(features)  

              
class Critic(Model):
    """Critic Network"""
    def __init__(self):
	super(Critic, self).__init__()
	self._layer1 = Dense(units=256, name="critic_l1", kernel_initializer='lecun_normal', activation='relu')
        self._layer2 = Dense(units=256, name="critic_l2", kernel_initializer='lecun_normal', activation='relu')
	self._out = Dense(units=1, name="critic_out", activation='linear')

    def call(self, input):
        features = self._layer1(input)
        features = self._layer2(features)
	return self._out(features) 


class Agent:
    def __init__(self, state_dim, action_dim, value_clip, vf_loss_coef, minibatch, batch_size, ppo_epochs, gamma, lam, learning_rate, save_path, is_training, summary_writer):
	  
	self.state_dimension = state_dim
	self.action_dimension = action_dim  

	self._minibatch = minibatch	
	self.timestep_per_batch = batch_size
        self.batch_data = EpisodeBuffer(self.timestep_per_batch)
	self.ppo_epochs = ppo_epochs

	self._actor = Actor(self.action_dimension)
        self._critic = Critic()
	self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	self.clip_value = value_clip
	self.gamma = gamma
	self.gae_lambda = lam
	self.critic_coef = vf_loss_coef   
	self._std = tf.eye(self.action_dimension) * 0.36
	self._scale = tf.linalg.cholesky(self._std)

	self._save_model_path = save_path

	self.is_training = is_training
	self.a_loss = tf.keras.metrics.Mean('a_loss', dtype=tf.float32)
	self.c_loss = tf.keras.metrics.Mean('c_loss', dtype=tf.float32)
	self.writer = summary_writer

    def act(self, state):
        """
	Queries an action from the actor network, should be called from run_training.
	Params:
		state - the observation at the current timestep
	Return:
		if is_training:
		action_mean - action value 
                else:
		value - the value of the current state
		dist - distribution of mean - action_mean and std - self._std
	"""
	state = tf.expand_dims(tf.cast(state, dtype=tf.float32), 0)
        action_mean = self._actor(state)
        action_mean = tf.squeeze(action_mean)
	value = self._critic(state)
        dist = tfp.distributions.MultivariateNormalTriL(loc=action_mean, scale_tril=self._scale)

        if self.is_training:    
	    return dist, value
        else:   
	    return action_mean  

    def remember(self, state, action, probs, vals, reward, done):
	self.batch_data.store(state, action, probs, vals, reward, done)

    def actor_loss(self, advantages, new_log_probs, old_log_probs):
	ratio = tf.math.exp(new_log_probs - old_log_probs)
	policy_loss = -tf.reduce_mean(tf.math.minimum(ratio * advantages, tf.clip_by_value(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value) * advantages))
	return policy_loss
	
    def critic_loss(self, discounted_rewards, predicted_values):
        return keras.metrics.mean_squared_error(discounted_rewards, predicted_values)

    def generalized_advantage_estimate(self, next_value):
	batch_states, batch_actions, batch_logprobs, batch_values, batch_rewards, batch_masks = self.batch_data.get_data()
	batch_values = np.append(batch_values, next_value)

        returns = []
	advantages = []
	gae = 0
	for step in reversed(range(len(batch_rewards))):
	    delta = batch_rewards[step] + self.gamma * batch_values[step+1] * batch_masks[step] - batch_values[step]
	    gae = delta + self.gamma * self.gae_lambda * batch_masks[step] * gae
	    returns.insert(0, gae + batch_values[step])
	    advantages.insert(0, gae)

	advantages = np.array(advantages, dtype=np.float32)
	returns = np.array(returns, dtype=np.float32)
	# normalize advantages
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
	return returns, advantages

    def train_ppo(self, states, old_probs, actions, advantages, returns):
	"""
	Runs a single learning step for NN.

	"""
	with tf.GradientTape() as tape:
	    dist, value = self.act(states)
	    value = tf.squeeze(value)
	    new_probs = tf.squeeze(dist.log_prob(actions))
	    actor_loss = self.actor_loss(advantages, new_probs, old_probs)
	    critic_loss = self.critic_loss(returns, value)
	    # might add kl divergence
	    total_loss = actor_loss + self.critic_coef * critic_loss
	gradients = tape.gradient(total_loss, self._actor.trainable_variables + self._critic.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._actor.trainable_variables + self._critic.trainable_variables))  
        return actor_loss, critic_loss  
	

    def update_ppo(self, returns, advantages, update_num):
	"""
	Trains PPO for K epochs. Clears the batch.
	"""
	batch_size = int(len(self.batch_data) / self._minibatch)

	batch_states, batch_actions, batch_old_probs, _, _, _ = self.batch_data.get_data()

	batch_states = tf.convert_to_tensor(batch_states, np.float32)
	batch_old_probs = tf.convert_to_tensor(batch_old_probs, np.float32)
	batch_actions = tf.convert_to_tensor(batch_actions, np.float32)
	batch_advantages = tf.convert_to_tensor(advantages, np.float32)
	batch_returns = tf.convert_to_tensor(returns, np.float32)

	train_dataset = tf.data.Dataset.from_tensor_slices((batch_states, batch_old_probs, batch_actions, batch_advantages, batch_returns))
	train_dataset = train_dataset.shuffle(buffer_size=len(self.batch_data)).batch(batch_size)
	
	# Optimize policy for K epochs:
        for epoch in range(self.ppo_epochs):
	    log_step = update_num*self.ppo_epochs + epoch
	    actor_batch_losses = []
	    critic_batch_losses = []
            for (states, old_probs, actions, advantages, returns) in train_dataset:
		actor_loss, critic_loss = self.train_ppo(states, old_probs, actions, advantages, returns)
		actor_batch_losses.append(actor_loss)
	        critic_batch_losses.append(critic_loss)
	    a = self.a_loss(actor_batch_losses)
	    rospy.loginfo("actor batch mean loss {}".format(a))
	    c = self.c_loss(critic_batch_losses)
	    rospy.loginfo("critic batch mean losses {}".format(c))
	    with self.writer.as_default():
		tf.summary.scalar('actor_loss', self.a_loss.result(), step=log_step)
		tf.summary.scalar('critic_loss', self.c_loss.result(), step=log_step)
 	    self.a_loss.reset_states()
            self.c_loss.reset_states()

        # Clear the memory
        self.batch_data.clear()

    def save_weights(self):
        self._actor.save_weights(self._save_model_path + "/bvr_mobile_manipulation/actor_ppo", save_format='tf')
        self._critic.save_weights(self._save_model_path + "/bvr_mobile_manipulation/critic_ppo", save_format='tf')

    def load_weights(self):
        self._actor.load_weights(self._save_model_path + "/bvr_mobile_manipulation/actor_ppo")
        self._critic.load_weights(self._save_model_path + "/bvr_mobile_manipulation/critic_ppo")
	    


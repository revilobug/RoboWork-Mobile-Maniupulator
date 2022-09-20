import rospy
import gym
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.insert(0, '/home/rwl/autonomous_mobile_manipulation_ws/src/ros-gym-mobile-manipulation/scripts/robowork_envs/task_envs')

from std_msgs.msg import String
import bvrSIM_mobile_grasp 
from ppo import Agent

# increment after training run
run_num = 9

def run_training(training_config):
    """
	Runs training on BvrSIM.
	Possible Future Improvements:
	1. adding KL penalty to the loss function
	2. including early stop threshold 
	3. changing training parameters, or CNN structure
	
    """
    
    # create environment
    rospy.loginfo("Creating BvrSIMMobileGrasp...")
    env_name = "BvrSIMMobileGrasp-v0"
    env = gym.make(env_name)
    rospy.loginfo("BvrSIMMobileGrasp created.")

    # create reward publisher
    rl_data_publisher = rospy.Publisher("/mobile_grasp_rl_data", String, queue_size=5)

    # get state and action dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # log using tensorboard
    save_path = training_config["save_model_path"]
    log_dir = save_path + '/PPO_data_logs/' + str(run_num)
    summary_writer = tf.summary.create_file_writer(log_dir)
    average_reward = tf.keras.metrics.Mean('average_reward', dtype=tf.float32)

    # instantiate agent
    agent = Agent(state_dim, action_dim, training_config["value_clip"], training_config["vf_loss_coef"],
                      training_config["minibatch"], training_config["buffer_max_size"], training_config["ppo_epochs"],
                      training_config["gamma"], training_config["lam"], training_config["learning_rate"],
                      training_config["save_model_path"], training_config["is_training"], summary_writer)

    # get neccessary data from parameter server
    num_steps = training_config["num_steps_per_episode"] # max number of steps per episode
    update_every = training_config["steps_before_update"] # frequency of update 
    num_episodes = training_config["num_episodes_training"]
    print_freq = training_config["print_frequency"]
    log_freq = training_config["log_frequency"]
    save_freq = training_config["save_weights_frequency"]

    """
    log_dir = save_path + '/PPO_logs'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_file = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".csv"
    log_f = open(log_file,"w+")
    log_f.write('episode,timestep,reward\n')
    """

    episode_score_history = [] # keeps track of episode scores
    learn_iters = 0 # how many updates on PPO were made
    update_steps = 0 # keep track of timesteps taken in training

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    max_timesteps = 2000000

    wins = 0 # used in early stopping. If the agent achieves the task 5 times in a row stop training

    eps = 0
    while update_steps < max_timesteps:
        #for eps in range(num_episodes): 
	rospy.loginfo("Entering episode: {}".format(eps))
	# after each episode reset the environment, get initial observation
	state = env.reset()
        total_episode_reward = 0

	# while episode is not done or the episode step is less than maximum episode length
	for episode_step in range(1, num_steps+1):
            
	    # choose action
	    # get action distribution (MultivariateNormal), and value of the state
	    dist, val = agent.act(state)
	    # sample
            action = dist.sample()
	  
	    # get observation
	    next_state, reward, done, _ = env.step(action.numpy())

	    # decide on mask for GAE. done - terminal (0.0), not done - non-terminal (1.0)
	    if done["is_done"]:
		mask = 0.0
	    else:
		mask = 1.0

	    # increment update_steps
	    update_steps += 1

	    # get log_probability
	    log_prob = tf.squeeze(dist.log_prob(action))

	    # add episode reward
	    total_episode_reward += reward

            # if episode terminated reward for reaching the goal and penalize for getting stuck
	    if done["is_done"]:	
		if done["is_success"]:
		    total_episode_reward += 100
		    wins += 1
                else:
		    total_episode_reward += -100
		    wins = 0
			
	    # save data
	    agent.remember(state, action, log_prob, val, reward, done["is_done"])

	    # update state 
	    state = next_state

	    if update_steps % update_every == 0: 
		_, next_value = agent.act(next_state)
		returns, advantages = agent.generalized_advantage_estimate(next_value)
		rospy.loginfo("Advantages {}".format(advantages))
		rospy.loginfo("Updating policy using PPO.")
		agent.update_ppo(returns, advantages, learn_iters)
		learn_iters +=1
	    """
 	    if update_steps % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i, update_steps, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
	    """

            if update_steps % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                rospy.loginfo('Episode: {} Score: {} Avg Score: {} Time Steps: {} Learning Steps: {}'.format(eps, episode_step, print_avg_reward, update_steps, learn_iters))

                print_running_reward = 0
                print_running_episodes = 0
	    
	    if update_steps % save_freq == 0:
	        rospy.loginfo("Saving parameters")
                agent.save_weights()

	    if done["is_done"]:
		rospy.loginfo("Ending episode---------------------------------------------------------------------------------------------------------------------------------------------------")
		rospy.loginfo("At step: {}".format(episode_step))
		break
	    rospy.loginfo("Training timestep {}.".format(update_steps))

	eps += 1
	episode_score_history.append(total_episode_reward)

	rospy.loginfo("Logging mean rewards to the file. After episode: {} reward: {}".format(eps, total_episode_reward))
        with summary_writer.as_default():
            tf.summary.scalar('reward', average_reward(episode_score_history), step=eps)
		
	average_reward.reset_states()	

        print_running_reward += total_episode_reward
        print_running_episodes += 1

        log_running_reward += total_episode_reward
        log_running_episodes += 1
	
	if wins == 5:
	    break

    # log_f.close()
    env.close()

def run_test(training_config):
    pass	

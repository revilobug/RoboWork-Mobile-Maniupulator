#!/usr/bin/env python

import rospy
import gym
import numpy as np
import datetime
import sys
sys.path.insert(0, '/home/rwl/autonomous_mobile_manipulation_ws/src/ros-gym-mobile-manipulation/scripts/robowork_envs/task_envs')

import ppo_runner
import bvrSIM_mobile_grasp

def main():
    # get the training/testing configuration
    training_config = rospy.get_param("/bvr_SIM/mobile_manipulation_rl/training")
    is_training = training_config["is_training"]
    
    # run training
    if is_training:
	rospy.loginfo("Starting Training...")
    	ppo_runner.run_training(training_config)
	rospy.loginfo("Training completed.") 

if __name__ == '__main__':
    rospy.init_node('mobile_manipulation', anonymous=True, disable_signals=True)
    main()

"""log_level=rospy.DEBUG,"""

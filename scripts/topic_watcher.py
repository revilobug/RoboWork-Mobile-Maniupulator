#!/usr/bin/env python

import rospy
import numpy as np 
import random
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


"""
TopicWatcher is responsible for catching any message published to either joint limit or singularity topic.
Sends a boolean variable back to the running pipeline.
"""


class TopicWatcher(object):

    def __init__(self):
	self._jointlimit_recorder = rospy.Publisher('/jointlimit_register', Bool, queue_size=1, latch=True)
	self._singularity_recorder = rospy.Publisher('/singularity_register', Bool, queue_size=1, latch=True)

    def run(self):
	limit_message = Bool()
	singularity_message = Bool()
	rate = rospy.Rate(5)
        while not rospy.is_shutdown():
	    rospy.logdebug('Joint limit service launched')
            data = None
	    singularity = None
	    reached_limits = False
	    reached_singularity = False
	    trigger = True
	    
            while trigger:
		try:
	            # rospy.logdebug("Waiting for a message to come up....")
	            try:
                        data = rospy.wait_for_message("/bvr_SIM/main_armcompliance_controller/jointlimit", JointState, timeout=0.05)
		        singularity = rospy.wait_for_message("/bvr_SIM/main_armcompliance_controller/singularity", JointState, timeout=0.05)
	            except:
		        pass

	            if data is None:
		        reached_limits = False
			limit_message.data = False
	            else:
	                reached_limits = True

	            if singularity is None:
		        reached_singularity = False
			singularity_message.data = False
	            else:
	                reached_singularity = True

	    	    if reached_limits:
	                rospy.loginfo('Got new message\nReached_limits = {}'.format(reached_limits))
		        rospy.loginfo('Publishing to the message topic')
		        limit_message.data = True
		        self._jointlimit_recorder.publish(limit_message)
		        data = None
		        reached_limits = False

		    if reached_singularity:
	                rospy.loginfo('Got new message\nReached_singularity = {}'.format(reached_limits))
		        rospy.loginfo('Publishing to the message topic')
		        singularity_message.data = True
		        self._singularity_recorder.publish(singularity_message)
		        singularity = None
		        reached_singularity = False

		    rate.sleep()
		except KeyboardInterrupt:
		    trigger = False
		
if __name__ == "__main__":
    random.seed()
    rospy.init_node('topic_watcher', anonymous=True)
    subscriber = TopicWatcher()
    subscriber.run()


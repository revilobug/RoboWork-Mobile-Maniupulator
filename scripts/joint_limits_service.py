#!/usr/bin/env python
import rospkg
rospack = rospkg.RosPack()
p = rospack.get_path('robowork_control')
p = p + '/scripts'

import sys
sys.path.append(p)

import home_ur5e_SIM

import rospy
import numpy as np 
import random
import actionlib
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerRequest
from trajectory_msgs.msg import *
from control_msgs.msg import *

hosting_robot_prefix = "bvr_SIM/"
ur5_e_robot_prefix = "main_arm_SIM/"
ur5e_arm_prefix = hosting_robot_prefix + ur5_e_robot_prefix
ur5e_arm_controller_topic = ur5_e_robot_prefix + "compliance_controller/follow_joint_trajectory"


"""
Joint Limit Service is responsible for catching 
"""


class JointLimitService(object):

    def __init__(self):
	rospy.Subscriber("/bvr_SIM/main_armcompliance_controller/singularity", JointState, self._singularity_callback)
	# rospy.Subscriber("/bvr_SIM/main_armcompliance_controller/jointlimit", JointState, self._jointlimit_callback) 
	self.client = actionlib.SimpleActionClient(ur5e_arm_controller_topic, FollowJointTrajectoryAction)
	

    def _singularity_callback(self, data):
        """
        Catches the singularity warning message and saves it
	:param data:
        :return:
        """
	rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.name)
	self.singularity = data
	# rospy.logdebug("Data saved: ")
	# rospy.loginfo(self.get_singularity)
 
    def _jointlimit_callback(self, data):
	"""
        Catches the joint limit warning message and saves it
	:param data:
        :return:
        """
	# rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.name)
	self.joint_limit = data.position
	rospy.logdebug("Data saved: ")
	rospy.loginfo(self.joint_limit)
    
    def get_singularity(self):
        """
        Returns the JointState message published to singularity topic
        :return: singularity
        """
        return self.singularity

    def get_jointlimit(self):
        """
        Returns the JointState message published to joint limit topic
        :return: singularity
        """
        return self.joint_limit

    def trigger_request(self):
	rospy.wait_for_service('/bvr_SIM/bvr_SIM/wrench_to_joint_vel_pub/toggle_joint_limit_margin')
	sos_service = rospy.ServiceProxy('/bvr_SIM/bvr_SIM/wrench_to_joint_vel_pub/toggle_joint_limit_margin', Trigger)
	sos = TriggerRequest()
	result = sos_service(sos)

	rospy.loginfo(result)

    def move():
	g = FollowJointTrajectoryGoal()
	g.trajectory = JointTrajectory()
	g.trajectory.joint_names = JOINT_NAMES
	g.trajectory.points = [JointTrajectoryPoint(positions=Qhome, velocities=[0]*6, time_from_start=rospy.Duration(slowdown*3.0))]
	self.client.send_goal(g)
	try:
	    self.client.wait_for_result()
	except KeyboardInterrupt:
	    self.client.cancel_goal()
	    raise

    def run(self):
        while not rospy.is_shutdown():
            data = None
	    reached_limits = False

            while data is None:
                try:
		    rospy.loginfo("Waiting for a message to come up....")
                    data = rospy.wait_for_message("/bvr_SIM/main_armcompliance_controller/jointlimit", JointState, timeout=10)
		    singularity = rospy.wait_for_message("/bvr_SIM/main_armcompliance_controller/singularity", JointState, timeout=10)
                except:
                    pass

	    reached_limits = True

	    if reached_limits:
	        rospy.loginfo("\nGot new message\nReached_limits = %s", reached_limits)
	        rospy.loginfo(data)
	        self.trigger_request()

	        try:
        	    print("Waiting for ur5_e_arm server...")
        	    self.client.wait_for_server()
        	    print("Connected to ur5_e_arm server")
        	    self.move()
    		except KeyboardInterrupt:
        	    rospy.signal_shutdown("KeyboardInterrupt")
       		    raise

	        self.trigger_request()
	    
	    rospy.sleep(3)
	    data = None
	    reached_limits = False


if __name__ == "__main__":
    # global client
    random.seed()
    rospy.init_node('limit_subscriber', anonymous=True, disable_signals=False)
    subscriber = JointLimitService()
    subscriber.run()



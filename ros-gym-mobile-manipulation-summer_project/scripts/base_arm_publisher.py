#!/usr/bin/env python
import os
import rospy
import numpy as np 
import random
import math
import actionlib
import subprocess

from geometry_msgs.msg import Twist 
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerRequest
from control_msgs.msg import *
from trajectory_msgs.msg import *

hosting_robot_prefix = "/bvr_SIM/"
ur5_e_robot_prefix = "main_arm_SIM/"
ur5e_arm_prefix = hosting_robot_prefix + ur5_e_robot_prefix
ur5e_arm_controller_topic = ur5_e_robot_prefix + "compliance_controller/follow_joint_trajectory"

JOINT_NAMES = [ur5e_arm_prefix+'shoulder_pan_joint',
               ur5e_arm_prefix+'shoulder_lift_joint',
               ur5e_arm_prefix+'elbow_joint',
               ur5e_arm_prefix+'wrist_1_joint',
               ur5e_arm_prefix+'wrist_2_joint',
               ur5e_arm_prefix+'wrist_3_joint']

shoulder_pan_home = 0.0 #1.0*math.pi/2
shoulder_lift_home = -1.5*math.pi/6
elbow_home = -4.5*math.pi/6
wrist_1_home = -0.15*math.pi/2
wrist_2_home = 1.0*math.pi/2
wrist_3_home = 0.0*math.pi/2

Qhome = [shoulder_pan_home                    ,shoulder_lift_home                     ,elbow_home                     ,wrist_1_home                    ,wrist_2_home                    ,wrist_3_home]
slowdown = 1.0

global client
client = None

class JointPublisher(object):

    def __init__(self):
        self._base_publisher = rospy.Publisher("/bvr_SIM/bvr_SIM_velocity_controller/cmd_vel", Twist, queue_size=10)
        self._arm_publisher = rospy.Publisher("/bvr_SIM/interactive_markers_main_arm_SIM/main_arm_SIM_compliance/cmd_vel", Twist, queue_size=10)
	self.is_stuck = False
	rospy.Subscriber("/jointlimit_register", Bool, self._joint_limit_callback)
	rospy.Subscriber("/singularity_register", Bool, self._singularity_callback)
	#self.client = actionlib.SimpleActionClient(ur5e_arm_controller_topic, FollowJointTrajectoryAction)

    def publish_base_velocity(self):

	base_vel = Twist()

	# here convert velocity output of the NN to Twist velocities accordingly

	"""
        print("Testing base movement. Move your robot")
	linear_forward = input("Move forward: 1, Move backwards: 0 ")
	angular_forward = input("Move left: 1, Move right: 0 ")
	linear_speed = input("Linear speed. Please enter a value between 0 and 1 ")
	angular_speed = input("Angular speed. Please enter a value between 0 and 1 ")

	if(linear_forward):
	    base_vel.linear.x = abs(linear_speed)
	else:
	    base_vel.linear.x = -abs(linear_speed)
	if(angular_forward):
	    base_vel.angular.z = abs(angular_speed)
	else:
	    base_vel.angular.z = -abs(angular_speed) 
        """
	base_vel.linear.x = random.uniform(-1, 1)
	base_vel.linear.y = 0
	base_vel.linear.z = 0

	base_vel.angular.x = 0
        base_vel.angular.y = 0
	base_vel.angular.z = random.uniform(-1, 1)

	# publish base action
	self._base_publisher.publish(base_vel)

    def publish_arm_velocity(self):
	arm_vel = Twist()

	# here convert velocity output of the NN to Twist velocities accordingly
	arm_vel.linear.x = 0.2
	arm_vel.linear.y = 0.2
	arm_vel.linear.z = 0.2

        arm_vel.angular.x = random.uniform(0, 1)
        arm_vel.angular.y = random.uniform(0, 1)
        arm_vel.angular.z = random.uniform(0, 1)

	# publish arm action	
	self._arm_publisher.publish(arm_vel)  

    def reset_velocities(self): 

	base_vel = Twist()
	base_vel.linear.x = 0
	base_vel.linear.y = 0
	base_vel.linear.z = 0

	base_vel.angular.x = 0
	base_vel.angular.y = 0
	base_vel.angular.z = 0

	self._base_publisher.publish(base_vel)

	arm_vel = Twist()  
	arm_vel.linear.x = 0
	arm_vel.linear.y = 0
	arm_vel.linear.z = 0

        arm_vel.angular.x = 0
        arm_vel.angular.y = 0
        arm_vel.angular.z = 0	

	self._arm_publisher.publish(arm_vel)    

    def _joint_limit_callback(self, data):
	self.is_stuck = data.data

    def check_joint_limit(self):
	rospy.loginfo(self.is_stuck)
	if self.is_stuck:
	    rospy.logdebug('the robot is stuck... handling the situation')
	    return True
	else:
	    return False

    def _singularity_callback(self, data):
	self.singularity = data.data

    def check_singularity(self):
	rospy.loginfo(self.singularity)
	if self.singularity:
	    rospy.logdebug('the robot reached singularity... handling the situation')
	    return True
	else:
	    return False

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
    client.send_goal(g)
    try:
	client.wait_for_result()
    except KeyboardInterrupt:
	client.cancel_goal()
	raise

    
if __name__ == "__main__":

    random.seed()
    rospy.init_node('arm_and_base_publisher_node', anonymous=True, log_level=rospy.DEBUG)
    arm_and_base_publisher = JointPublisher()
    rate_value = 5   
    rate = rospy.Rate(rate_value)

    while not rospy.is_shutdown():
        rospy.loginfo("Publishing an action")
	# publish velocities
	arm_and_base_publisher.publish_base_velocity()
        arm_and_base_publisher.publish_arm_velocity()
	# check for joint limit
	if arm_and_base_publisher.check_joint_limit():
	    # pass zero velocities
	    arm_and_base_publisher.reset_velocities()
	    #trigger joint limit margin
	    arm_and_base_publisher.trigger_request()

            # here reset the arm
	    # not connecting to the server, i believe it's a namespace issue
	    try:
		cmd = ["rosrun","robowork_control","home_ur5e_SIM.py", 'ROS_NAMESPACE="bvr_SIM"']
		proc = subprocess.Popen(cmd)
		#time.sleep(1)  # maybe needed to wait the process to do something useful
		proc.terminate()
    	    except KeyboardInterrupt:
        	rospy.signal_shutdown("KeyboardInterrupt")
       		raise

	    # toggle back
	    arm_and_base_publisher.trigger_request()
        # sleep till the rest of rate
	rate.sleep()



#!/usr/bin/env python
"""
0.1 - started implementing Robot Environment
0.2 - cleaned up the code 
"""

import rospy 
import numpy as np
import random
import math
import time
import actionlib
import tf2_geometry_msgs
import tf2_ros
import geometry_msgs.msg as geo_msgs
import tf
from openai_ros import robot_gazebo_env
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest
from control_msgs.msg import *
from trajectory_msgs.msg import *
from actionlib_msgs.msg import *
import threading


'''
Arm control variables.
'''
hosting_robot_prefix = "bvr_SIM/"
ur5_e_robot_prefix = "main_arm_SIM/"
ur5e_arm_prefix = hosting_robot_prefix + ur5_e_robot_prefix
ur5e_arm_controller_topic = '/'+hosting_robot_prefix + ur5_e_robot_prefix + "compliance_controller/follow_joint_trajectory"

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


'''
Class BvrSIMEnv - Robot Environment contains the functionalities that the robot needs in order to be controlled. 
'''

class BvrSIMEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        rospy.logdebug("Entered BvrSIMEnv")

        # transform for ee pose
        self.tf_transformer = tf.TransformListener()

        self.controllers_list = []

        self._robot_env_params()

        super(BvrSIMEnv, self).__init__(robot_name_space = self._robot_namespace, controllers_list = self.controllers_list, reset_controls = False, start_init_physics_parameters=True, reset_world_or_sim="NO_RESET_SIM")
        self.seed()
        
        # Publishers for velocity commands
        self._base_publisher = rospy.Publisher("/bvr_SIM/bvr_SIM_velocity_controller/cmd_vel", Twist, queue_size=1)
        self._arm_publisher = rospy.Publisher("/bvr_SIM/interactive_markers_main_arm_SIM/main_arm_SIM_compliance/cmd_vel", Twist, queue_size=1)
        # self._check_publishers_connection()

        # Subscriber to joint limits and singularity watcher	
	rospy.Subscriber("/bvr_SIM/main_armcompliance_controller/jointlimit", JointState, self._joint_limit_callback)
	rospy.Subscriber("/bvr_SIM/main_armcompliance_controller/singularity", JointState, self._singularity_callback)

        # observation space
        # PoseStamped.pose contains of position and quaternion orientation (need to convert to euler)
        rospy.Subscriber("/bvr_SIM/bvr_SIM_odom_message_to_tf/pose", geo_msgs.PoseStamped, self._base_pose_callback)
        rospy.Subscriber("/bvr_SIM/joint_states", JointState, self._joint_state_callback)

        # connection to server that resets the arm
	self.client = actionlib.SimpleActionClient(ur5e_arm_controller_topic, FollowJointTrajectoryAction)
	rospy.Subscriber("/bvr_SIM/main_arm_SIM/compliance_controller/follow_joint_trajectory/result", FollowJointTrajectoryActionResult, self._goal_callback)


        rospy.logdebug("BvrSIMEnv initialized")

    def _robot_env_params(self):
        self._robot_namespace = rospy.get_param("/bvr_SIM/mobile_manipulation_rl/robot_namespace")
        self._arm_namespace = rospy.get_param("/bvr_SIM/mobile_manipulation_rl/arm_namespace")

        self.gripper_frame = "bvr_SIM/main_arm_SIM/gripper_manipulation_link"
        self.world_frame = "map"
        self.base_frame = "bvr_SIM/bvr_base_link"

        self.goalPoseTimber = np.array([2.970025, 0.108175, 0.303864])
        self.goalPoseCinder = np.array([3.07991, 0.107552, 0.049095])

        self._base_pose = geo_msgs.PoseStamped()
        self._joint_states = JointState()
	self._action_result = FollowJointTrajectoryActionResult()
        self._transform_data = tuple()

	self._is_stuck = None
        self._singularity = None
	# assume it's correctly reset
	self._status = 3
	self._result = 0

        self.update_rate = 5 # maybe change to get_param from the server 
        return	

    """
     
    Publisher/Subscriber methods 

    """

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()
		
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(5) # 10hz
        while self._base_publisher.get_num_connections() == 0 and self._arm_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to base and arm yet so we wait and try again")
            rospy.logdebug(self._base_publisher.get_num_connections())
            rospy.logdebug(self._arm_publisher.get_num_connections())
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("Base and Arm Publisher Ready")	

    def _base_pose_callback(self, data):
        self._base_pose = data
    
    def get_base_pose(self):
        return self._base_pose

    def _joint_state_callback(self, data):
        self._joint_states = data
        return

    def get_joint_states(self):
        return self._joint_states

    def _joint_limit_callback(self, data):
	self._is_stuck = data

    def check_joint_limit(self):
	if self._is_stuck:
	    rospy.loginfo('the robot is stuck... handling the situation')
	    return True
	else:
	    return False

    def _singularity_callback(self, data):
	self._singularity = data

    def check_singularity(self):
	if self._singularity:
            rospy.loginfo('the robot reached singularity... handling the situation')
	    return True
	else:
	    return False

    def _goal_callback(self, data):
	self._status = data.status.status
	self._result = data.result.error_code
	#rospy.loginfo("goal status int: {}".format(self._status))
	#rospy.loginfo("goal result int: {}".format(self._result))
   
    def get_goal_data(self):
	return self._status, self._result

    def _store_transform_callback(self):
        # keep storing tf transformations
        # rospy.logdebug("entered transform callback")
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_transformer.lookupTransform(self.world_frame, self.gripper_frame, rospy.Time(0))
                # rospy.loginfo((trans, rot))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

            self._transform_data = (trans, rot)

    def get_latest_transform_data(self):
        return self._transform_data

    """

    Methods for moving the robot.

    """

    def move_base(self, velocity):
        """
            Converts the velocity to Twist message and publish to the base topic
            :param velocity: Numpy Array. holding 8 velocities. velocity[0:2] - base velocities, velocity[2:] - arm velocities
            :return:
        """
        base_vel = Twist()
        # here convert velocity output of the NN to Twist velocities accordingly
        
        base_vel.linear.x = velocity[0]
        base_vel.angular.z = velocity[1]
       # rospy.logdebug("BvrSIM Base Twist Command>>" + str(base_vel))
        
        # publish base action
        self._base_publisher.publish(base_vel)

    def move_arm(self, velocity):
        """
	    Converts the velocity to Twist message and publish to the arm topic
	    :param velocity: Numpy Array. holding 8 velocities. velocity[0:2] - base velocities, velocity[2:] - arm velocities
	    :return:
        """
        arm_vel = Twist()
        
        # here convert velocity output of the NN to Twist velocities accordingly
        
        arm_vel.linear.x = velocity[2]
        arm_vel.linear.y = velocity[3]
        arm_vel.linear.z = velocity[4]	

        arm_vel.angular.x = velocity[5]
        arm_vel.angular.y = velocity[6]
        arm_vel.angular.z = velocity[7]
        # rospy.logdebug("BvrSIM Arm Twist Command>>" + str(arm_vel))
	
        # publish arm action
        self._arm_publisher.publish(arm_vel)
    
    def move_arm_home(self):
        """
            Moves arm home using actionlib client. Used after each episode. make sure to reset velocities before calling this function.
            :return:
        """
	goal_finished = False
        try:
            rospy.loginfo("Waiting for ur5_e_arm server...")
            wait_for_srv = self.client.wait_for_server()
            rospy.loginfo("Connected to ur5_e_arm server success: {}".format(wait_for_srv))      
            g = FollowJointTrajectoryGoal()
	    g.trajectory = JointTrajectory()
	    g.trajectory.joint_names = JOINT_NAMES
	    g.trajectory.points = [JointTrajectoryPoint(positions=Qhome, velocities=[0]*6, time_from_start=rospy.Duration(slowdown*3.0))]
	    self.client.send_goal(g)
	    try:
		#rospy.loginfo("entered try in move arm home, measuring time")
		start = time.time()
	        goal_finished = self.client.wait_for_result(timeout=rospy.Duration(10.0))
		rospy.sleep(2)
		end = time.time()
		rospy.loginfo("wait for result: {} finished in {}".format(goal_finished, (end-start)))
		status, error = self.get_goal_data()
		rospy.loginfo("goal status: {} and error code: {}".format(status, error))
		timeout = time.time() + 5*60
		
		while status is not 3:
		    rospy.loginfo("trying to reset arm again...")
		    if time.time() > timeout:
			break
		    try:
			rospy.loginfo("Waiting for ur5_e_arm server...")
			wait_for_srv = self.client.wait_for_server()
			rospy.loginfo("Connected to ur5_e_arm server success: {}".format(wait_for_srv))      
			self.client.send_goal(g)
			rospy.sleep(1)
			goal_finished = self.client.wait_for_result(timeout=rospy.Duration(5.0))
			rospy.sleep(2)
			status, error = self.get_goal_data()
			rospy.loginfo("goal status: {} and error code: {}".format(status, error))
		    except KeyboardInterrupt:
			self.client.cancel_goal()
			raise
	    except KeyboardInterrupt:
	        self.client.cancel_goal()
	        raise
	except KeyboardInterrupt:
            rospy.signal_shutdown("KeyboardInterrupt")
            raise
	return goal_finished
	

    def move_combined(self, velocity):
        """
            Runs move base and arm in rate 5Hz
            :param velocity: Numpy Array. holding 8 velocities. velocity[0:2] - base velocities, velocity[2:] - arm velocities
            :return:
        """
        rate = rospy.Rate(self.update_rate)
        # self._check_publishers_connection()

	# clip the velocities to [-1, 1] range
	"""
	for i, vel in enumerate(velocity):
  	    if vel < -1.0:
    		velocity[i] = -1.0
 	    if vel > 1.0:
    		velocity[i] = 1.0
	"""        
        if not rospy.is_shutdown():	
            # rospy.loginfo("Publishing an action: " + str(velocity))	
	    try:
                self.move_base(velocity)
                self.move_arm(velocity)
                rate.sleep()
	    except:
		rospy.loginfo("Publishing an action failed ")

    def reset_velocities(self): 
	"""
	    Resets the velocity command to zero.
	    :param:
	    :return:
        """
	rate = rospy.Rate(self.update_rate)
        # self._check_publishers_connection()

	base_vel = Twist()
	base_vel.linear.x = 0
	base_vel.linear.y = 0
	base_vel.linear.z = 0

	base_vel.angular.x = 0
	base_vel.angular.y = 0
	base_vel.angular.z = 0

	arm_vel = Twist()  
	arm_vel.linear.x = 0
	arm_vel.linear.y = 0
	arm_vel.linear.z = 0

        arm_vel.angular.x = 0
        arm_vel.angular.y = 0
        arm_vel.angular.z = 0

        if not rospy.is_shutdown():	
            # rospy.loginfo("Publishing an action: " + str(velocity))	
	    try:
		#rospy.loginfo("Setting 0 action")
                self._base_publisher.publish(base_vel)
		self._arm_publisher.publish(arm_vel)
                rate.sleep()
	    except:
		rospy.loginfo("resetting velocities failed ")
	
	
    def trigger_request(self):
	"""
	Triggers joint limit margin. Used in case resetting the arm when the arm reaches joint limit.
	"""
	rospy.wait_for_service('/bvr_SIM/bvr_SIM/wrench_to_joint_vel_pub/toggle_joint_limit_margin')
	sos_service = rospy.ServiceProxy('/bvr_SIM/bvr_SIM/wrench_to_joint_vel_pub/toggle_joint_limit_margin', Trigger)
	sos = TriggerRequest()
	result = sos_service(sos)
	rospy.loginfo(result)
	rospy.sleep(0.1)

    def reset_arm(self):
	"""
	resets the arm to home position. Used in case of reaching singularity or inability to move.
	"""
	moving_done = False
	while not rospy.is_shutdown():
	    try:
	    	self.trigger_request()
		#rospy.loginfo("Moving arm home")
		try:
	    	    moving_done = self.move_arm_home()
		except:
		    rospy.loginfo("failed to move arm home")
		rospy.loginfo("setting is_stuck and singularity to None")
		self._is_stuck = None
		self._singularity = None
	    	self.trigger_request()
	    except rospy.ROSException as e:
                rospy.logerr("Failed to reset the arm.")
                raise e	 
	    if moving_done:
		# rospy.loginfo("breaking the reset_arm")
		break  
	    
    def move_arm_after_finished_episode(self):
	moving_done = False
	while not rospy.is_shutdown():
	    try:
	    	moving_done = self.move_arm_home()
	    except rospy.ROSException as e:
                rospy.logerr("Failed to reset the arm.")
                raise e	 
	    if moving_done:
		#rospy.loginfo("move_arm_after_finished_episode")
		break  
	
    """

    Methods needed in running Agent that need to be overridden in Task Environment

    """

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()



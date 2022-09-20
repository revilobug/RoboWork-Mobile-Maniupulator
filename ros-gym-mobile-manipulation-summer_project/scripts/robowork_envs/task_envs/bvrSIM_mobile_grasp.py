import rospy
from gym.envs import register
from gym import spaces
import numpy as np
import math
import random
import geometry_msgs.msg as geo_msgs
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from tf.transformations import quaternion_multiply
import tf

import sys
sys.path.insert(0, '/home/rwl/autonomous_mobile_manipulation_ws/src/ros-gym-mobile-manipulation/scripts/robowork_envs/robot_envs')

from bvrSIM_env import BvrSIMEnv

timestep_limit_per_episode = 10000 

reg = register(
        id='BvrSIMMobileGrasp-v0',
        entry_point='bvrSIM_mobile_grasp:BvrSIMMobileGraspEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

"""
Class BvrSIMEnv - Task Environment contains the functionalities that the robot needs in order to be controlled.
"""
class BvrSIMMobileGraspEnv(BvrSIMEnv):
    def __init__(self):
        rospy.logdebug("Initializing BvrSIMMobileGraspEnv")

        # call super constructor
        super(BvrSIMMobileGraspEnv, self).__init__()

	# read params
        self._set_params()

        # action space: array of size (8,)
        # [0:2] base velocities: (x, yaw)
        # [2:] joint velocities: (x, y, z, roll, pitch, yaw) 
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # observation space: array of size (23,)
        # [0:3] base state (x, y, yaw) w.r.t. goal
        # [3:9] arm joint angles(6)
        # [9:17] velocities (8) from previous step	
	# [17:23] ee state (x, y, z, roll, pitch, yaw) w.r.t. goal
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32) 

        self._goal_object = None
        self._set_goal()

        rospy.logdebug("MobileGraspEnv initialized")

    def _set_params(self):
	"""
	Define parameters for the environment.
        """
        param_prefix = "/bvr_SIM/mobile_manipulation_rl"
        #self._sleep_on_action = rospy.get_param(param_prefix + "/simulation/sleep_on_action")
        self._cinder_pose = rospy.get_param(param_prefix + "/simulation/init_cinder_pose")
        self._timber_pose = rospy.get_param(param_prefix + "/simulation/init_timber_pose")

        arm_joints = rospy.get_param(param_prefix + "/main_arm_joints")
        self._arm_joint_names = ["bvr_SIM/main_arm_SIM/" + joint for joint in arm_joints]

        goal_params = rospy.get_param(param_prefix + "/goal")
        self._goal_position_tolerance = goal_params["position_tolerance"]
        self._goal_orientation_tolerance = goal_params["orientation_tolerance"]
        self._goal_completion_reward = goal_params["completion_reward"]
	self._max_distance_from_goal = 5

        return

    def _set_goal(self):
        # goal pose is the position of the timber board
        self._goal_object = self.gazebo.get_model_state("timber_board")
        return

    """
    Overridden Methods
    """

    def _set_init_pose(self):
        """
    	Sets the Robot in its init pose
        """

        # get position of base
    	base_pose, _ = self.getBaseposition()
        base_x = base_pose[0]
        base_y = base_pose[1]

        # determine random distance from base using polar coordinates
	distance = self._max_distance_from_goal
        # distance = random.randrange(self._max_distance_from_goal - 2) + 1
        theta = random.random() * np.pi

        # convert polar to cartesian relative to map
        self.goal_x = distance * np.cos(theta) + base_x
        self.goal_y = distance * np.sin(theta) + base_y

        # set goal pose
        self._set_goal_pose(theta, self.goal_x, self.goal_y)
        return 

    def _get_obs(self):   # move this to task env
        """
	    Returns the observation.
        """ 
        # base position and orientation
        basePose, baseEuler = self.getBaseposition()

        baseYaw = np.array([baseEuler[2]])

        self.gazebo.unpause()
        # End-Effector position and orientation
        eePose, eeEulers = self.getEEposition()
        self.gazebo.pause()

        # joint angles and velocity
        jointAngleObs = self.getJointObs()

        # goal relative positions
        timberBasePose, timberEEPose, timberBaseEuler, timberEEEuler = self.getGoalPose()
	"""
        rospy.logdebug(basePose.shape)
        rospy.logdebug(baseYaw.shape)
        rospy.logdebug(eePose.shape)
        rospy.logdebug(eeEulers.shape)
        rospy.logdebug(jointAngleObs.shape)
        rospy.logdebug(timberBasePose.shape)
        rospy.logdebug(timberEEPose.shape)
        rospy.logdebug(timberBaseEuler.shape)
        rospy.logdebug(timberEEEuler.shape)
	"""
        obs = np.concatenate((basePose, baseYaw, eePose, eeEulers, jointAngleObs, timberBasePose, timberEEPose, timberBaseEuler, timberEEEuler))
	# rospy.logdebug("observation {}".format(obs))
        return obs

    def reset(self):
        """
        Resets the simulation after each episode. Returns initial observation. 
        In this case resetting the simulation is:
        1. Changing the goal object pose to random
        2. Resetting arm to home postition
        :return: obs 
        """
        rospy.logdebug("Reseting BvrSIMEnvironment")
        self.gazebo.unpauseSim()
	self.reset_velocities()

        if self.check_joint_limit() or self.check_singularity():
	    print("resetting arm")
            self.reset_arm()
        else:      
	    print("moving arm home after finished episode")
            self.move_arm_after_finished_episode()

        self.gazebo.pauseSim()
        self._set_init_pose()  
        self._init_env_variables()
        # unclear? 
        # self._update_episode()
        obs = self._get_obs()

        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs


    def _set_action(self, action):
        """
        Applies the given action to the simulation. 
        Calls the methods that publish the velocity command.
        Saves last velocity action that can be reused in NN.
        :param action: numpy array of size 8
        """
        # check for sing joint lim?
        rospy.logdebug("Velocity Action: " + str(action))
        self.move_combined(action)


    def _is_done(self, observations):
        """
        Indicates whether or not the episode is done.
        1) bvrSIM has reached the goal position 
        2) bvrSIM entered singularity
        3) bvrSIM is unable to move 	
        """

	# check if action execution failed
        done = {"is_done": False, "is_success": False}

        # check if goal reached        
        # euclidean distance between ee pose and goal object pose
        distance = self._get_ee_dist_to_goal()

        _, _, q_r, _ = self.getGoalPose()

        # q_euler = euler_from_quaternion(q_r)

        distance_close_enough = distance <= self._goal_position_tolerance
        orientation_close_enough = all(euler <= self._goal_orientation_tolerance for euler in q_r)

        if distance_close_enough and orientation_close_enough:
            rospy.logdebug("Goal achieved")
            done["is_done"] = True
            done["is_success"] = True

        if self.check_joint_limit() or self.check_singularity():
            rospy.loginfo("Episode failed to be completed")
            done["is_done"] = True
            done["is_success"] = False

        return done

    def _compute_reward(self, observations, done):
        """
	    Calculates the reward to give based on the observations given.
        """
	
        dist = self._get_ee_dist_to_goal()
	# rospy.loginfo("distance from the goal: {}".format(dist))

        reward = 0.5*(-dist) + math.exp(-(dist ** 2)) 

        return reward

    def _init_env_variables(self):
        """
        Called on simulatin reset. Set the goal reference point for the future use.
        Ex: set robot goal
        """
        self._set_goal()

    def _get_ee_dist_to_goal(self):
        ee_pose, _ = self.getEEposition()
        _, eeGoalDist, _, _ = self.getGoalPose()

        # euclidean distance between ee pose and goal object pose
        distance = np.linalg.norm(eeGoalDist)

        return distance

    def _set_goal_pose(self, anglerotation, timber_x, timber_y):
        # cartesian difference between timber and cinder
        x_diff = self.goalPoseCinder[0] - self.goalPoseTimber[0]
        y_diff = self.goalPoseCinder[1] - self.goalPoseTimber[1]
        # take into account random orientation of goal
        goalX_shift = np.cos(anglerotation) * x_diff - np.sin(anglerotation) * y_diff
        goalY_shift = np.sin(anglerotation) * x_diff + np.cos(anglerotation) * y_diff

        # find quaternion rotation
        goalOrientationTimber = list(quaternion_from_euler(0,0, anglerotation))
        goalOrientationCinder = list(quaternion_from_euler(0,0, anglerotation))

        # set timber and cinder in right position
        self.gazebo.pauseSim()

        self.gazebo.set_model_state("timber_board", timber_x, timber_y, 0.303864, goalOrientationTimber[0],goalOrientationTimber[1],goalOrientationTimber[2], goalOrientationTimber[3])
        self.gazebo.set_model_state("cinder_block", timber_x + goalX_shift, timber_y + goalY_shift, 0, goalOrientationCinder[0],goalOrientationCinder[1], goalOrientationCinder[2], goalOrientationCinder[3])

        self.gazebo.unpauseSim()
        return  


    def getEEposition(self):  
        
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_transformer.lookupTransform(self.world_frame, self.gripper_frame, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        # convert quaternion to roll, pitch, yaw
        eeEulers = list(euler_from_quaternion(rot))

        # return observation space
	end_effector_position = np.array(trans), np.array(eeEulers)
        return end_effector_position

    # returns (x, y), (yaw) of base
    def getBaseposition(self):
        base_message = self.get_base_pose()

        # (x, y)
        basePose = [base_message.pose.position.x, 
                    base_message.pose.position.y]

        # base rotation in quaterninon
        baseRot = [base_message.pose.orientation.w,
                base_message.pose.orientation.y,
                base_message.pose.orientation.x,
                base_message.pose.orientation.z]

        # convert quaternion to yaw
        baseEulers = list(euler_from_quaternion(baseRot))

        #return observation
        return np.array(basePose), np.array(baseEulers)

    # return angle and velocity of
    # 3 elbow_joint
    # 5 shoulder_lift_joint
    # 6 shoulder_pan_joint
    # 7 wrist_1_joint
    # 8 wrist_2_joint
    # 9 wrist_3_joint
    def getJointObs(self):
        # get raw position
        arm_message = self.get_joint_states()

        # break into joint angle and velocity
        joint_angles = list(arm_message.position)
        joint_velocity = list(arm_message.velocity)

        # angles
        jointObs = joint_angles[4:9]
        jointObs.insert(0, joint_angles[2])

        # velocity
        jointObs.append(joint_velocity[2])
        jointObs = jointObs + joint_velocity[4:9]
        
        # return observation
        return np.array(jointObs)

    # return (x, y, z) and (roll, pitch, yaw) of goal timber
    def getGoalPoseHelper(self):
        # obtain location from Gazebo
        goalObject = self.gazebo.get_model_state("timber_board")

        #(x, y, z)
        goalPose = [goalObject.pose.position.x,
                    goalObject.pose.position.y,
                    goalObject.pose.position.z]

        # (roll, pitch, yaw)
        goalQuat = [goalObject.pose.orientation.w,
                        goalObject.pose.orientation.x,
                        goalObject.pose.orientation.y,
                        goalObject.pose.orientation.z]

        # return observation
        return np.array(goalPose), goalQuat

    # return relative coordinates of goal to EE and Base
    def getGoalPose(self):
        eePose, eeEuler = self.getEEposition()
        basePose, baseEuler = self.getBaseposition()

        # Euler not relative
        goalPose, goalQuat = self.getGoalPoseHelper()

        # q_1 = ee/base Euler
        # q_2 = goal pose
        # rospy.logdebug(eeEuler)
        # rospy.logdebug(baseEuler)
        eeQuat = quaternion_from_euler(eeEuler[0],eeEuler[1],eeEuler[2])
        eeInvs = np.zeros(4)
        eeInvs[0] = eeQuat[1]
        eeInvs[1] = eeQuat[2]
        eeInvs[2] = eeQuat[3]
        eeInvs[3] = (-1) * eeQuat[0]

        baseQuat = quaternion_from_euler(baseEuler[0], baseEuler[1], baseEuler[2])
        baseInvs = np.zeros(4)
        baseInvs[0] = baseQuat[1]
        baseInvs[1] = baseQuat[2]
        baseInvs[2] = baseQuat[3]
        baseInvs[3] = (-1) * baseQuat[0]

        eeRelativeEuler = list(euler_from_quaternion(tf.transformations.quaternion_multiply(goalQuat, eeInvs)))
        baseRelativeEuler = list(euler_from_quaternion(tf.transformations.quaternion_multiply(goalQuat, baseInvs)))

        # get relative coordinates
        timberEEFrame = np.subtract(eePose, goalPose)
        timberBaseFrame = np.subtract(basePose, np.array(goalPose[0], goalPose[1]))

        # return observation space
        return timberBaseFrame, timberEEFrame, np.array(eeRelativeEuler), np.array(baseRelativeEuler)





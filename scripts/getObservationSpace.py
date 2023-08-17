#!/usr/bin/env python

import rospy
import tf
import geometry_msgs.msg as geo_msgs
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import tf2_geometry_msgs
import tf2_ros
from robowork_envs.robot_envs.simple_gazebo_connection import SimpleGazeboConnection
import numpy as np
import time

# returns (x, y, z), (roll, pitch, yaw) of end-effector
def getEEposition():
    # create a transform listener object
    transformer = tf.TransformListener()
    frameSource = 'map'
    frameTarget = 'bvr_SIM/main_arm_SIM/gripper_manipulation_link'

    # transformer.waitForTransform(frameSource, frameTarget, rospy.Time(), rospy.Duration(0))
    # (trans,rot) = transformer.lookupTransform(frameSource, frameTarget, rospy.Time())

    total = 0
    run = 0

    start = time.time()
    while not rospy.is_shutdown() and run < 100000:
        try:
            (trans,rot) = transformer.lookupTransform(frameSource, frameTarget, rospy.Time(0))
            run += 1
        except (tf.LookupException):
            # print ("lookup exception")
            continue
        except (tf.ConnectivityException):
            # print ("connectivity exception")
            continue
        except (tf.ExtrapolationException):
            # print ("extrapolation exception")
            continue

    end = time.time()
    total = end - start
    print (total)
    print (run)
    print (total / run)

    # convert quaternion to roll, pitch, yaw
    # eeEulers = list(euler_from_quaternion(rot))

    # return observation space
    # return np.array(trans), np.array(eeEulers)


# returns (x, y), (yaw) of base
def getBaseposition():
    base_message = rospy.wait_for_message("/bvr_SIM/bvr_SIM_odom_message_to_tf/pose", geo_msgs.PoseStamped)

    # (x, y)
    basePose = [base_message.pose.position.x, 
                base_message.pose.position.y]

    # (yaw)
    baseRot = [base_message.pose.orientation.w,
               base_message.pose.orientation.y,
               base_message.pose.orientation.x,
               base_message.pose.orientation.z]

    # convert quaternion to yaw
    baseEulers = list(euler_from_quaternion(baseRot))

    #return observation
    return np.array(basePose), np.array([baseEulers[2]])

# return angle and velocity of
# 3 elbow_joint
# 5 shoulder_lift_joint
# 6 shoulder_pan_joint
# 7 wrist_1_joint
# 8 wrist_2_joint
# 9 wrist_3_joint
def getJointObs():
    # get raw position
    arm_message = rospy.wait_for_message("/bvr_SIM/joint_states", JointState)

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
def getGoalPoseHelper():
    # obtain location from Gazebo
    gazebo = SimpleGazeboConnection()
    goalObject = gazebo.get_model_state("timber_board")

    #(x, y, z)
    goalPose = [goalObject.pose.position.x,
                goalObject.pose.position.y,
                goalObject.pose.position.z]

    # (roll, pitch, yaw)
    goalEuler = [goalObject.pose.orientation.w,
                       goalObject.pose.orientation.x,
                       goalObject.pose.orientation.y,
                       goalObject.pose.orientation.z]

    goalObject1 = gazebo.get_model_state("cinder_block")

    goalPose1 = [goalObject1.pose.position.x,
                goalObject1.pose.position.y,
                goalObject1.pose.position.z]

    goalEuler1 = [goalObject1.pose.orientation.w,
                       goalObject1.pose.orientation.x,
                       goalObject1.pose.orientation.y,
                       goalObject1.pose.orientation.z]

    gazebo.pauseSim()
    # gazebo.resetSimulation()
    gazebo.set_model_state("cinder_block", goalPose1[0], goalPose1[1], goalPose1[2],
                                 goalEuler1[0], goalEuler1[1], goalEuler1[2], goalEuler1[3])

    gazebo.set_model_state("timber_board", goalPose[0], goalPose[1], goalPose[2],
                                 goalEuler[0], goalEuler[1], goalEuler[2], goalEuler[3])
    gazebo.unpauseSim()

    # return observation
    return np.array(goalPose), np.array(goalEuler)

# return relative coordinates of goal to EE and Base
def getGoalPose():
    eePose, _ = getEEposition()
    basePose, _ = getBaseposition()

    # Euler not relative
    goalPose, goalEuler = getGoalPoseHelper()

    # get relative coordinates
    timberEEFrame = np.subtract(eePose, goalPose)
    timberBaseFrame = np.subtract(eePose, goalPose)

    # return observation space
    return timberBaseFrame, timberEEFrame


def main():
    # create node
    rospy.init_node('getPose', anonymous=True)
    

    gazebo = SimpleGazeboConnection()
    # goalObject = gazebo.get_model_state("timber_board")

    # gazebo.unpauseSim()
    # _, _ = getBaseposition()
    
    
    gazebo.unpauseSim()
    getEEposition()



    # print('.')

    # #(x, y, z)
    # goalPose = [goalObject.pose.position.x,
    #             goalObject.pose.position.y,
    #             goalObject.pose.position.z]

    # # (roll, pitch, yaw)
    # goalEuler = [goalObject.pose.orientation.w,
    #                    goalObject.pose.orientation.x,
    #                    goalObject.pose.orientation.y,
    #                    goalObject.pose.orientation.z]

    # goalObject1 = gazebo.get_model_state("cinder_block")

    # goalPose1 = [goalObject1.pose.position.x,
    #             goalObject1.pose.position.y,
    #             goalObject1.pose.position.z]

    # goalEuler1 = [goalObject1.pose.orientation.w,
    #                    goalObject1.pose.orientation.x,
    #                    goalObject1.pose.orientation.y,
    #                    goalObject1.pose.orientation.z]

    # # constants
    # goalPoseTimber = np.array([2.970025, 0.108175, 0.303864])
    # goalPoseCinder = np.array([3.07991, 0.107552, 0.049095])

    # x_diff = goalPoseCinder[0] - goalPoseTimber[0]
    # y_diff = goalPoseCinder[1] - goalPoseTimber[1]

    # # three inputs
    # anglerotation = 0
    # timber_x = 3
    # timber_y = 1

    # goalX_shift = np.cos(anglerotation) * x_diff - np.sin(anglerotation) * y_diff
    # goalY_shift = np.sin(anglerotation) * x_diff + np.cos(anglerotation) * y_diff

    # goalOrientationTimber = list(quaternion_from_euler(0,0, anglerotation))
    # goalOrientationCinder = list(quaternion_from_euler(0,0, anglerotation))

    # gazebo.pauseSim()

    # gazebo.set_model_state("timber_board", timber_x, timber_y, 0.303864, goalOrientationTimber[0],goalOrientationTimber[1],goalOrientationTimber[2], goalOrientationTimber[3])
    # gazebo.set_model_state("cinder_block", timber_x + goalX_shift, timber_y + goalY_shift, 0, goalOrientationCinder[0],goalOrientationCinder[1], goalOrientationCinder[2], goalOrientationCinder[3])

    # gazebo.unpauseSim()
    # gazebo.resetSimulation()
    # gazebo.pauseSim()
    # gazebo.set_model_state("cinder_block", goalPose1[0], goalPose1[1], goalPose1[2],
    #                              goalEuler1[0], goalEuler1[1], goalEuler1[2],goalEuler1[3])
    # gazebo.unpauseSim()
    # time.sleep(1)
    # gazebo.pauseSim()

    # gazebo.set_model_state("timber_board", goalPose[0], goalPose[1], goalPose[2],
    #                              goalEuler[0], goalEuler[1], goalEuler[2],goalEuler[3])
    # time.sleep(1)
    # gazebo.unpauseSim()

    # start = time.time()

    # # base position and orientation
    # basePose, baseYaw = getBaseposition()

    # # End-Effector position and orientation
    # eePose, eeEulers = getEEposition()

    # # joint angles and velocity
    # jointAngleObs = getJointObs()

    # # goal relative positions
    # timberBasePose, timberEEPose = getGoalPose()

    # obs = np.concatenate((basePose, baseYaw, eePose, eeEulers, jointAngleObs, timberBasePose, timberEEPose))


    # print(end - start)

    # print (obs)


if __name__ == '__main__':
    main()

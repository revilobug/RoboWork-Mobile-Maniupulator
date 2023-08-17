#!/usr/bin/env python

import rospy
import tf
import geometry_msgs.msg as geo_msgs
import numpy as np

def getEEposition():
    # create a transform listener object
    transformer = tf.TransformListener()
    frameSource = 'map'
    frameTarget = 'bvr_SIM/main_arm_SIM/gripper_manipulation_link'

    while not rospy.is_shutdown():
        try:
            (trans,rot) = transformer.lookupTransform(frameSource, frameTarget, rospy.Time())
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

def getBaseposition():
    base_message = rospy.wait_for_message("/bvr_SIM/bvr_SIM_odom_message_to_tf/pose", geo_msgs.PoseStamped)
    # rospy.loginfo(base_message)
    return base_message.pose

def main():
    rospy.init_node('getPose', anonymous=True)
    
    #base position
    rawbasePose = getBaseposition()
    basePose = np.zeros(3)
    basePose[0] = rawbasePose.position.x
    basePose[1] = rawbasePose.position.y
    basePose[2] = rawbasePose.position.z

    #base rotation
    baseRot = np.zeros(4)
    baseRot[0] = rawbasePose.orientation.w
    baseRot[1] = rawbasePose.orientation.y
    baseRot[2] = rawbasePose.orientation.x
    baseRot[3] = rawbasePose.orientation.z

    #End-Effector position and rotation
    rawEEPose, rawEERot = getEEposition()
    EEPose = np.array(rawEEPose)
    EERot = np.array(rawEERot)

    #print information about base
    print ("Base positon: ", basePose)
    print ("Base rotation: ", baseRot)

    #print information about End-Effector
    print ("End-Effector position: ", EEPose)
    print ("End-Effector rotation: ", EERot)
    #rospy.Subscriber("/bvr_SIM/bvr_SIM_odom_message_to_tf/pose", geo_msgs.PoseStamped, callback)

    #rospy.spin()


if __name__ == '__main__':
    main()

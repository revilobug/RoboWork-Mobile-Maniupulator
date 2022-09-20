import rospy
import geometry_msgs.msg as geo_msgs

def callback(data):
    rospy.loginfo(data)

def main():
    rospy.init_node('getPose', anonymous=True)

    rospy.Subscriber("/bvr_SIM/bvr_SIM_odom_message_to_tf/pose", geo_msgs.PoseStamped, callback)

    rospy.spin()

if __name__ == '__main__':
    main()

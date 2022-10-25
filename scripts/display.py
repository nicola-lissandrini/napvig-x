#!/usr/bin/env python

from math import sqrt
import signal
import rospy 
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D, Vector3, PoseStamped
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt

class DisplayNode:
    def measures_callback (self, measures: Float32MultiArray):
        self.meas_np = np.array (measures.data[1:])
        self.meas_np = np.reshape (self.meas_np, (measures.layout.dim[0].size, measures.layout.dim[1].size))
        
        self.draw ()

    def history_callback (self, historyMsg: Float32MultiArray):
        self.history = np.array (historyMsg.data[1:])
        
        if (len (historyMsg.layout.dim) > 2):
            self.multi_history = True
            self.history = np.reshape (self.history, (historyMsg.layout.dim[0].size, historyMsg.layout.dim[1].size, historyMsg.layout.dim[2].size))
        else:
            self.multi_history = False
            self.history = np.reshape (self.history, (historyMsg.layout.dim[0].size, historyMsg.layout.dim[1].size))
        

    def values_callback (self, valuesMsg: Float32MultiArray):
        values_data = np.array (valuesMsg.data[1:])

        self.values = np.reshape (values_data, (valuesMsg.layout.dim[0].size, valuesMsg.layout.dim[1].size))
        

    def gradients_callback (self, gradientsMsg):
        gradients_data = np.array (gradientsMsg.data[1:])

        self.gradients = np.reshape (gradients_data, (gradientsMsg.layout.dim[0].size, gradientsMsg.layout.dim[1].size))


    def command_callback (self, commandMsg):
        self.command = commandMsg

    def draw_turtle (self, command: Pose2D, color="r", arrow_color="k"):
        dir_x = np.cos (command.theta)
        dir_y = np.sin (command.theta)

        plt.scatter (command.x, command.y, 4.5, c=color)
        plt.quiver (command.x, command.y, dir_x, dir_y, color=arrow_color)

    def pose_callback (self, pose: PoseStamped):
        self.pose = Pose2D ()

        self.pose.x = pose.pose.position.x
        self.pose.y = pose.pose.position.y

        orientation_list = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        (_, _, self.pose.theta) = euler_from_quaternion (orientation_list)

    def target_callback (self, target_msg):
        self.target = np.array (target_msg.data[1:])

    def __init__ (self):
        rospy.init_node ("display")

        self.fig = plt.figure ()
        
        self.meas_sub = rospy.Subscriber ("/napvig/outputs/measures", Float32MultiArray, self.measures_callback, queue_size=1)
        self.values_sub = rospy.Subscriber ("/napvig/outputs/values", Float32MultiArray, self.values_callback, queue_size=1)
        self.gradients_sub = rospy.Subscriber ("/napvig/outputs/gradients", Float32MultiArray, self.gradients_callback, queue_size=1)
        self.pose_sub = rospy.Subscriber ("/napvig/outputs/pose_debug", PoseStamped, self.pose_callback, queue_size=1)
        self.command_sub = rospy.Subscriber (rospy.get_param ("/napvig/topics/pubs/command"), Pose2D, self.command_callback, queue_size=1)
        self.target_sub = rospy.Subscriber ("/napvig/outputs/target_in_measures", Float32MultiArray, self.target_callback, queue_size=1)
        self.target_history = rospy.Subscriber ("/napvig/outputs/history", Float32MultiArray, self.history_callback, queue_size=1)

        self.range_min = rospy.get_param ("/napvig/napvig/debug/output_range/min")
        self.range_max = rospy.get_param ("/napvig/napvig/debug/output_range/max")
        self.range_step = rospy.get_param ("/napvig/napvig/debug/output_range/step")
        self.fig.canvas.mpl_connect('close_event', handle_close)

        self.meas_np = None
        self.values = None
        self.gradients = None
        self.command = None
        self.pose = None
        self.target = None
        self.history = None
        self.multi_history = False

        plt.show ()

    
    
    def draw (self):
        self.fig.clear ()
        
        if (not self.values is None):
            sq = int (sqrt (self.values.shape[0]))
            grid_x = np.reshape (self.values[:,0], (sq, sq))
            grid_y = np.reshape (self.values[:,1], (sq, sq))
            val_grid = np.reshape (self.values[:,2], (sq, sq))

            plt.pcolormesh (grid_x, grid_y, val_grid, edgecolors="none", antialiased=True, vmin=0, vmax=1)

        if (not self.gradients is None):
            plt.quiver (self.gradients[:,0], self.gradients[:,1], self.gradients[:,2], self.gradients[:,3], minshaft=0.1)

        if (not self.meas_np is None):
            plt.scatter (self.meas_np[:, 0], self.meas_np[:, 1], 2.5, color="black")

        if (not self.command is None and self.history is None):
            self.draw_turtle (self.command)

        if (not self.pose is None):
            self.draw_turtle (self.pose)

        if (not self.target is None):
            plt.scatter (self.target[0], self.target[1], 4, color="red")

        if (not self.history is None):
            if (self.multi_history):
                for i in range(np.shape(self.history)[0]):
                    plt.quiver (self.history[i,:,0], self.history[i,:,1], self.history[i,:,2], self.history[i,:,3])
            else:
                plt.quiver (self.history[:,0], self.history[:,1], self.history[:,2], self.history[:,3])
            #plt.scatter (self.history[:,0], self.history[:,1], 4, color="blue")
            
        
        plt.gca().set_xlim (self.range_min, self.range_max)
        plt.gca().set_ylim (self.range_min, self.range_max)
        plt.draw ()

    def spin (self):
        rospy.spin ()

def signal_handler (sig, frame):
    plt.close ()

def handle_close (fig):
    quit ()

if __name__ == "__main__":
    try:
        signal.signal (signal.SIGINT, signal_handler)
        dn = DisplayNode ()
        dn.spin
    except rospy.ROSInterruptException:
        pass
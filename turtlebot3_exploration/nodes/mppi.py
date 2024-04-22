#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
import tf
from geometry_msgs.msg import PointStamped
from tf.transformations import euler_from_quaternion
from math import sin, cos
import numpy as np

class mppi_working:

    def __init__(self):
        self.flag=0
        self.flag1=0
        # self.current_time = rospy.Time.now()
        # self.last_time = rospy.Time.now()
        rospy.init_node('turtlebot_exploration')
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
         # self.width=None
        # self.height = None
        # self.resolution = None
        # self.origin_position_x = None
        # self.origin_position_y = None
        # self.map_data = None
        self.rate=rospy.Rate(1)
        self.num_of_trajectories=100
        self.horizon=30
        self.radius=0.03
        self.dt=0.5
        # self.theta= None
        self.trajectory_states=np.zeros((self.num_of_trajectories,self.horizon,2))
        self.actions_of_trajectories=np.zeros((self.num_of_trajectories,self.horizon,2,1))
        self.cost_of_trajectories=np.zeros(self.num_of_trajectories)
        self.best_trajectory=np.zeros((self.horizon,2))
        self.best_actions=np.zeros((self.horizon,2))
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
       
        
        
        
        
        

    def map_callback(self,msg):
        # print("map")
        self.flag=1
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_position_x = msg.info.origin.position.x
        self.origin_position_y = msg.info.origin.position.y
        self.theta=0
        self.map_data = msg.data  # Occupancy grid data (1D array)
        self.map_data=np.array(self.map_data)
        # print("Map width:"+ str(self.width))
        # print("Map height:"+ str(self.height))
        # print("Map resolution:"+ str(self.resolution))
        # print("Map origin position (x, y):", (self.origin_position_x), self.origin_position_y)
        # Print first few elements of the occupancy grid data
        # print("First 10 elements of occupancy grid data:", self.map_data[:10])


    def odom_callback(self, msg):
        # print("odom")
        self.flag1=1
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        orientation_quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        _, _, self.robot_theta = euler_from_quaternion(orientation_quaternion)
        # print("Robot x,y,theta : "+str(self.robot_x)+str(self.robot_y)+str(self.robot_theta))
    
    # def caller(self):
    #     while not rospy.is_shutdown():
    #      if self.flag==1:
    #           print("Odom pose: ",self.robot_x,self.robot_y)
    #           print("Map pose",self.origin_position_x,self.origin_position_y)
    #           a=self.transform_point_odom_to_map((self.robot_x,self.robot_y))
    #           print("Odom pos in map: ",a)
    #           self.flag=0
    #     self.rate.sleep()

    def transform_point_odom_to_map(self,odom_point):
        # Initialize node
        # rospy.init_node('point_transformer')

        # Initialize tf listener
        

        # Create a PointStamped message representing the point in the odom frame
        point_odom = PointStamped()
        point_odom.header.frame_id = 'odom'  # Set frame ID to odom
        point_odom.point.x = odom_point[0]
        point_odom.point.y = odom_point[1]
        point_odom.point.z = 0.0  # Assuming the point lies in the XY plane

        # Wait for the transformation between odom and map frames to become available
        self.tf_listener.waitForTransform('odom', '/map', rospy.Time(0), rospy.Duration(1.0))

        # Transform the point from odom to map frame
        try:
            point_map = self.tf_listener.transformPoint('/map', point_odom)
            # rospy.loginfo("Point in map frame: (%.2f, %.2f)", point_map.point.x, point_map.point.y)
            return (point_map.point.x, point_map.point.y)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # rospy.logerr("Failed to transform point: %s", e)
            return None

    
    

    def grid_indices_to_map_coordinates(self,row, col):
        x = self.origin_position_x + (col * self.resolution)
        y = self.origin_position_y + (row * self.resolution)
        return round(x,5), round(y,5)

    def map_coordinates_to_grid_indices(self,x, y):
        col = int((x - self.origin_position_x) / self.resolution)
        row = int((y - self.origin_position_y) / self.resolution)
        return row, col


    def step(self, state, action):
        x, y, theta = state
        v, w = action
        theta += w * self.dt
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt

        new_state = np.array([x, y, theta])
        return new_state
    

    def make_trajectories(self):
        
        delta_actions_v = np.random.uniform(low=0, high=0.2, size=(self.num_of_trajectories, self.horizon,1))#changed high from 0.22 to 0.5
        delta_actions_w=np.random.uniform(low=-0.2, high=0.2, size=(self.num_of_trajectories, self.horizon,1))                                    
        delta_actions=np.array(list(zip(delta_actions_v.T, delta_actions_w.T))).T
        self.actions_of_trajectories=delta_actions
        for i in range(self.num_of_trajectories):
            state=(self.robot_x,self.robot_y,self.robot_theta)
            for j in range(self.horizon):
                action=delta_actions[i][j]
                new_state=self.step(state,action)
                self.trajectory_states[i,j][0]=new_state[0]
                self.trajectory_states[i,j][1]=new_state[1]
                state=new_state
        # return self.trajectory_states
    
    def calculate_costs(self):
            exploration_weight=-100#0.7
            obstacle_weight = 10000000#0.3
            free_weight=0

            for i in range(self.num_of_trajectories):
                cost=0
                unknowns_in_range = 0
                distance=0
                proximity_cost = 0
                number_of_frontiers = 0
                for j in range(self.horizon):
                    x_robot_frame,y_robot_frame=self.trajectory_states[i,j]
                    x_map_frame,y_map_frame=self.transform_point_odom_to_map((x_robot_frame,y_robot_frame))
                    
                    row,col=self.map_coordinates_to_grid_indices(x_map_frame,y_map_frame)
                        
                    linear_index= row * self.width + col
                    
                    #range_of_view_LIDAR
                    for k in range(-10,10):
                        for l in range(-10,10):
                            #print(linear_index, k,l)
                            if(row+k<384 and row+k>0 and col+l>0 and col+l<384):
                                index = linear_index + (k*self.width) + l#((row+k)*self.width) + (col)+l
                                if int(self.map_data[index]) == -1:
                                    unknowns_in_range += -10
                    #print(unknowns_in_range)
                            
                    
                    #collision to account for robot proximity            
                    for m in range(-2,2):
                        for n in range(-2,2):
                            if(row+m<384 and row+m>0 and col+n>0 and col+n<384):
                                index = linear_index + (m*self.width) + n
                                if int(self.map_data[index]) == 100:
                                    proximity_cost += 100000
                    #print(proximity_cost)
                    
                    unknowns = 0
                    knowns = 0
                    for o in range(-1,1):
                        for p in range(-1,1):
                            #print(linear_index, k,l)
                            if(row+o<384 and row+o>0 and col+p>0 and col+p<384):
                                index = linear_index + (o*self.width) + p#((row+k)*self.width) + (col)+l
                                if int(self.map_data[index]) == -1:
                                    unknowns += 1
                                
                                if unknowns>3 and unknowns<6:
                                    number_of_frontiers +=1
                                    
                   
                    if int(self.map_data[linear_index])>1 and int(self.map_data[linear_index])<101:#==100:
                        cost+= obstacle_weight#/(j+1)
                        
                    if int(self.map_data[linear_index])== -1:
                        cost+= exploration_weight#*(j+1)

                    if int(self.map_data[linear_index])== 0: 
                        cost+= free_weight#*(j+1)
                    # else:
                    #     cost=float('-inf')
                    #     break
                    cost = cost + unknowns_in_range + proximity_cost + (number_of_frontiers*-10)
                self.cost_of_trajectories[i]=cost
            # return self.cost_of_trajectories
    

    def choose_trajectory(self):
        
            index=np.argmin(self.cost_of_trajectories)
            self.best_trajectory=self.trajectory_states[index]
            self.best_actions=self.actions_of_trajectories[index]
            # print(self.best_trajectory)
            # print(self.best_actions)
            # print(self.cost_of_trajectories)while
  


            for i,speed in enumerate(self.best_actions):
                if i==5:
                    
                    break
                else:
                    print(speed)
                    linear_speed=speed[0][0]
                    angular_speed=speed[1][0]
                    print(linear_speed,angular_speed)
                    cmd_vel_msg = Twist()
                    cmd_vel_msg.linear.x = linear_speed
                    cmd_vel_msg.angular.z = angular_speed
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    print("Number of msgs published: ",i)

            # cmd_vel_msg = Twist()
            # cmd_vel_msg.linear.x = 0
            # cmd_vel_msg.angular.z = 0
            # self.cmd_vel_pub.publish(cmd_vel_msg)
            
    def call_mppi(self):        
        while not rospy.is_shutdown():
         if self.flag==1 and self.flag1==1:
            self.make_trajectories()
            self.calculate_costs()
            self.choose_trajectory()
            self.flag= 0
            self.flag1=0

        #  elif self.flag==0:
        #     cmd_vel_msg = Twist()
        #     cmd_vel_msg.linear.x = 0
        #     cmd_vel_msg.angular.z = 0
        #     self.cmd_vel_pub.publish(cmd_vel_msg)
         self.rate.sleep()


if __name__ == '__main__':
    try:       
            mppi = mppi_working()
            mppi.call_mppi()

    except rospy.ROSInterruptException:
        pass

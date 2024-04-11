
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion

import numpy as np

#Map cell values to set
OBSTACLE_VALUE = 100    #to be displayed in black on RVIZ (with map color scheme)
FREE_SPACE_VALUE = 0
UNEXPLORED_SPACE_VALUE = -1


class Agent(Node):
    """
    This class is used to define the behavior of ONE agent
    """
    def __init__(self):
        Node.__init__(self, "Agent")
        
        self.load_params()

        #initialize attributes
        self.agents_pose = [None]*self.nb_agents    #[(x_1, y_1), (x_2, y_2), (x_3, y_3)] if there are 3 agents
        self.x = self.y = self.yaw = None   #the pose of this specific agent running the node
        self.front_dist = self.left_dist = self.right_dist = 0.0    #range values for each ultrasonic sensor
        self.last_left_dist = None
        self.last_right_dist = None
        self.last_bool_right = False
        self.last_bool_left = False
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.init_map()

        #Subscribe to agents' pose topic
        odom_methods_cb = [self.odom1_cb, self.odom2_cb, self.odom3_cb]
        for i in range(1, self.nb_agents + 1):  
            self.create_subscription(Odometry, f"/bot_{i}/odom", odom_methods_cb[i-1], 1)
        
        if self.nb_agents != 1: #if other agents are involved subscribe to the merged map topic
            self.create_subscription(OccupancyGrid, "/merged_map", self.merged_map_cb, 1)
        
        #Subscribe to ultrasonic sensor topics for the corresponding agent
        self.create_subscription(Range, f"{self.ns}/us_front/range", self.us_front_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed at front of the robot
        self.create_subscription(Range, f"{self.ns}/us_left/range", self.us_left_cb, qos_profile=qos_profile_sensor_data)   #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the left side of the robot
        self.create_subscription(Range, f"{self.ns}/us_right/range", self.us_right_cb, qos_profile=qos_profile_sensor_data) #subscribe to the agent's own us front topic to get distance measurements from ultrasonic sensor placed on the right of the robot
        
        self.cmd_vel_pub = self.create_publisher(Twist, f"{self.ns}/cmd_vel", 1)    #publisher to send velocity commands to the robot

        #Create timers to autonomously call the following methods periodically
        self.create_timer(0.2, self.map_update) #0.2s of period <=> 5 Hz
        self.create_timer(0.5, self.strategy)      #0.5s of period <=> 2 Hz
        self.create_timer(1, self.publish_maps) #1Hz
    

    def load_params(self):
        """ Load parameters from launch file """
        self.declare_parameters(    #A node has to declare ROS parameters before getting their values from launch files
            namespace="",
            parameters=[
                ("ns", rclpy.Parameter.Type.STRING),    #robot's namespace: either 1, 2 or 3
                ("robot_size", rclpy.Parameter.Type.DOUBLE),    #robot's diameter in meter
                ("env_size", rclpy.Parameter.Type.INTEGER_ARRAY),   #environment dimensions (width height)
                ("nb_agents", rclpy.Parameter.Type.INTEGER),    #total number of agents (this agent included) to map the environment
            ]
        )

        #Get launch file parameters related to this node
        self.ns = self.get_parameter("ns").value
        self.robot_size = self.get_parameter("robot_size").value
        self.env_size = self.get_parameter("env_size").value
        self.nb_agents = self.get_parameter("nb_agents").value
    

    def init_map(self):
        """ Initialize the map to share with others if it is bot_1 """
        self.map_msg = OccupancyGrid()
        self.map_msg.header.frame_id = "map"    #set in which reference frame the map will be expressed (DO NOT TOUCH)
        self.map_msg.header.stamp = self.get_clock().now().to_msg() #get the current ROS time to send the msg
        self.map_msg.info.resolution = self.robot_size  #Map cell size corresponds to robot size
        self.map_msg.info.height = int(self.env_size[0]/self.map_msg.info.resolution)   #nb of rows
        self.map_msg.info.width = int(self.env_size[1]/self.map_msg.info.resolution)    #nb of columns
        self.map_msg.info.origin.position.x = -self.env_size[1]/2   #x and y coordinates of the origin in map reference frame
        self.map_msg.info.origin.position.y = -self.env_size[0]/2
        self.map_msg.info.origin.orientation.w = 1.0    #to have a consistent orientation in quaternion: x=0, y=0, z=0, w=1 for no rotation
        self.map = np.ones(shape=(self.map_msg.info.height, self.map_msg.info.width), dtype=np.int8)*UNEXPLORED_SPACE_VALUE #all the cells are unexplored initially
        self.w, self.h = self.map_msg.info.width, self.map_msg.info.height  
    

    def merged_map_cb(self, msg):
        """ 
            Get the current common map and update ours accordingly.
            This method is automatically called whenever a new message is published on the topic /merged_map.
            'msg' is a nav_msgs/msg/OccupancyGrid message.
        """
        received_map = np.flipud(np.array(msg.data).reshape(self.h, self.w))    #convert the received list into a 2D array and reverse rows
        for i in range(self.h):
            for j in range(self.w):
                if (self.map[i, j] == UNEXPLORED_SPACE_VALUE) and (received_map[i, j] != UNEXPLORED_SPACE_VALUE):
                    self.map[i, j] = received_map[i, j]


    def odom1_cb(self, msg):
        """ 
            Get agent 1 position.
            This method is automatically called whenever a new message is published on topic /bot_1/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 1:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[0] = (x, y)
        # self.get_logger().info(f"Agent 1: ({x:.2f}, {y:.2f})")
    

    def odom2_cb(self, msg):
        """ 
            Get agent 2 position.
            This method is automatically called whenever a new message is published on topic /bot_2/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 2:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[1] = (x, y)
        # self.get_logger().info(f"Agent 2: ({x:.2f}, {y:.2f})")


    def odom3_cb(self, msg):
        """ 
            Get agent 3 position.
            This method is automatically called whenever a new message is published on topic /bot_3/odom.
            'msg' is a nav_msgs/msg/Odometry message.
        """
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if int(self.ns[-1]) == 3:
            self.x, self.y = x, y
            self.yaw = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.agents_pose[2] = (x, y)
        # self.get_logger().info(f"Agent 3: ({x:.2f}, {y:.2f})")

    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates."""
        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y
        resolution = self.map_msg.info.resolution
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        return grid_x, grid_y

    def calculate_obstacle_position(self, distance, sensor_angle):
        """
        Calculer la position de l'obstacle détecté par le capteur.
        :param distance: Distance mesurée par le capteur.
        :param sensor_angle: Angle du capteur par rapport à l'orientation du robot.
        :return: Coordonnées de la grille de la carte où se trouve l'obstacle.
        """
        # Calculer l'angle global de l'obstacle par rapport au monde
        global_angle = self.yaw + sensor_angle

        # Calculer la position globale de l'obstacle
        obstacle_x = self.x + distance * math.cos(global_angle)
        obstacle_y = -(self.y + distance * math.sin(global_angle))

        # Convertir en coordonnées de grille
        grid_x, grid_y = self.world_to_grid(obstacle_x, obstacle_y)
        return grid_x, grid_y    

    def map_update(self):
        sensor_angles = {'front': 0, 'left': np.pi / 2, 'right': -np.pi / 2}
        sensor_max_range = 3  # La portée maximale des capteurs en mètres
        
        for sensor, angle_offset in sensor_angles.items():
            distance = getattr(self, f"{sensor}_dist")
            sensor_range = min(distance, sensor_max_range)  # Utilise la distance mesurée ou la portée maximale
            
            # Itérer sur la distance depuis le robot jusqu'à l'obstacle ou la portée maximale du capteur
            for d in np.linspace(self.robot_size/2, sensor_range, num=int(sensor_range / self.map_msg.info.resolution) + 1):
                obstacle_x, obstacle_y = self.x + d * math.cos(self.yaw + angle_offset), self.y + d * math.sin(self.yaw + angle_offset)
                
                # Convertir en coordonnées de grille
                grid_x, grid_y = self.world_to_grid(obstacle_x, -obstacle_y)  # Assume -y pour correspondre à votre convention
                
                if 0 <= grid_x < self.map_msg.info.width and 0 <= grid_y < self.map_msg.info.height:
                    if d < distance:
                        # Marquer comme espace libre si on est pas encore à la distance de l'obstacle
                        self.map[grid_y, grid_x] = FREE_SPACE_VALUE

                    elif d == distance and distance < sensor_max_range:
                        # Marquer comme obstacle si la distance correspond à celle de l'obstacle détecté
                        self.map[grid_y, grid_x] = OBSTACLE_VALUE



    def us_front_cb(self, msg):
        """ 
            Get measurement from the front ultrasonic sensor.
            This method is automatically called whenever a new message is published on topic /bot_x/us_front/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.front_dist = msg.range


    def us_left_cb(self, msg):
        """ 
            Get measurement from the ultrasonic sensor placed on the left.
            This method is automatically called whenever a new message is published on topic /bot_x/us_left/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.left_dist = msg.range


    def us_right_cb(self, msg):
        """ 
            Get measurement from the ultrasonic sensor placed on the right.
            This method is automatically called whenever a new message is published on topic /bot_x/us_right/range, where 'x' is either 1, 2 or 3.
            'msg' is a sensor_msgs/msg/Range message.
        """
        self.right_dist = msg.range
    

    def publish_maps(self):
        """ 
            Publish updated map to topic /bot_x/map, where x is either 1, 2 or 3.
            This method is called periodically (1Hz) by a ROS2 timer, as defined in the constructor of the class.
        """
        self.map_msg.data = np.flipud(self.map).flatten().tolist()  #transform the 2D array into a list to publish it
        self.map_agent_pub.publish(self.map_msg)    #publish map to other agents


    def strategy(self):
        obstacle_distance = 1.5
        desired_distance_from_wall = 1.0

        # Initialiser un message Twist pour commander le robot
        cmd_vel = Twist()

        # Vérifier la distance aux obstacles
        obstacle_front = self.front_dist < obstacle_distance
        wall_on_left = self.left_dist < obstacle_distance
        wall_on_right = self.right_dist < obstacle_distance

        if obstacle_front and not wall_on_right and not wall_on_left:
            # Si un obstacle est détecté devant, tourner à droite
            cmd_vel.angular.z = 1.2
        
        elif obstacle_front and wall_on_left and not wall_on_right:
            # Si un obstacle est détecté devant et un mur a gauche, tourner à droite
            cmd_vel.angular.z = -1.2

        elif obstacle_front and wall_on_right and not wall_on_left:
            # Si un obstacle est détecté devant et un mur a droite, tourner à gauche
            cmd_vel.angular.z = 1.2

        elif obstacle_front and wall_on_right and wall_on_left:
            # Si cul de sac alors demi-tour
            cmd_vel.angular.z = np.pi

        elif wall_on_left and not wall_on_right and not obstacle_front:
            # Si un mur est détecté à gauche, suivre le mur
            getting_closer = False
            getting_further = False

            if self.last_left_dist is not None:

                getting_closer = self.left_dist < self.last_left_dist
                getting_further = self.left_dist > self.last_left_dist

                if self.left_dist > desired_distance_from_wall:
                    # Si le robot s'éloigne du mur, augmentez l'angle de rotation pour revenir vers le mur
                    cmd_vel.angular.z = 0.12 if getting_further else 0.01
                elif self.left_dist < desired_distance_from_wall:
                    # Si le robot est trop proche, diminuez l'angle ou tournez légèrement à droite pour s'éloigner du mur
                    cmd_vel.angular.z = -0.12 if getting_closer else -0.01
                cmd_vel.linear.x = 0.3

            # Mise à jour de la dernière distance mesurée
            self.last_left_dist = self.left_dist
            
        elif wall_on_right and not wall_on_left and not obstacle_front:
            # Si un mur est détecté à droite, suivre le mur
            getting_closer = False

            if self.last_right_dist is not None:

                getting_closer = self.right_dist < self.last_right_dist
                getting_further = self.right_dist > self.last_right_dist

                if self.right_dist > desired_distance_from_wall:
                    # Si le robot s'éloigne du mur, diminuez l'angle de rotation pour revenir vers le mur
                    cmd_vel.angular.z = -0.12 if getting_further else -0.01
                elif self.right_dist < desired_distance_from_wall:
                    # Si le robot est trop proche, augmentez l'angle ou tournez légèrement à gauche pour s'éloigner du mur
                    cmd_vel.angular.z = 0.12 if getting_closer else 0.01
                cmd_vel.linear.x = 0.3

            # Mise à jour de la dernière distance mesurée
            self.last_right_dist = self.right_dist
        
        else:
            # Si aucun mur n'est détecté
            # Rotation si perte de mur
            if self.last_bool_right :
                cmd_vel.angular.z = -1.1
            elif self.last_bool_left :
                cmd_vel.angular.z = 1.1
            else :
                cmd_vel.linear.x = 0.4   

        self.last_bool_left = wall_on_left
        self.last_bool_right = wall_on_right

        # Publier la commande de vitesse
        self.cmd_vel_pub.publish(cmd_vel)


def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

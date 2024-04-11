
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
        self.map_agent_pub = self.create_publisher(OccupancyGrid, f"/{self.ns}/map", 1) #publisher for agent's own map
        self.initial_rotation_done = False
        self.rotating_at_frontier = False
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
        sensor_max_range = 3  
        # La portée maximale des capteurs en mètres
        
        for sensor, angle_offset in sensor_angles.items():
            distance = getattr(self, f"{sensor}_dist")
            sensor_range = min(distance, sensor_max_range)  # Utilise la distance mesurée ou la portée maximale
            
            # Itérer sur la distance depuis le robot jusqu'à l'obstacle ou la portée maximale du capteur
            for d in np.linspace(0, sensor_range, num=int(sensor_range / self.map_msg.info.resolution) + 1):
                obstacle_x, obstacle_y = self.x + d * math.cos(self.yaw + angle_offset), self.y + d * math.sin(self.yaw + angle_offset)
                
                # Convertir en coordonnées de grille
                grid_x, grid_y = self.world_to_grid(obstacle_x, -obstacle_y) 
                
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
    
    def calculate_ranks_and_assign(self, frontiers):
        """
        Calculer les rangs de chaque robot pour chaque frontière et assigner chaque robot à la meilleure frontière.
        """
        assignments = {}  # Dictionnaire pour garder la trace de l'assignation robot-frontière

        # Calculer la distance de chaque robot à chaque frontière
        for frontier in frontiers:
            distances = []
            for pos in self.agents_pose:
                if pos is not None:  # Vérifier si la position de l'agent est connue
                    distance = np.sqrt((pos[0] - frontier[0])**2 + (pos[1] - frontier[1])**2)
                    distances.append((distance, pos))
            
            # Trier les distances pour calculer les rangs
            distances.sort()
            for rank, (_, pos) in enumerate(distances):
                if pos == (self.x, self.y):  # Si la position courante correspond à celle de cet agent
                    assignments[frontier] = rank

        # Assigner à chaque robot la frontière avec le rang le plus bas
        best_frontier = min(assignments, key=assignments.get)
        return best_frontier
    
    def perform_360_rotation(self):
        cmd_vel = Twist()
        cmd_vel.angular.z = 2 * np.pi 
        self.cmd_vel_pub.publish(cmd_vel)

    def strategy(self):
        # Trouver les frontières dans la carte
        frontiers = self.find_frontiers()
        cmd_vel = Twist()

        if not self.initial_rotation_done:
            self.perform_360_rotation()
            self.initial_rotation_done = True  # Assurez-vous de définir ceci correctement une fois la rotation terminée
        else:
            best_frontier = self.calculate_ranks_and_assign(frontiers)
            self.move_to_frontier(best_frontier)


    def find_frontiers(self):
        """
        Détecter les frontières dans la carte actuelle.
        """
        frontiers = []
        for y in range(self.h):
            for x in range(self.w):
                # Condition pour identifier une frontière: un espace inexploré adjacent à un espace libre
                if self.map[y, x] == FREE_SPACE_VALUE:
                    neighbors = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
                    for ny, nx in neighbors:
                        if 0 <= ny < self.h and 0 <= nx < self.w and self.map[ny, nx] == UNEXPLORED_SPACE_VALUE:
                            frontiers.append((x, y))  # Ajoute la position de la grille en tant que frontière
                            break
        return frontiers

    def move_to_frontier(self, frontier):
        # Convertir les coordonnées de la grille (frontière) en coordonnées mondiales
        world_x, world_y = self.grid_to_world(frontier[0], frontier[1])

        # Calculer la direction et la distance vers la frontière
        direction = math.atan2(world_y - self.y, world_x - self.x)
        distance = math.sqrt((world_x - self.x) ** 2 + (world_y - self.y) ** 2)

        # Normaliser la différence d'angle pour s'assurer qu'elle est dans [-pi, pi]
        angle_diff = direction - self.yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Créer le message Twist pour commander le robot
        cmd_vel = Twist()

        # Si le robot n'est pas orienté vers la direction cible, ajuster la vitesse angulaire
        if abs(angle_diff) > 0.1:  # Seuil d'angle pour commencer à avancer, ajustez selon vos besoins
            cmd_vel.angular.z = 2.0 * angle_diff  # Coefficient proportionnel pour la correction de l'angle
            cmd_vel.linear.x = 0.0  # Éviter d'avancer tant que l'orientation n'est pas correcte
        else:
            # Une fois orienté presque correctement, commencer à avancer
            cmd_vel.linear.x = min(0.5, distance)  # Avancer à une vitesse max de 0.5 m/s, ajustez selon vos besoins
            cmd_vel.angular.z = 0.0  # Plus besoin de tourner

        self.cmd_vel_pub.publish(cmd_vel)


    def grid_to_world(self, grid_x, grid_y):
        """
        Convertir les coordonnées de la grille en coordonnées mondiales.
        """
        world_x = grid_x * self.map_msg.info.resolution + self.map_msg.info.origin.position.x
        world_y = grid_y * self.map_msg.info.resolution + self.map_msg.info.origin.position.y
        return world_x, world_y


    

def main():
    rclpy.init()

    node = Agent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

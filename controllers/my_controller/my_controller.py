
import math
import time
import numpy as np
from sklearn.cluster import KMeans
from controller import Robot
from controller import Camera

class ColorSortingRobot:
    def __init__(self):
        # Initialize robot and devices
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize motors and sensors
        self.base_motor = self.robot.getDevice('base_motor')
        self.first_motor = self.robot.getDevice('first_motor')
        self.second_motor = self.robot.getDevice('second_motor')
        self.gripper = self.robot.getDevice('vacuum gripper')
        self.camera = self.robot.getDevice('camera')
        
        # Enable devices
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.camera.enableRecognitionSegmentation()
        
        # Constants
        self.zero = [0, 0, 0.03]    # Robot center
        self.balls = [0.105, 0.105, 0.013]    # Ball slide
        self.red_bowl = [0, 0.13, 0.3]
        self.green_bowl = [0, -0.13, 0.3]
        self.blue_bowl = [0.15, 0, 0.3]
        self.yellow_bowl = [0, -0.15, 0.3]  # Added yellow bowl position
        self.purple_bowl = [-0.15,0.15, 0.3]
        self.a = 0.1  # Length of first arm
        self.c = 0.14  # Length of second arm
        
        # Color reference points (RGB values)
        self.color_references = {
            'red': np.array([[255, 0, 0]]),
            'green': np.array([[0, 255, 0]]),
            'blue': np.array([[0, 0, 255]]),
            'yellow': np.array([[255, 255, 0]]),
            'purple': np.array([[128, 0, 128]])   # Added yellow reference
        }
        
        # Initialize K-means classifier with 4 clusters
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.train_color_classifier()

    def train_color_classifier(self):
        """Train the K-means classifier with reference colors"""
        training_data = np.vstack([
            self.color_references['red'],
            self.color_references['green'],
            self.color_references['blue'],
            self.color_references['yellow'],
            self.color_references['purple']    # Added yellow to training data
        ])
        self.kmeans.fit(training_data)
        
    def detect_color(self, image):
        """
        Detect color using K-means clustering
        Returns: 'red', 'green', 'blue', 'yellow' or None
        """
        if not image:
            return None
            
        # Extract colors from image
        colors = np.array([
            [image[i][j] for j in range(len(image[0]))]
            for i in range(len(image))
        ])
        colors = colors.reshape(-1, 3)
        
        # Remove black background pixels
        mask = np.any(colors != [0, 0, 0], axis=1)
        colors = colors[mask]
        
        if len(colors) == 0:
            return None
            
        # Find dominant color using K-means
        dominant_color = np.mean(colors, axis=0).reshape(1, -1)
        
        # Calculate color distances
        distances = {
            'red': np.linalg.norm(dominant_color - self.color_references['red']),
            'green': np.linalg.norm(dominant_color - self.color_references['green']),
            'blue': np.linalg.norm(dominant_color - self.color_references['blue']),
            'yellow': np.linalg.norm(dominant_color - self.color_references['yellow']),
            'purple': np.linalg.norm(dominant_color - self.color_references['purple'])
        }
        
        # Additional yellow detection logic
        r, g, b = dominant_color[0]
        if r > 100 and g > 100 and b < 100:  # Strong yellow detection
            return 'purple'
            
        return min(distances.items(), key=lambda x: x[1])[0]
        
    def inverse_kinematics_3d(self, zero, point, a, c):
        """Calculate inverse kinematics for the arm"""
        # Swap Y and Z coordinates
        zero = [zero[0], zero[2], zero[1]]
        point = [-point[0], -point[2], point[1]]
        
        # Calculate theta
        theta = math.atan2(point[2] - zero[2], point[0] - zero[0])
        
        # Calculate length of b
        b = math.sqrt((point[0] - zero[0])**2 + (point[2] - zero[2])**2)
        
        # Calculate A'
        delta_y = point[1] - zero[1]
        delta_w = math.sqrt((point[0] - zero[0])**2 + (point[1] - zero[1])**2 + (point[2] - zero[2])**2)
        A_prime = math.atan2(delta_y, delta_w)
        
        # Calculate A
        A = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) + A_prime
        
        # Calculate B
        B = math.pi - math.acos((a**2 + c**2 - b**2) / (2 * a * c))
        
        return theta, A, B
        
    def move_arm(self, target_position):
        """Move the arm to a target position"""
        theta, A, B = self.inverse_kinematics_3d(self.zero, target_position, self.a, self.c)
        self.base_motor.setPosition(theta)
        self.first_motor.setPosition(A)
        self.second_motor.setPosition(B)
        
    def get_target_bowl(self, color):
        """Get the target bowl position based on color"""
        bowl_positions = {
            'red': self.red_bowl,
            'green': self.green_bowl,
            'blue': self.blue_bowl,
            'yellow': self.yellow_bowl,
            'purple': self.purple_bowl    # Added yellow bowl
        }
        return bowl_positions.get(color)
        
    def run(self):
        i = 0
        while self.robot.step(self.timestep) != -1:
            if i == 0:    # Move to balls
                self.move_arm(self.balls)
                
            elif i == 100*4:    # Enable gripper
                self.gripper.turnOn()
                
            elif i == 200*4:    # Detect color and move to bowl
                image = self.camera.getRecognitionSegmentationImageArray()
                color = self.detect_color(image)
                
                if color:
                    print(f"Detected {color}")
                    target_bowl = self.get_target_bowl(color)
                    if target_bowl:
                        self.move_arm(target_bowl)
                    elif color == 'purple':  # No bowl for purple
                        print("No bowl for purple, skipping")    
                    
            elif i == 300*4:    # Drop ball and reset
                self.gripper.turnOff()
                i = -1
                
            i += 1

# Create and run the robot
robot = ColorSortingRobot()
robot.run()

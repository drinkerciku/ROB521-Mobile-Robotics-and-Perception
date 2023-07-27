#!/usr/bin/env python3
#Standard Libraries
import sys
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from scipy.spatial import cKDTree
import math
import matplotlib.pyplot as plt

## Myhal SEED = 10 FOR RRT
## Myhal SEED = 16 FOR RRT*
# RRT-STAR and RRT Seed - Willow (24)
np.random.seed(24)

def load_map(filename):
    im = mpimg.imread("./maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("./maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.trajFromParent = None
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        # User Defined Variables
        self.cellRes = self.map_settings_dict["resolution"]
        self.originTheta = self.map_settings_dict["origin"][2]
        self.originXY = np.array([[self.map_settings_dict["origin"][0]],
                                  [self.map_settings_dict["origin"][1]]])

        # map dimensions (bounds the region of the actual labyrinth not the whole 'world')
        self.topL = [-3, 12]
        self.topR = [45, -49]

        #Robot information
        self.robot_radius = 0.22 #m
        ## CHOSEN TO MATCH THE follow_path.py range of inputs during trjectory rollout
        self.vel_max = 0.28 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #rad/s (Feel free to change!) 

        self.v_options = np.linspace(-self.vel_max, self.vel_max, 5)
        self.w_options = np.linspace(-self.rot_vel_max, self.rot_vel_max, 5)

        self.mapIsMyhal = False

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.5 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"]**2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2*(1 + 1/2)**(1/2)*(self.lebesgue_free / self.zeta_d)**(1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        ## FIX BUG WHEN LOADING THE MAP - pygame_utils.py was modified accordingly
        # Pygame window for visualization 
        if map_filename == 'myhal.png':
            shape = (self.occupancy_map.shape[1]*10, self.occupancy_map.shape[0]*10)
            self.rot_vel_max = np.pi/2
            self.v_options = np.linspace(0, self.vel_max, 5)
            self.w_options = np.linspace(-self.rot_vel_max, self.rot_vel_max, 5)
            self.num_substeps = 10
            self.mapIsMyhal = True
        else:
            shape = (900, 900)

        self.window = pygame_utils.PygameWindow(
            "Path Planner", shape, self.occupancy_map.T.shape, self.map_settings_dict, self.goal_point, self.stopping_dist, map_filename)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards

        probGoal = np.random.rand()

        if probGoal < 0.05:
            randX = self.goal_point[0][0] + 3*self.stopping_dist*np.random.rand()
            randY = self.goal_point[1][0] + 3*self.stopping_dist*np.random.rand()
            return np.array([[randX], [randY]])

        if not self.mapIsMyhal: 
            randX = np.random.rand()*(self.topR[0] - self.topL[0]) + self.topL[0]
            randY = np.random.rand()*(self.topR[1] - self.topL[1]) + self.topL[1]
        else:
            randX = np.random.rand()*(self.bounds[0, 1] - self.bounds[0, 0]) + self.bounds[0, 0] 
            randY = np.random.rand()*(self.bounds[1, 1] - self.bounds[1, 0]) + self.bounds[1, 0]
        
        return np.array([[randX], [randY]])
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        
        closest = self.closest_node(point)
        closestPt = self.nodes[closest].point[:2,:].reshape((2,1))

        if np.linalg.norm(point - closestPt) <= 0.1:
            return True

        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node

        dist = cKDTree(np.stack([node.point[:-1, :] for node in self.nodes], axis = 0).squeeze(-1))
        bestDist , Id = dist.query(point.T, k = 1)

        return Id[0]
    
    def simulate_trajectory(self, node_i: Node, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does have many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]

        robot_traj = None

        v, w = self.robot_controller(node_i.point, point_s)

        trajBest = self.trajectory_rollout(v, w, node_i.point)

        deltaXY = trajBest[:2,:] - point_s

        batchDist = np.linalg.norm(deltaXY, axis = 0)

        minId = np.argmin(batchDist)
        
        robot_traj = trajBest[:,:(minId + 1)]

        return robot_traj
    
    def robot_controller(self, poseS, pointE):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        # initialize output to default values
        v = 0
        w = 0
        # loop variable
        bestDist = np.inf
        # determine control law that takes us closest to end-point pointE
        for i in range(0, len(self.v_options)):
            
            v_test = self.v_options[i]

            for j in range(0,len(self.w_options)):
                
                w_test = self.w_options[j]

                traj_ij = self.trajectory_rollout(v_test, w_test, poseS)

                deltaXY = traj_ij[:2,:] - pointE

                batchDist = np.linalg.norm(deltaXY, axis = 0)

                minId = np.argmin(batchDist)
                
                if batchDist[minId] < bestDist:
                    v = v_test
                    w = w_test
                    bestDist = batchDist[minId]

        return v, w
    
    def trajectory_rollout(self, vel, rot_vel, x_0):

        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # x_k_1 = x_k + B(x_k)u_k*dt
        B = lambda x_theta : np.array([[np.cos(x_theta), 0],[np.sin(x_theta), 0],[0, 1]])

        # initialize output
        trajectory = x_0
        
        # prepare variables needed for computation
        dt = self.timestep/self.num_substeps
        u = np.array([[vel],[rot_vel]])
        # control is computed from the current pose
        # in the global reference frame
        currState = x_0
        nextState = np.zeros((3,1))

        for i in range(0, self.num_substeps):
            
            # compute B(x_k)
            B_k = B(currState[2][0])
            # compute new state
            nextState = currState + np.dot(B_k,u)*dt

            # adjust heading to be in [-pi, pi]
            if nextState[2] > np.pi:
                nextState[2] = nextState[2] - 2*np.pi
            elif nextState[2] < -np.pi:
                nextState[2] = nextState[2] + 2*np.pi

            # save current state and progress to the next timestep
            trajectory = np.hstack((trajectory, nextState))
            
            currState = nextState
        
        return trajectory
    
    def point_to_cell(self, point):

        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        
        # compute rotation matrix to map points on the reference 
        # frame if a rotation is present
        theta = self.originTheta
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [-np.sin(theta), np.cos(theta)]])
        
        # transform points to cell coordinates
        pts_origin = (np.dot(R, point) - np.dot(R, self.originXY))/self.cellRes
        
        ptsX = pts_origin[0, :]
        ptsY = self.map_shape[0] - pts_origin[1,:]
        ptsOut = np.vstack((ptsX, ptsY))

        return ptsOut.astype(int) #(2, N)

    def points_to_robot_circle(self, points):
        
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        # transform robot coordinates to cell coordinates
        cellPos = self.point_to_cell(points)
        
        # create a base region for the radius specified
        radRes = int(self.robot_radius/self.cellRes) + 1
        
        # attempt to be more conservative with Myhal map
        if self.mapIsMyhal:
            radRes += 1
        
        rr, cc = disk((0, 0), radRes)
        baseCoords = np.vstack((rr,cc), dtype = np.int32)
        
        # create result place holder
        tempX = np.zeros((2,1))
        pts2Rob = {}        
        
        for i in range(0, cellPos.shape[1]):

            # check for duplicates
            keyLoc = (cellPos[0][i], cellPos[1][i])

            if keyLoc not in pts2Rob:

                # extract current points
                tempX[0] = cellPos[0][i]
                tempX[1] = cellPos[1][i]
                # map the occupancy region
                tempDiff = tempX + baseCoords # shape of (2, 2*radRes + 1)
                # adjust for corner cases
                tempDiff[0, tempDiff[0] >= self.occupancy_map.shape[1]] = self.occupancy_map.shape[1] - 1
                tempDiff[1, tempDiff[1] >= self.occupancy_map.shape[0]] = self.occupancy_map.shape[0] - 1
                
                remCells = np.where(tempDiff < 0)[1]
                tempDiffDel = np.delete(tempDiff, remCells, axis = 1)
                # save in dictionary only unique coordinates and entries
                pts2Rob[keyLoc] = np.unique(tempDiffDel, axis = 1)

        return pts2Rob

    def checkCollision(self, points):

        # points are of shape (2, N) in discrete coordinates

        # get the occupancy map for each region
        pts2RobPoses = self.points_to_robot_circle(points) # dictionary with occupancy regions

        # check if pose is collision free
        for key in pts2RobPoses:

            # get occupancy map: (2, M) array 
            mapOcc = pts2RobPoses[key].astype(int)

            # check if any of the cells contains a value of 0 and report collision as True
            crash = np.any(self.occupancy_map[mapOcc[1,:], mapOcc[0, :]] == 0)

            if crash:
                return True

        return False

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point

        # initialize return variable
        resTraj = None
        tooClose = False

        # partition input node
        point_s = node_i[:2,0].reshape(2,1)
        robHead = node_i[2,0]

        # important info : distance and bearing angle to the robots refernce frame
        dist = np.linalg.norm(point_s - point_f)
        theta = np.arctan2(point_f[1,0] - point_s[1,0], point_f[0,0] - point_s[0,0])

        if dist <= 0.1:

            tooClose = True
            
        elif (abs(theta - robHead) < 1e-3) or ((abs(abs(theta - robHead) - np.pi) < 1e-3)):

            # choose an appropriate step resolution according to the distance
            stepNum = 10
            resTraj = np.linspace(point_s, point_f, num = stepNum)[:,:,0].T
            resTraj = np.vstack((resTraj, robHead*np.ones((1, stepNum))))
        
        else:

            # tranform point_f to robots frame through T_vw
            R_vw = np.array([[np.cos(robHead), np.sin(robHead)],
                             [-np.sin(robHead), np.cos(robHead)]])
            o_vw = - np.dot(R_vw, point_s)

            T_vw = np.hstack((R_vw, o_vw))
            T_vw = np.vstack((T_vw, np.array([[0,0,1]])))
            T_inv = np.linalg.inv(T_vw)

            # augment point_f
            point_f_w = np.vstack((point_f, np.array([[1]])))
            point_f_v = np.dot(T_vw, point_f_w)

            # compute the the center of circle by using vector S->F
            alpha = point_f_v[0,0]/point_f_v[1,0]
            xC = 0
            yC = alpha*point_f_v[0,0]/2 + point_f_v[1,0]/2
            # radius of circle
            r = abs(yC)

            # compute arc-angle
            cosRho = 1 - 2*((dist/(2*r))**2)

            # ILL-CONDITIONED PROBLEM - SKIP CONNECTION
            if abs(abs(cosRho) - 1) < 1e-6:
                return (True, None)

            rho = np.arctan2(np.sqrt(1-cosRho**2), cosRho)

            # compute arc-length
            l = rho*r
            # select an appropriate stepsize
            stepNum = np.ceil(l/self.robot_radius).astype(np.int32) + 1
            # determine direction of traversing the arc length for
            # each of the possible combinations (4 to consider)

            dirY = -1 # decreasing with each substep
            dirX = -1 # decreasing with each substep

            if yC > 0:
                dirY = 1

            if point_f_v[0,0] > 0:
                dirX = 1

            # start determining the trajectory
            dRho = rho/stepNum
            resTraj = node_i
            rhoAcc = 0

            ## FOR DEBUGGING
            # plt.figure()
            # plt.plot(0, 0, 'rX')
            # plt.plot(point_f_v[0,0], point_f_v[1,0], 'ro')
        
            for i in range(0, stepNum + 1):

                # compute next waypoint in vehicle frame
                currPt = np.array([[r*np.sin(rhoAcc)*dirX + xC],
                                   [-r*np.cos(rhoAcc)*dirY + yC],
                                   [1]])

                ## FOR DEBUGGING
                # plt.plot(currPt[0,0], currPt[1,0], 'b.')

                # transform to world frame and update heading
                tempPt = np.dot(T_inv, currPt)
                rhoAcc += dRho
                tempPt[2, 0] = robHead + rhoAcc*(dirX*dirY)

                if tempPt[2] > np.pi:
                    tempPt[2] = tempPt[2] - 2*np.pi
                elif tempPt[2] < -np.pi:
                    tempPt[2] = tempPt[2] + 2*np.pi
                
                # save result
                resTraj = np.hstack((resTraj, tempPt))

            ## FOR DEBUGGING
            # plt.close()

            
        ## FOR DEBUGGING
        # if not tooClose:
        #     plt.figure()
        #     plt.plot(node_i[0,0], node_i[1,0], 'rX')
        #     plt.plot(point_f[0,0], point_f[1,0], 'ro')

        #     for j in range(0,resTraj.shape[1]):
        #         plt.plot(resTraj[0,j], resTraj[1,j], 'b.')
        #     plt.close()

        return (tooClose, resTraj)
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle (2 x N)

        # compute total euclidean distance along the path
        result = 0

        for i in range(1, trajectory_o.shape[1]):
            result += np.linalg.norm(trajectory_o[:2, i].reshape((2,1)) - trajectory_o[:2, i-1].reshape((2,1)))

        return result
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost

        parent = self.nodes[node_id]

        # iterate through the children recursively and update the cost
        for childID in parent.children_ids:
            trajParent = self.nodes[childID].trajFromParent
            self.nodes[childID].cost = parent.cost + self.cost_to_come(trajParent)
            self.update_children(childID)

        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        
        i = 0

        while True: #Most likely need more iterations than this to complete the map!
            
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            # check how close we were able to reach our goal (we may not be able 
            # to reach the point exactly or at all)
            lastPt = trajectory_o[:,-1].reshape((3,1))
            lastPt2D = lastPt[:2,0].reshape((2,1))

            # check if a similar point is already in our tree
            checkDuplicate = self.check_if_duplicate(lastPt2D)

            if checkDuplicate:
                continue

            didCollide = self.checkCollision(trajectory_o[:2,:])

            if not didCollide:

                # add new node on our list
                newNode = Node(lastPt, closest_node_id, 0)
                newNode.trajFromParent = trajectory_o
                self.nodes.append(newNode)
                self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)

                # check if we are at goal
                if np.linalg.norm(lastPt2D - self.goal_point) < self.stopping_dist:
                    break
                
                # # FOR DEBUGGING
                # for j in range(0, trajectory_o.shape[1]):
                #    self.window.add_point(trajectory_o[:-1, j].copy())

            i += 1

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        while True: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            # check how close we were able to reach our goal (we may not be able 
            # to reach the point exactly or at all)
            lastPt = trajectory_o[:,-1].reshape((3,1))
            lastPt2D = lastPt[:2,0].reshape((2,1))

            # check if a similar point is already in our tree
            checkDuplicate = self.check_if_duplicate(lastPt2D)

            if checkDuplicate:
                continue

            didCollide = self.checkCollision(trajectory_o[:2,:])

            if not didCollide:

                # add new node on our list
                newNode = Node(lastPt, closest_node_id, 0)
                newNode.trajFromParent = trajectory_o
                newNode.cost = self.cost_to_come(trajectory_o) + self.nodes[closest_node_id].cost
                self.nodes.append(newNode)
                self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)

                # compute radius of neighbours
                rN = self.ball_radius()

                # find nearest neighbours within rN
                dist = cKDTree(np.stack([node.point[:-1, :] for node in self.nodes], axis = 0).squeeze(-1))
                Id = dist.query_ball_point(lastPt[:2,0], r = rN)

                currCost = newNode.cost
                newId = len(self.nodes) - 1
                currParentId = closest_node_id
                currTraj = trajectory_o

                ## FOR DEBUGGING
                # for j in range(0, trajectory_o.shape[1]):
                #     self.window.add_point(trajectory_o[:-1, j].copy(), color = (0,0,0))

                lowerCost = False

                #Last node rewire
                for n in Id:

                    closeFlag, tempTraj = self.connect_node_to_point(self.nodes[n].point, lastPt2D)

                    if (n == newId) or closeFlag:
                        continue

                    # check for collision
                    didCollideTemp = self.checkCollision(tempTraj[:2,:])

                    if not didCollideTemp:

                        # compute cost 
                        tempCost = self.cost_to_come(tempTraj) + self.nodes[n].cost

                        if currCost >= tempCost:
                            currCost = tempCost
                            currTraj = tempTraj
                            currParentId = n
                            #lowerCost = True
                            

                # rewire child
                if currParentId != closest_node_id or lowerCost:
                    # remove child from the list
                    self.nodes[closest_node_id].children_ids.remove(newId)
                    # update new parent data
                    self.nodes[newId].trajFromParent = currTraj
                    self.nodes[newId].cost = currCost
                    self.nodes[newId].parent_id = currParentId
                    self.nodes[currParentId].children_ids.append(newId)
                    
                    ## FOR DEBUGGING
                    # for j in range(0, currTraj.shape[1]):
                    #     self.window.add_point(currTraj[:-1, j].copy(), color = (255,0,0))

                # Close node rewire
                for n in Id:
                    
                    # temporary variable
                    tempPId = self.nodes[n].parent_id

                    closeFlag, tempTraj = self.connect_node_to_point(lastPt, self.nodes[n].point[:2,0].reshape((2,1)))

                    if closeFlag or (n == newId):
                        continue

                    # check for collision
                    didCollideTemp = self.checkCollision(tempTraj[:2,:])

                    if not didCollideTemp:

                        # compute cost and rewire if possible
                        tempCost = self.cost_to_come(tempTraj) + self.nodes[newId].cost

                        if tempCost <= self.nodes[n].cost:
                            self.nodes[tempPId].children_ids.remove(n)
                            self.nodes[n].parent_id = newId
                            self.nodes[n].cost = tempCost
                            self.nodes[n].trajFromParent = tempTraj
                            self.nodes[newId].children_ids.append(n)
                            self.update_children(n)

                            ## FOR DEBUGGING
                            # for j in range(0, currTraj.shape[1]):
                            #     self.window.add_point(currTraj[:-1, j].copy(), color = (0,0,255))
                            

                # check if we are at goal
                if np.linalg.norm(lastPt2D - self.goal_point) < self.stopping_dist:
                    break

        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    
    #Set map information
    # map_filename = "myhal.png"
    # map_setings_filename = "myhal.yaml"
    # #robot information
    # goal_point = np.array([[7], [0]]) #m

    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42], [-44]]) #m

    stopping_dist = 0.5 #m

    # HAD SOME ISSUES IN MY VM SO I INCREASED THE LIMIT
    sys.setrecursionlimit(4000)
    print("Recursion limit set to {}".format(sys.getrecursionlimit()))

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    for i in range (1, node_path_metric.shape[1]):
        pt1 = node_path_metric[:, i-1].reshape((3,1))
        pt2 = node_path_metric[:, i].reshape((3,1))
        path_planner.window.add_line(pt1[:2, 0].copy(), pt2[:2, 0].copy(), width = 3, color = (0, 255, 0))

    pygame.image.save(path_planner.window.screen, f"image.png")      

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)

if __name__ == '__main__':
    main()

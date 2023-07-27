#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
from skimage.draw import disk
import rospy
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils


TRANS_GOAL_TOL = .1  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.025, 0.13, 0.26]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .025  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions


#Map Handling Functions
def load_map(filename):
    import matplotlib.image as mpimg
    import cv2 
    im = cv2.imread("../maps/" + filename)
    im = cv2.flip(im, 0)
    # im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    im_np = np.logical_not(im_np)     #for ros
    return im_np

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        # print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        # map = rospy.wait_for_message('/map', OccupancyGrid)
        # self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        # self.map_resolution = round(map.info.resolution, 5)
        # self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        # self.map_nonzero_idxes = np.argwhere(self.map_np)
        map_filename = "myhal.png"
        occupancy_map = load_map(map_filename)
        self.map_np = occupancy_map
        self.map_resolution = 0.05
        self.map_origin = np.array([ 0.2 , 0.2 ,-0. ])
        self.map_nonzero_idxes = np.argwhere(self.map_np)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'shortest_path.npy')).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)
        self.prev_ctrl = np.array([0,0])

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def trajectory_rollout(self, vel, rot_vel, x_0):

        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # x_0 is expected as x, y, theta
        # x_k_1 = x_k + B(x_k)u_k*dt
        B = lambda x_theta : np.array([[np.cos(x_theta), 0],[np.sin(x_theta), 0],[0, 1]])

        # initialize output
        trajectory = np.zeros((3, 1))

        # prepare variables needed for computation
        u = np.array([[vel],[rot_vel]])
        dt = INTEGRATION_DT
        # control is computed from the current pose
        # in the global reference frame
        currState = np.reshape(x_0, (3,1))
        nextState = np.zeros((3,1))

        for i in range(0, self.horizon_timesteps):

            # compute B(x_k)
            B_k = B(currState[2][0])
            # compute new state
            #print(np.shape(B_k))
            #print(np.shape(u))
            nextState = currState + np.dot(B_k,u)*dt

            # print("state propagation")
            # print(currState)
            # print(nextState)

            # adjust heading to be in [-pi, pi]
            if nextState[2] > np.pi:
                nextState[2] = nextState[2] - 2*np.pi
            elif nextState[2] < -np.pi:
                nextState[2] = nextState[2] + 2*np.pi

            # save current state and progress to the next timestep
            if i == 0:
                trajectory = nextState
            else:
                trajectory = np.hstack((trajectory, nextState))

            currState = nextState

        return np.transpose(trajectory)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function

        # transform robot coordinates to cell coordinates
        # cellPos = self.point_to_cell(points)
        cellPos = points

        # create a base region for the radius specified
        radRes = int(COLLISION_RADIUS/self.map_resolution) + 1# TODO: check if self map resolution is legit
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
                tempDiff[0, tempDiff[0] >= self.map_np.shape[1]] = self.map_np.shape[1] - 1
                tempDiff[1, tempDiff[1] >= self.map_np.shape[0]] = self.map_np.shape[0] - 1
                remCells = np.where(tempDiff < 0)[1]
                tempDiffDel = np.delete(tempDiff, remCells, axis = 1)
                # save in dictionary only unique coordinates and entries
                pts2Rob[keyLoc] = np.unique(tempDiffDel, axis = 1)

        return pts2Rob

    def checkCollision(self, points_T):
        # points are of shape (2, N) in discrete coordinates
        points = np.transpose(points_T)
        # get the robot occupancy map for each set of center points
        pts2RobPoses = self.points_to_robot_circle(points) # dictionary with occupancy regions
        # dictionary with keys being center points, and data being an array of coordinates of rob occupancy

        # check if pose is collision free
        for key in pts2RobPoses:

            # get robot occupancy map: (2, M) array
            mapOcc = pts2RobPoses[key].astype(int) # coordiantes based on set passed in by "points"
            # print("occupied cells by key:{}".format(key))
            # print(mapOcc)

            # plt.scatter(-self.map_xy[0], self.map_xy[1])
            # plt.scatter(-mapOcc[1,:], mapOcc[0,:])
            # plt.scatter(-key[1], key[0], s = 100)
            # plt.scatter(-points[1,0], points[0,0])
            # plt.show()
            # exit()

            # check if any of the environment occupancy map cells occupied by the robot contains a value of 0 and report collision as True
            crash = np.any(self.map_np[mapOcc[1, :], mapOcc[0, :]] == 100) # TODO: ensure this is how map_np works
            if crash:
                return True

        return False

    def calculate_cost(self, cand_opt, end_pt):
        # 2 preferences: low dist from goal, low change from previous ctrl
        trans_factor = 16
        rot_factor = 0
        diff_rot_factor = 0

        # difference in goal poses penalized
        curr_goal = self.cur_goal
        pose_diff = np.abs(curr_goal - end_pt)
        loc_cost = trans_factor*(pose_diff[0] + pose_diff[1])
        rot_cost = rot_factor*pose_diff[2] / loc_cost

        # change in control penalzie
        curr_trans_opt = cand_opt[0]
        last_trans_opt = self.prev_ctrl[0]
        curr_rot_opt = cand_opt[1]
        last_rot_opt = self.prev_ctrl[1]
        ctrl_chg_cost = np.abs(curr_trans_opt - last_trans_opt) + diff_rot_factor*np.abs(curr_rot_opt - last_rot_opt)

        return loc_cost + rot_cost + ctrl_chg_cost # TODO: maybe implement a saturating cost for rotational error

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            t1 = rospy.Time.now()
            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)

            # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects

            for i, opts in enumerate(self.all_opts):
                trans_vel = opts[0]
                rot_vel = opts[1]
                pred_traj = self.trajectory_rollout(trans_vel, rot_vel, self.pose_in_map_np)
                local_paths[1:,i,:] = pred_traj #assign complete trajectory one at a time
                # print(local_paths[:,0,:])
            # print("start")
            # print(local_paths[0,:,:])
            # print("end pts")
            # print(local_paths[-1,:,:])

            t2 = rospy.Time.now()
            traj_rollout_time = t2 - t1

            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = np.rint((self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution).astype(int) #round to nearest int as index
            # print("local path shape")
            # print(np.shape(local_paths_pixels))
            # plt.scatter(-self.map_xy[0], self.map_xy[1])
            # plt.scatter(-local_paths_pixels[0,:,1], local_paths_pixels[0,:,0])
            # print(-local_paths_pixels[0,0,0:2])
            #
            # print(-local_paths_pixels[-1,:,0:2])
            # plt.scatter(-local_paths_pixels[-1,:,1], local_paths_pixels[-1,:,0])
            # plt.show()


            valid_opts = list(range(self.num_opts))
            invalid_opts = []
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            for opt in range(local_paths_pixels.shape[1]):
                # for timestep in range(local_paths_pixels.shape[0]):

                rob_center_pixels = local_paths_pixels[:,opt,:]

                # plt.scatter(-self.map_xy[0], self.map_xy[1])
                # plt.scatter(-rob_center_pixels[:,1], rob_center_pixels[:,0])
                # plt.show()
                # print(rob_center_pixels)
                # plt.scatter(-self.map_xy[0], self.map_xy[1])
                # plt.scatter(-rob_center_pixels[:,1], rob_center_pixels[:,0])
                # plt.show()
                # print(np.shape(rob_center_pixels))

                if self.checkCollision(rob_center_pixels): # if there is a collision anywhere along timstep
                    # print(rob_center_pixels)
                    # plt.scatter(-self.map_xy[0], self.map_xy[1])
                    # plt.scatter(-rob_center_pixels[:,1], rob_center_pixels[:,0], c='red')
                    # plt.show()
                    # exit()
                    valid_opts.remove(opt) # we immediately remove the option from the lists
            t3 = rospy.Time.now()
            # collison_det_time = t3 - t2
            # remove trajectories that were deemed to have collisions

            # calculate final cost and choose best option
            # final_cost = np.zeros(self.num_opts)
            final_cost = np.zeros(len(valid_opts))
            for i in range(0, len(valid_opts)):
                # print("checking opts")
                # print(local_paths[-1,i,:])
                cur_opt = self.all_opts_scaled[i]
                final_cost[i] = self.calculate_cost(cur_opt, local_paths[-1,valid_opts[i],:])

            if final_cost.size == 0:  # hardcoded recovery if all options have collision
            # if np.count_nonzero(final_cost) == 0:
                control = [-.1, 0]
            else:
                best_opt = valid_opts[final_cost.argmin()]
                # print("chosen_control")
                # print(final_cost.argmin())
                # print(best_opt)
                # print(self.all_opts[best_opt])
                # print(local_paths[-1,best_opt,:])
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))

            # t4 = rospy.Time.now()
            # cost_calc_time = t4 - t3
            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            # print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                #  control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))
            # print("traj rollout time: {}, collision det time: {}, cost calc time: {}".format(traj_rollout_time, collison_det_time, cost_calc_time))
            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass
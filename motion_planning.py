import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid, prune_path_bres, find_start_goal, breadth_first, a_star_bfs, create_grid_bfs
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from skimage.morphology import medial_axis
from skimage.util import invert
from math import pi

class States(Enum):
    MANUAL = auto()
    ARMING =    auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        #print('global home {0}, global position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         #self.local_position))
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 5:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def proj_min_req(self, data, global_goal, TARGET_ALTITUDE, SAFETY_DISTANCE):
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        grid_start = (int(self.local_position[0]) - north_offset, int(self.local_position[1]) - east_offset) 
        local_goal = global_to_local(global_goal, self.global_home)
        grid_goal = (int(local_goal[0]) - north_offset, int(local_goal[1]) - east_offset)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        pruned_path = prune_path_bres(path, grid)
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in pruned_path]
        for i in range(1, len(waypoints)):
            heading = np.arctan2(waypoints[i][0] - waypoints[i-1][0], waypoints[i][1] - waypoints[i-1][1]) - pi/2
            waypoints[i][3] =  -heading
        return waypoints

    def medial_axis_Astar(self, data, global_goal, TARGET_ALTITUDE, SAFETY_DISTANCE):
        grid, north_offset, east_offset = create_grid_bfs(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        skeleton = medial_axis(invert(grid))
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        grid_start = (int(self.local_position[0]) - north_offset, int(self.local_position[1]) - east_offset) 
        local_goal = global_to_local(global_goal, self.global_home)
        grid_goal = (int(local_goal[0]) - north_offset, int(local_goal[1]) - east_offset)
        skel_start, skel_goal = find_start_goal(skeleton, grid_start, grid_goal)
        mapa = breadth_first(invert(skeleton).astype(np.int), tuple(skel_goal), tuple(skel_start))
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star_bfs(invert(skeleton).astype(np.int), mapa, tuple(skel_start), tuple(skel_goal))
        path.append(grid_goal)
        pruned_path = prune_path_bres(path, grid)
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in pruned_path]
        for i in range(1, len(waypoints)):
            heading = np.arctan2(waypoints[i][0] - waypoints[i-1][0], waypoints[i][1] - waypoints[i-1][1]) - pi/2
            waypoints[i][3] =  -heading
        return waypoints

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 25
        SAFETY_DISTANCE = 5
        self.target_position[2] = TARGET_ALTITUDE
        filename = 'colliders.csv'
        firstline = open(filename).readline().replace(',','')

        l = []
        for t in firstline.split():
            try:
                l.append(float(t))
            except ValueError:
                pass
        lat0, lon0 = l

        self.set_home_position(lon0, lat0, 0)
        print('global home {0}, global position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)

        while True:      
            try:
                user_input = input("\nEnter longitude and latitude separated by commas:\nExamples:"
                            "-122.395075, 37.797183\n         -122.401044, 37.795678\n         -122.393173, 37.792542\n: ")
                input_list = user_input.split(',')
                global_goal = [float(x.strip()) for x in input_list]
                global_goal.append(0)
                print(global_goal)
                break
            except (ValueError, NameError, AttributeError) as e:
                print("{input} is not valid, please follow the format".format(input=user_input))

        waypoints = []

        while True:
            user_input = int(input("\nChoose Method: [1] Grid-A_star-EucHeuristic-Bresenham [2] Medial_Axis-A_start-BFSHeuristic-Bresenham:\n"))
            if user_input == 1:
                self.waypoints = self.proj_min_req(data, global_goal, TARGET_ALTITUDE, SAFETY_DISTANCE)
                break
            elif user_input == 2:
                self.waypoints = self.medial_axis_Astar(data, global_goal, TARGET_ALTITUDE, SAFETY_DISTANCE)
                break
            else:
                continue

        if len(self.waypoints) == 0:
            self.disarming_transition()

        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()

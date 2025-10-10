"""Utility script for publishing circular MPC waypoints for quick testing."""

from __future__ import annotations

import itertools
import math
import time
from typing import List, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

POSITION_TOL = 0.2  # metres before advancing to the next setpoint
CIRCLE_RADIUS = 1.5
CIRCLE_ALTITUDE = 1.5
CIRCLE_CENTER = (0.0, 0.0)
CIRCLE_SPEED = 1.0  # tangential speed in m/s
CIRCLE_LOOPS = 2.0
STEP_DT = 0.1  # seconds between setpoints
ODOM_TOPIC = '/hummingbird/ground_truth/odometry'
SETPOINT_TOPIC = '/mpc_controller/setpoint'


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Return quaternion (x, y, z, w) for a pure yaw rotation."""
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def generate_circle_waypoints(radius: float,
                              altitude: float,
                              center: Tuple[float, float],
                              speed: float,
                              loops: float,
                              dt: float) -> List[Tuple[float, float, float, float]]:
    """Generate (x, y, z, yaw) tuples along a planar circle."""
    radius = float(radius)
    speed = max(float(speed), 1e-3)
    loops = max(float(loops), 0.1)
    dt = max(float(dt), 1e-3)

    circumference = max(2.0 * math.pi * radius, 1e-6)
    distance_per_step = speed * dt
    steps_per_loop = max(32, int(math.ceil(circumference / distance_per_step)))
    total_steps = max(steps_per_loop, int(math.ceil(steps_per_loop * loops)))

    theta = np.linspace(0.0, 2.0 * math.pi * loops, total_steps, endpoint=False)
    cx, cy = center
    waypoints: List[Tuple[float, float, float, float]] = []
    for angle in theta:
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        yaw = angle + math.pi / 2.0  # face along direction of travel
        waypoints.append((x, y, altitude, yaw))
    return waypoints


def create_waypoint_publisher() -> None:
    rospy.init_node('circle_waypoint_publisher', anonymous=True)
    pub = rospy.Publisher(SETPOINT_TOPIC, PoseStamped, queue_size=10)
    rate = rospy.Rate(1.0 / STEP_DT)

    waypoints = generate_circle_waypoints(CIRCLE_RADIUS,
                                          CIRCLE_ALTITUDE,
                                          CIRCLE_CENTER,
                                          CIRCLE_SPEED,
                                          CIRCLE_LOOPS,
                                          STEP_DT)
    rospy.loginfo('Generated %d circle waypoints (radius=%.2f, loops=%.1f, speed=%.2f m/s)',
                  len(waypoints), CIRCLE_RADIUS, CIRCLE_LOOPS, CIRCLE_SPEED)

    waypoint_cycle = itertools.cycle(waypoints)

    while not rospy.is_shutdown():
        x, y, z, yaw = next(waypoint_cycle)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        try:
            current_pose = rospy.wait_for_message(ODOM_TOPIC, Odometry, timeout=1.0)
            dx = current_pose.pose.pose.position.x - x
            dy = current_pose.pose.pose.position.y - y
            dz = current_pose.pose.pose.position.z - z
            if math.sqrt(dx * dx + dy * dy + dz * dz) < POSITION_TOL:
                rospy.loginfo('Waypoint reached, holding for %.2f s', STEP_DT)
                time.sleep(STEP_DT)
                continue
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0, 'Waiting for odometry on %s', ODOM_TOPIC)

        rospy.loginfo('Publishing circle waypoint x=%.2f y=%.2f z=%.2f yaw=%.1f deg',
                      x, y, z, math.degrees(yaw))
        pub.publish(pose_msg)
        rate.sleep()


def main() -> None:
    try:
        create_waypoint_publisher()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time
import math

def smooth_waypoints(waypoints, smoothing_factor=0.1):
    smoothed = [waypoints[0]]
    for i in range(1, len(waypoints)):
        prev = smoothed[-1]
        curr = waypoints[i]
        new_point = (
            prev[0] + smoothing_factor * (curr[0] - prev[0]),
            prev[1] + smoothing_factor * (curr[1] - prev[1]),
            prev[2] + smoothing_factor * (curr[2] - prev[2])
        )
        smoothed.append(new_point)
    return smoothed
def create_waypoint_publisher():
    pub = rospy.Publisher('/mpc_controller/setpoint', PoseStamped, queue_size=10)
    rospy.init_node('waypoint_publisher', anonymous=True)
    rate = rospy.Rate(1)  # 1 Hz

    wpts = [
        (0.0, 0.0, 2.0),
        (0.0, 1.0, 2.0),
        (1.0, 2.0, 2.0),
        (2.0, 3.0, 2.0),
        (3.0, 4.0, 2.0),
        (0.0, 0.0, 2.0)
    ]
    waypoints = smooth_waypoints(wpts, smoothing_factor=0.8)

    def subdivide_waypoints(wpts, max_delta=1.0):
        """Insert intermediate points so that the max per-axis delta between successive
        waypoints does not exceed max_delta."""
        if not wpts:
            return []

        new_wpts = [wpts[0]]
        for idx in range(1, len(wpts)):
            start = new_wpts[-1]
            end = wpts[idx]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]
            max_dist = max(abs(dx), abs(dy), abs(dz))
            if max_dist <= max_delta:
                new_wpts.append(end)
            else:
                steps = int(math.ceil(max_dist / max_delta))
                # add intermediate points (excluding start, include end)
                for s in range(1, steps):
                    t = s / float(steps)
                    interp = (
                        start[0] + dx * t,
                        start[1] + dy * t,
                        start[2] + dz * t
                    )
                    new_wpts.append(interp)
                new_wpts.append(end)
        return new_wpts

    expanded_waypoints = subdivide_waypoints(waypoints, max_delta=1.0)

    print("Starting waypoint publisher...")
    print("Original waypoints:", waypoints)
    print("Expanded waypoints (after subdivision):", expanded_waypoints)

    # while not rospy.is_shutdown():
    for wp in expanded_waypoints:
        pose = PoseStamped()
        current_pose = rospy.wait_for_message('/hummingbird/ground_truth/odometry', Odometry)
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        pose.pose.position.x = wp[0]
        pose.pose.position.y = wp[1]
        pose.pose.position.z = wp[2]
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0  # No rotation
        if (abs(current_pose.pose.pose.position.x - wp[0]) < 0.1 and
            abs(current_pose.pose.pose.position.y - wp[1]) < 0.1 and
            abs(current_pose.pose.pose.position.z - wp[2]) < 0.1):
            rospy.loginfo(f"Reached waypoint")  # Wait for 2 seconds at the waypoint
            time.sleep(2)  # Pause for 2 seconds at the waypoint
            continue
        time.sleep(1)  # Simulate time taken to reach the waypoint
        rospy.loginfo(f"Publishing waypoint")
        pub.publish(pose)
            
def main():
    try:
        create_waypoint_publisher()
    except rospy.ROSInterruptException:
        pass        
if __name__ == '__main__':
    main()


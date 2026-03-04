#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class SafetyController(Node):
    """
    Monitors LIDAR and publishes a stop command to the safety mux input when
    an obstacle is within the TTC threshold. Stays silent when safe — the mux
    then lets the lower-priority navigation command through.

    Real car topics:
      scan_topic:  /scan
      drive_topic: /vesc/low_level/input/safety

    Simulation topics:
      scan_topic:  /scan
      drive_topic: /drive
    """

    def __init__(self):
        super().__init__("safety_controller")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/low_level/input/safety")
        self.declare_parameter("ttc_threshold", 0.3)       # seconds
        self.declare_parameter("assumed_speed", 1.0)       # m/s — used for TTC when speed unknown
        self.declare_parameter("cone_half_angle_deg", 25.0)
        self.declare_parameter("car_half_width", 0.25)     # meters
        self.declare_parameter("safe_scan_count", 3)       # hysteresis

        self.SCAN_TOPIC = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.TTC_THRESHOLD = self.get_parameter("ttc_threshold").get_parameter_value().double_value
        self.ASSUMED_SPEED = self.get_parameter("assumed_speed").get_parameter_value().double_value
        self.CONE_HALF_ANGLE = np.deg2rad(
            self.get_parameter("cone_half_angle_deg").get_parameter_value().double_value
        )
        self.CAR_HALF_WIDTH = self.get_parameter("car_half_width").get_parameter_value().double_value
        self.SAFE_SCAN_COUNT = self.get_parameter("safe_scan_count").get_parameter_value().integer_value

        self.is_stopped = False
        self.consecutive_safe_scans = 0

        self.scan_sub = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.scan_callback, 1
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.DRIVE_TOPIC, 1
        )

        self.get_logger().info(
            f"SafetyController: scan={self.SCAN_TOPIC}, output={self.DRIVE_TOPIC} | "
            f"TTC<{self.TTC_THRESHOLD}s @ {self.ASSUMED_SPEED}m/s, "
            f"cone=±{np.rad2deg(self.CONE_HALF_ANGLE):.0f}°, "
            f"car_half_width={self.CAR_HALF_WIDTH}m, hysteresis={self.SAFE_SCAN_COUNT} scans"
        )

    def _median_filter(self, ranges, window=3):
        """Sliding-window median, ignoring inf so noise spikes don't corrupt neighbors."""
        n = len(ranges)
        filtered = np.empty(n, dtype=np.float32)
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            chunk = ranges[lo:hi]
            finite_vals = chunk[np.isfinite(chunk)]
            filtered[i] = float(np.median(finite_vals)) if len(finite_vals) > 0 else np.inf
        return filtered

    def scan_callback(self, scan_msg: LaserScan):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isnan(ranges), np.inf, ranges)   # nan → safe (no return)
        ranges = np.where(ranges < 0.05, np.inf, ranges)      # sensor noise floor → safe
        ranges = self._median_filter(ranges, window=3)

        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        angles = angle_min + np.arange(len(ranges)) * angle_inc

        # Forward cone
        cone_mask = (angles >= -self.CONE_HALF_ANGLE) & (angles <= self.CONE_HALF_ANGLE)
        cone_ranges = ranges[cone_mask]
        cone_angles = angles[cone_mask]

        # Project onto car axes
        forward_dist = cone_ranges * np.cos(cone_angles)
        lateral_dist = np.abs(cone_ranges * np.sin(cone_angles))

        # Only beams that pass through the car's body width matter
        valid = (
            (lateral_dist < self.CAR_HALF_WIDTH)
            & np.isfinite(cone_ranges)
            & (forward_dist > 0.0)
        )

        danger_detected = False
        if np.any(valid):
            min_forward = float(np.min(forward_dist[valid]))
            ttc = min_forward / max(self.ASSUMED_SPEED, 0.01)
            if ttc < self.TTC_THRESHOLD:
                danger_detected = True
                self.get_logger().warn(
                    f"SAFETY STOP: obstacle at {min_forward:.2f}m, "
                    f"TTC={ttc:.2f}s < {self.TTC_THRESHOLD}s",
                    throttle_duration_sec=0.5,
                )

        # Update hysteresis state
        if danger_detected:
            self.consecutive_safe_scans = 0
            self.is_stopped = True
        elif self.is_stopped:
            self.consecutive_safe_scans += 1
            if self.consecutive_safe_scans >= self.SAFE_SCAN_COUNT:
                self.is_stopped = False
                self.get_logger().info("Safety released — path clear.")

        # Publish stop only when needed; silence = mux falls through to navigation
        if self.is_stopped:
            stop = AckermannDriveStamped()
            stop.drive.speed = 0.0
            stop.drive.steering_angle = 0.0
            self.drive_pub.publish(stop)


def main():
    rclpy.init()
    node = SafetyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

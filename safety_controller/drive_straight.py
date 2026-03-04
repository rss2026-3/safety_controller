#!/usr/bin/env python3
"""
Simple node that publishes a constant forward drive command at 20 Hz.
Used for brick testing: gives the safety controller something to override.

Real car:  drive_topic = /vesc/input/navigation
Sim:       drive_topic = /drive
"""
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class DriveStraight(Node):

    def __init__(self):
        super().__init__("drive_straight")

        self.declare_parameter("drive_topic", "/vesc/input/navigation")
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("steering_angle", 0.0)

        drive_topic = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.speed = self.get_parameter("speed").get_parameter_value().double_value
        self.steering = self.get_parameter("steering_angle").get_parameter_value().double_value

        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.get_logger().info(
            f"DriveStraight: publishing {self.speed}m/s to {drive_topic}"
        )

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DriveStraight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

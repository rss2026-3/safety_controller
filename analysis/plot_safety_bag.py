#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_SPEED_TOPIC = "/vesc/low_level/ackermann_cmd"
DEFAULT_TRIGGER_TOPIC = "/safety/ttc_triggered"
DEFAULT_TTC_TOPIC = "/safety/ttc"


def read_bag_topics(bag_path, topics_of_interest):
    """
    Read selected topics from a ROS 2 bag.

    Returns:
        data: dict mapping topic -> list of (timestamp_sec, deserialized_msg)
    """
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}

    missing = [topic for topic in topics_of_interest if topic not in type_map]
    if missing:
        print("Warning: these topics were not found in the bag:")
        for topic in missing:
            print(f"  {topic}")

    msg_type_map = {
        topic: get_message(type_map[topic])
        for topic in topics_of_interest
        if topic in type_map
    }

    data = {topic: [] for topic in topics_of_interest}

    while reader.has_next():
        topic, raw_data, timestamp_ns = reader.read_next()

        if topic not in msg_type_map:
            continue

        msg = deserialize_message(raw_data, msg_type_map[topic])
        timestamp_sec = timestamp_ns * 1e-9
        data[topic].append((timestamp_sec, msg))

    return data


def extract_speed_series(speed_msgs):
    """
    Extract time and speed arrays from AckermannDriveStamped messages.
    """
    times = []
    speeds = []

    for t, msg in speed_msgs:
        times.append(t)
        speeds.append(msg.drive.speed)

    return np.array(times), np.array(speeds)


def extract_bool_series(bool_msgs):
    """
    Extract time and boolean arrays from std_msgs/Bool messages.
    """
    times = []
    vals = []

    for t, msg in bool_msgs:
        times.append(t)
        vals.append(bool(msg.data))

    return np.array(times), np.array(vals, dtype=bool)


def extract_float_series(float_msgs):
    """
    Extract time and float arrays from std_msgs/Float32 messages.
    """
    times = []
    vals = []

    for t, msg in float_msgs:
        times.append(t)
        vals.append(float(msg.data))

    return np.array(times), np.array(vals, dtype=float)


def find_rising_edges(times, trigger_vals):
    """
    Return timestamps where trigger goes False -> True.
    """
    if len(times) == 0:
        return np.array([])

    rising_times = []
    prev = False

    for t, val in zip(times, trigger_vals):
        if val and not prev:
            rising_times.append(t)
        prev = val

    return np.array(rising_times)


def normalize_time(*arrays):
    """
    Shift all non-empty time arrays so earliest timestamp becomes 0.
    """
    non_empty = [arr for arr in arrays if len(arr) > 0]
    if not non_empty:
        return arrays

    t0 = min(arr[0] for arr in non_empty)
    return tuple(arr - t0 for arr in arrays)


def main():
    parser = argparse.ArgumentParser(description="Plot speed vs time from a ROS 2 bag.")
    parser.add_argument("bag_path", help="Path to rosbag directory")
    parser.add_argument("--speed-topic", default=DEFAULT_SPEED_TOPIC)
    parser.add_argument("--trigger-topic", default=DEFAULT_TRIGGER_TOPIC)
    parser.add_argument("--ttc-topic", default=DEFAULT_TTC_TOPIC)
    parser.add_argument(
        "--no-ttc",
        action="store_true",
        help="Do not plot TTC values, even if topic exists",
    )
    args = parser.parse_args()

    topics = [args.speed_topic, args.trigger_topic]
    if not args.no_ttc:
        topics.append(args.ttc_topic)

    data = read_bag_topics(args.bag_path, topics)

    speed_msgs = data.get(args.speed_topic, [])
    trigger_msgs = data.get(args.trigger_topic, [])
    ttc_msgs = data.get(args.ttc_topic, []) if not args.no_ttc else []

    if len(speed_msgs) == 0:
        raise RuntimeError(f"No messages found for speed topic: {args.speed_topic}")

    speed_t, speed_v = extract_speed_series(speed_msgs)

    trigger_t = np.array([])
    trigger_vals = np.array([], dtype=bool)
    if len(trigger_msgs) > 0:
        trigger_t, trigger_vals = extract_bool_series(trigger_msgs)

    ttc_t = np.array([])
    ttc_vals = np.array([])
    if len(ttc_msgs) > 0:
        ttc_t, ttc_vals = extract_float_series(ttc_msgs)

        # Optional cleanup if you used -1.0 as "invalid TTC"
        ttc_vals = np.where(ttc_vals < 0.0, np.nan, ttc_vals)

    # Normalize all times to start at 0
    speed_t, trigger_t, ttc_t = normalize_time(speed_t, trigger_t, ttc_t)

    # Find TTC activation events
    activation_times = find_rising_edges(trigger_t, trigger_vals)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Speed trace
    ax1.plot(speed_t, speed_v, label="Commanded speed")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_title("Speed vs Time with TTC Activation")

    # Vertical lines for TTC activation
    first_line = True
    for t in activation_times:
        if first_line:
            ax1.axvline(t, linestyle="--", label="TTC activated")
            first_line = False
        else:
            ax1.axvline(t, linestyle="--")

    # Optional TTC trace on second axis
    if len(ttc_t) > 0:
        ax2 = ax1.twinx()
        ax2.plot(ttc_t, ttc_vals, alpha=0.7, label="TTC")
        ax2.set_ylabel("TTC (s)")

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax1.legend(loc="best")

    ax1.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
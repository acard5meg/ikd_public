#!/usr/bin/env python3

import numpy as np
import math
import torch
from geometry_msgs.msg import Quaternion, Pose
import rospy


# Class for general functions
class RobotUtilities:
    def __init__(self):
        self.queue_size = 0

    def rmap(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value

    def quaternion_to_yaw(self, quaternion):
        # Convert quaternion to yaw angle (in radians)
        quaternion_norm = math.sqrt(
            quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2
        )
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
        )

        return yaw

    def quaternion_to_roll(self, quaternion):
        # Convert quaternion to roll angle (in radians)
        quaternion_norm = math.sqrt(
            quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2
        )
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        roll = math.atan2(
            2.0 * (quaternion.y * quaternion.z + quaternion.w * quaternion.x),
            1.0 - 2.0 * (quaternion.x**2 + quaternion.y**2),
        )

        return roll

    def quaternion_to_pitch(self, quaternion):
        # Convert quaternion to pitch angle (in radians)
        quaternion_norm = math.sqrt(
            quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2
        )
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        pitch = math.asin(
            2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        )

        return pitch

    def yaw_to_quaternion(self, yaw):
        # Convert yaw angle (in radians) to quaternion
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)

        return quaternion

    def quat_to_yaw(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a batch quaternion into a yaw angle
        yaw is rotation around z in radians (counterclockwise)
        """
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(t3, t4)

    def quaternion_to_rpy(self, quaternion):
        # Convert quaternion to roll, pitch, and yaw angles
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def rpy_to_quaternion(self, roll, pitch, yaw):
        # Convert roll, pitch, and yaw angles to quaternion
        quaternion = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion.x = sr * cp * cy - cr * sp * sy
        quaternion.y = cr * sp * cy + sr * cp * sy
        quaternion.z = cr * cp * sy - sr * sp * cy
        quaternion.w = cr * cp * cy + sr * sp * sy

        return quaternion

    def clamp_angle(self, angles):
        angles += np.pi
        angles %= 2 * np.pi
        angles -= np.pi
        return angles

    def clamp_angle_tensor_(self, angles):
        angles += np.pi
        torch.remainder(angles, 2 * np.pi, out=angles)
        angles -= np.pi
        return angles

    def get_dist(self, start_pose, goal_pose):
        return math.sqrt(
            (goal_pose.position.x - start_pose.position.x) ** 2
            + (goal_pose.position.y - start_pose.position.y) ** 2
        )

    def map_value(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value

    def euler_to_rotation_matrix(self, euler_angles):
        """Convert Euler angles to a rotation matrix"""
        # Compute sin and cos for Euler angles
        cos = torch.cos(euler_angles)
        sin = torch.sin(euler_angles)
        zero = torch.zeros_like(euler_angles[:, 0])
        one = torch.ones_like(euler_angles[:, 0])
        # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
        R_x = torch.stack(
            [one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]],
            dim=1,
        ).view(-1, 3, 3)
        R_y = torch.stack(
            [cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]],
            dim=1,
        ).view(-1, 3, 3)
        R_z = torch.stack(
            [cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one],
            dim=1,
        ).view(-1, 3, 3)

        return torch.matmul(torch.matmul(R_z, R_y), R_x)

    def extract_euler_angles_from_se3_batch(self, tf3_matx):
        # Validate input shape
        if tf3_matx.shape[1:] != (4, 4):
            raise ValueError("Input tensor must have shape (batch, 4, 4)")

        # Extract rotation matrices
        rotation_matrices = tf3_matx[:, :3, :3]

        # Initialize tensor to hold Euler angles
        batch_size = tf3_matx.shape[0]
        euler_angles = torch.zeros(
            (batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype
        )

        # Compute Euler angles
        euler_angles[:, 0] = torch.atan2(
            rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2]
        )  # Roll
        euler_angles[:, 1] = torch.atan2(
            -rotation_matrices[:, 2, 0],
            torch.sqrt(
                rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2
            ),
        )  # Pitch
        euler_angles[:, 2] = torch.atan2(
            rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0]
        )  # Yaw

        return euler_angles

    def to_robot_torch(self, pose_batch1, pose_batch2):
        if not isinstance(pose_batch1, torch.Tensor):
            pose_batch1 = torch.tensor(pose_batch1).float()
        if not isinstance(pose_batch2, torch.Tensor):
            pose_batch2 = torch.tensor(pose_batch2).float()

        if pose_batch1.shape != pose_batch2.shape:
            raise ValueError("Input tensors must have same shape")

        if pose_batch1.shape[-1] != 6:
            raise ValueError("Input tensors must have last dim equal to 6")

        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = pose_batch1.shape[0]
        ones = torch.ones_like(pose_batch2[:, 0])
        transform = torch.zeros_like(pose_batch1)
        T1 = torch.zeros(
            (batch_size, 4, 4), device=pose_batch1.device, dtype=pose_batch1.dtype
        )
        T2 = torch.zeros(
            (batch_size, 4, 4), device=pose_batch2.device, dtype=pose_batch2.dtype
        )

        T1[:, :3, :3] = self.euler_to_rotation_matrix(pose_batch1[:, 3:])
        T2[:, :3, :3] = self.euler_to_rotation_matrix(pose_batch2[:, 3:])
        T1[:, :3, 3] = pose_batch1[:, :3]
        T2[:, :3, 3] = pose_batch2[:, :3]
        T1[:, 3, 3] = 1
        T2[:, 3, 3] = 1

        T1_inv = torch.inverse(T1)
        tf3_mat = torch.matmul(T2, T1_inv)

        transform[:, :3] = torch.matmul(
            T1_inv,
            torch.cat((pose_batch2[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2),
        ).squeeze()[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(tf3_mat)
        return transform

    def to_world_torch(self, Robot_frame, P_relative):
        if not isinstance(Robot_frame, torch.Tensor):
            Robot_frame = torch.tensor(Robot_frame).float()
        if not isinstance(P_relative, torch.Tensor):
            P_relative = torch.tensor(P_relative).float()

        if Robot_frame.shape != P_relative.shape:
            raise ValueError("Input tensors must have same shape")

        if Robot_frame.shape[-1] != 6:
            raise ValueError("Input tensors must have last dim equal to 6")

        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros(
            (batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype
        )
        T2 = torch.zeros(
            (batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype
        )

        R1 = self.euler_to_rotation_matrix(Robot_frame[:, 3:])
        R2 = self.euler_to_rotation_matrix(P_relative[:, 3:])

        T1[:, :3, :3] = R1
        T2[:, :3, :3] = R2
        T1[:, :3, 3] = Robot_frame[:, :3]
        T2[:, :3, 3] = P_relative[:, :3]
        T1[:, 3, 3] = 1
        T2[:, 3, 3] = 1

        T_tf = torch.matmul(T2, T1)

        transform[:, :3] = torch.matmul(
            T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)
        ).squeeze()[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(T_tf)
        return transform

    def ackermann_model(self, input):
        """
        Calculates the change in pose (x, y, theta) for a batch of vehicles using the Ackermann steering model.

        Parameters:
        velocity (torch.Tensor): Tensor of shape (batch_size,) representing the velocity of each vehicle.
        steering (torch.Tensor): Tensor of shape (batch_size,) representing the steering angle of each vehicle.
        wheelbase (float): The distance between the front and rear axles of the vehicles.
        dt (float): Time step for the simulation.

        Returns:
        torch.Tensor: Tensor of shape (batch_size, 3) representing the change in pose (dx, dy, dtheta) for each vehicle.
        """
        # Ensure the velocity and steering tensors have the same batch size

        velocity = input[:, 0] / 0.75
        steering = -input[:, 1] * 0.6
        wheelbase = 0.320
        dt = 0.1

        # Calculate the change in orientation (dtheta)
        dtheta = velocity / wheelbase * torch.tan(steering) * dt

        # Calculate change in x and y coordinates
        dx = velocity * torch.cos(dtheta) * dt
        dy = velocity * torch.sin(dtheta) * dt

        # Stack the changes in x, y, and theta into one tensor
        pose_change = torch.stack(
            (dx, dy, dx.clone() * 0, dx.clone() * 0, dx.clone() * 0, dtheta), dim=1
        )

        return pose_change

    def to_robot_se2(self, p1_batch, p2_batch):
        # Ensure the inputs are tensors
        p1_batch = torch.tensor(p1_batch, dtype=torch.float32)
        p2_batch = torch.tensor(p2_batch, dtype=torch.float32)

        # Validate inputs
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[-1] != 3:
            raise ValueError(
                "Both batches must be of the same shape and contain 3 elements per pose"
            )

        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack(
            [
                torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                torch.stack([torch.sin(theta1), torch.cos(theta1), y1]),
                torch.stack([zeros, zeros, ones]),
            ],
            dim=-1,
        ).permute(1, 2, 0)

        T2 = torch.stack(
            [
                torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                torch.stack([torch.sin(theta2), torch.cos(theta2), y2]),
                torch.stack([zeros, zeros, ones]),
            ],
            dim=-1,
        ).permute(1, 2, 0)

        # Inverse of T1 and transformation
        T1_inv = torch.inverse(T1)
        tf2_mat = torch.matmul(T2, T1_inv)

        # Extract transformed positions and angles
        transform = torch.matmul(
            T1_inv, torch.cat((p2_batch[:, :2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)
        ).squeeze()
        transform[:, 2] = torch.atan2(tf2_mat[:, 1, 0], tf2_mat[:, 0, 0])

        return transform

    def to_world_se2(self, p1_batch, p2_batch):
        # # Ensure the inputs are tensors
        # p1_batch = torch.tensor(p1_batch, dtype=torch.float32)
        # p2_batch = torch.tensor(p2_batch, dtype=torch.float32)

        # Validate inputs
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[-1] != 3:
            raise ValueError(
                "Both batches must be of the same shape and contain 3 elements per pose"
            )

        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack(
            [
                torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                torch.stack([torch.sin(theta1), torch.cos(theta1), y1]),
                torch.stack([zeros, zeros, ones]),
            ],
            dim=-1,
        ).permute(1, 2, 0)

        T2 = torch.stack(
            [
                torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                torch.stack([torch.sin(theta2), torch.cos(theta2), y2]),
                torch.stack([zeros, zeros, ones]),
            ],
            dim=-1,
        ).permute(1, 2, 0)

        # Inverse of T1 and transformation
        T_tf = torch.matmul(T2, T1)

        # Extract transformed positions and angles
        transform = torch.matmul(
            T1, torch.cat((p2_batch[:, :2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)
        ).squeeze()
        transform[:, 2] = torch.atan2(T_tf[:, 1, 0], T_tf[:, 0, 0])

        return transform

    def pose_difference(self, pose):
        pose_diff = torch.zeros(size=(len(pose), 6))
        pose_diff[1:] = self.to_robot_torch(pose[:-1], pose[1:])
        return pose_diff.tolist()

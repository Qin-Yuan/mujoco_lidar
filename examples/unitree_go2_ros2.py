import os
import subprocess
import signal
import mujoco
import mujoco.viewer as viewer
import argparse
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
import time


import rclpy
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

from unitree_go2 import OnnxController, _ONNX_DIR, _MJCF_PATH, _JOINT_NUM

class OnnxControllerRos2(OnnxController, Node):
    """ONNX controller for the Go-2 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        lidar_type: str = "mid360",
        stand: bool = False,
    ):
        super().__init__(
            mj_model,
            policy_path,
            default_angles,
            n_substeps,
            action_scale,
            lidar_type,
            stand,
        )
        Node.__init__(self, 'go2_node')

        self.init_topic_publisher()

    def init_topic_publisher(self):
        self.last_pub_time_tf = -1.
        self.pub_staticc_tf_once = False

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.lidar_puber = self.create_publisher(PointCloud2, '/sensor/lidar/pointcloud', 1)
        self.imu_pub = self.create_publisher(Imu, '/sensor/lidar/imu', 10)

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.last_pub_time_lidar = -1.
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # 创建ROS2 PointCloud2消息
        pc_msg = PointCloud2()
        pc_msg.header.frame_id = "lidar"
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12  # 3 个 float32 (x,y,z)
        pc_msg.height = 1
        pc_msg.is_dense = True
        self.pc_msg = pc_msg

    def get_site_tmat(self, mj_data, site_name):
        tmat = np.eye(4)
        tmat[:3,:3] = mj_data.site(site_name).xmat.reshape((3,3))
        tmat[:3,3] = mj_data.site(site_name).xpos
        return tmat

    def update_ros2(self, mj_data: mujoco.MjData) -> None:
        time_stamp = self.get_clock().now().to_msg()
        if not self.pub_staticc_tf_once:
            self.pub_staticc_tf_once = True
            self.publish_static_transform(mj_data, 'base_link', 'lidar')
            self.publish_static_transform(mj_data, 'base_link', 'imu')
            # self.publish_static_transform(mj_data, 'imu', 'lidar')
        # self.publish_tf(mj_data, time_stamp)
        self.publish_imu(mj_data, time_stamp)
        self.publish_lidar(mj_data, time_stamp)

    def publish_static_transform(self, mj_data, header_frame_id, child_frame_id):
        stfs_msg = TransformStamped()
        stfs_msg.header.stamp = self.get_clock().now().to_msg()
        stfs_msg.header.frame_id = header_frame_id
        stfs_msg.child_frame_id = child_frame_id

        tmat_base = self.get_site_tmat(mj_data, header_frame_id)
        tmat_child = self.get_site_tmat(mj_data, child_frame_id)
        tmat_trans = np.linalg.inv(tmat_base) @ tmat_child
        
        stfs_msg.transform.translation.x = tmat_trans[0, 3]
        stfs_msg.transform.translation.y = tmat_trans[1, 3]
        stfs_msg.transform.translation.z = tmat_trans[2, 3]

        quat = Rotation.from_matrix(tmat_trans[:3, :3]).as_quat()
        stfs_msg.transform.rotation.x = quat[0]
        stfs_msg.transform.rotation.y = quat[1]
        stfs_msg.transform.rotation.z = quat[2]
        stfs_msg.transform.rotation.w = quat[3]
        # print(child_frame_id)
        time.sleep(0.1)
        self.static_broadcaster.sendTransform(stfs_msg)

    def publish_tf(self, mj_data, time_stamp):
        if self.last_pub_time_tf > mj_data.time:
            self.last_pub_time_tf = mj_data.time
            return
        if mj_data.time - self.last_pub_time_tf < 1. / 10.:
            return
        self.last_pub_time_tf = mj_data.time

        trans_msg = TransformStamped()
        trans_msg.header.stamp = time_stamp
        trans_msg.header.frame_id = "odom"
        trans_msg.child_frame_id = "imu"
        trans_msg.transform.translation.x = mj_data.sensor("global_position").data[0]
        trans_msg.transform.translation.y = mj_data.sensor("global_position").data[1]
        trans_msg.transform.translation.z = mj_data.sensor("global_position").data[2]
        trans_msg.transform.rotation.w = mj_data.sensor("orientation").data[0]
        trans_msg.transform.rotation.x = mj_data.sensor("orientation").data[1]
        trans_msg.transform.rotation.y = mj_data.sensor("orientation").data[2]
        trans_msg.transform.rotation.z = mj_data.sensor("orientation").data[3]
        self.tf_broadcaster.sendTransform(trans_msg)

    def publish_lidar(self, mj_data, time_stamp):
        if self.last_pub_time_lidar > mj_data.time:
            self.last_pub_time_lidar = mj_data.time
            return
        if mj_data.time - self.last_pub_time_lidar < 1. / 10.:
            return
        self.last_pub_time_lidar = mj_data.time

        if self.dynamic_lidar:
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        self.lidar.trace_rays(mj_data, self.rays_theta, self.rays_phi)
        points = self.lidar.get_hit_points()

        self.pc_msg.header.stamp = time_stamp
        self.pc_msg.row_step = self.pc_msg.point_step * points.shape[0]
        self.pc_msg.width = points.shape[0]
        self.pc_msg.data = points.tobytes()

        self.lidar_puber.publish(self.pc_msg)

    def publish_imu(self, mj_data, time_stamp):
        imu_msg = Imu()
        imu_msg.header.stamp = time_stamp
        imu_msg.header.frame_id = "imu"

        # orientation: framequat (world -> imu)
        q = mj_data.sensor("orientation").data
        imu_msg.orientation.w = q[0]
        imu_msg.orientation.x = q[1]
        imu_msg.orientation.y = q[2]
        imu_msg.orientation.z = q[3]

        # angular velocity: gyro (imu frame)
        w = mj_data.sensor("gyro").data
        imu_msg.angular_velocity.x = w[0]
        imu_msg.angular_velocity.y = w[1]
        imu_msg.angular_velocity.z = w[2]

        # linear acceleration: accelerometer (imu frame, 含重力)
        a = mj_data.sensor("accelerometer").data
        # qys: fast_lio2 会自动缩放 9.8
        scale = 9.80
        imu_msg.linear_acceleration.x = a[0] / scale
        imu_msg.linear_acceleration.y = a[1] / scale
        imu_msg.linear_acceleration.z = a[2] / scale

        self.imu_pub.publish(imu_msg)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        rclpy.spin_once(self, timeout_sec=0.0)

        self.update_ros2(data)
        super().get_control(model, data)

    def cmd_vel_callback(self, msg: Twist):
        self.set_control(msg.linear.x, msg.linear.y, msg.angular.z)

def load_callback(model=None, data=None):
    global args
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxControllerRos2(
        model,
        policy_path=(_ONNX_DIR / "go2_policy.onnx").as_posix(),
        default_angles=np.array(model.keyframe("home").qpos[7:7+_JOINT_NUM]),
        n_substeps=n_substeps,
        action_scale=0.5,
        lidar_type=args.lidar,
        stand=args.stand,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与Unitree Go2 ROS2集成')
    parser.add_argument('--lidar', type=str, default='mid360', help='LiDAR型号 (airy, mid360)', choices=['airy', 'mid360'])
    parser.add_argument('--stand', default=True, action='store_true', help='是否静止显示')
    args = parser.parse_args()

    rclpy.init()

    # print("=" * 60)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    # cmd = f"rviz2 -d {folder_path}/./config/go2.rviz"
    cmd = f""
    # print(f"正在启动rviz2可视化:\n{cmd}")
    # print("=" * 60)
    
    # 启动 rviz2 进程
    rviz_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    
    try:
        viewer.launch(loader=load_callback)
    except:
        traceback.print_exc()
    finally:
        # 关闭 rviz2 进程
        print("正在关闭 rviz2 进程...")
        try:
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGTERM)
            rviz_process.wait(timeout=5)
            print("rviz2 进程已关闭")
        except:
            print("强制关闭 rviz2 进程...")
            os.killpg(os.getpgid(rviz_process.pid), signal.SIGKILL)
            print("rviz2 进程已强制关闭")

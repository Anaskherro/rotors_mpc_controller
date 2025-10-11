#!/usr/bin/env python3

"""RQT panel for commanding setpoints and services in the Rotors MPC stack."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry   #type: ignore
from python_qt_binding.QtCore import Qt, QTimer #type: ignore
from python_qt_binding.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox, # type: ignore
                                         QHBoxLayout, QLabel, QLineEdit,
                                         QPushButton, QVBoxLayout, QWidget) 
from qt_gui.plugin import Plugin
from std_srvs.srv import SetBool


@dataclass
class VehicleState:
    position: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return 0.0, 0.0, 0.0
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class MPCControlPlugin(Plugin):
    """Interactive control panel for the MPC controller."""

    def __init__(self, context) -> None:
        super().__init__(context)
        self.setObjectName('MPCControlPlugin')

        self._widget = QWidget()
        self._widget.setObjectName('RotorsMPCControlPanel')
        self._widget.setWindowTitle('Rotors MPC Control')

        self._frame_id_input = QLineEdit('world', self._widget)
        self._setpoint_topic_input = QLineEdit('/mpc_controller/setpoint', self._widget)
        self._odom_topic_input = QLineEdit('/hummingbird/ground_truth/odometry', self._widget)
        self._prepare_service_input = QLineEdit('/mpc_controller_node/prepare_trajectory', self._widget)
        self._start_service_input = QLineEdit('/mpc_controller_node/start_trajectory', self._widget)

        self._pos_inputs = [self._make_spin_box(-50.0, 50.0) for _ in range(3)]
        self._pos_inputs[2].setMinimum(-5.0)
        self._pos_inputs[2].setMaximum(20.0)

        # Orientation is expressed in degrees to simplify manual entry.
        self._rpy_inputs = [self._make_spin_box(-180.0, 180.0, decimals=2) for _ in range(3)]

        self._publish_button = QPushButton('Publish Setpoint', self._widget)
        self._prepare_button = QPushButton('Prepare Trajectory', self._widget)
        self._start_button = QPushButton('Start Trajectory', self._widget)
        self._stop_button = QPushButton('Stop Trajectory', self._widget)

        self._position_labels = [QLabel('0.00', self._widget) for _ in range(3)]
        self._orientation_labels = [QLabel('0.00', self._widget) for _ in range(3)]
        for lbl in self._position_labels + self._orientation_labels:
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._status_label = QLabel('Waiting for data...', self._widget)
        self._status_label.setWordWrap(True)

        self._build_layout()

        if context is not None:
            if context.serial_number() > 1:
                title = f'{self._widget.windowTitle()} ({context.serial_number()})'
                self._widget.setWindowTitle(title)
            context.add_widget(self._widget)

        self._state_lock = threading.Lock()
        self._latest_state: Optional[VehicleState] = None
        self._odom_sub: Optional[rospy.Subscriber] = None
        self._setpoint_pub: Optional[rospy.Publisher] = None
        self._prepare_client: Optional[rospy.ServiceProxy] = None
        self._start_client: Optional[rospy.ServiceProxy] = None
        self._setpoint_topic_name: Optional[str] = None
        self._odom_topic_name: Optional[str] = None
        self._prepare_service_name: Optional[str] = None
        self._start_service_name: Optional[str] = None

        self._refresh_timer = QTimer(self._widget)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.timeout.connect(self._update_display)
        self._refresh_timer.start()

        self._publish_button.clicked.connect(self._handle_publish)
        self._prepare_button.clicked.connect(self._handle_prepare)
        self._start_button.clicked.connect(self._handle_start)
        self._stop_button.clicked.connect(self._handle_stop)

        # Connections are established lazily to allow adjusting topics before publishing.
        self._connections_ready = False
        self._connection_retry_ms = 1000
        QTimer.singleShot(0, self._attempt_initial_connection)

    def _make_spin_box(self, minimum: float, maximum: float, *, decimals: int = 3) -> QDoubleSpinBox:
        box = QDoubleSpinBox(self._widget)
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setSingleStep(0.1)
        return box

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        topic_box = QGroupBox('ROS Interfaces', self._widget)
        topic_form = QFormLayout(topic_box)
        topic_form.addRow('Frame', self._frame_id_input)
        topic_form.addRow('Setpoint topic', self._setpoint_topic_input)
        topic_form.addRow('Odometry topic', self._odom_topic_input)
        topic_form.addRow('Prepare service', self._prepare_service_input)
        topic_form.addRow('Start service', self._start_service_input)

        input_box = QGroupBox('Command Setpoint', self._widget)
        input_layout = QFormLayout(input_box)

        pos_row = QHBoxLayout()
        for axis, box in zip(('X', 'Y', 'Z'), self._pos_inputs):
            column = QVBoxLayout()
            label = QLabel(axis, self._widget)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            column.addWidget(box)
            pos_row.addLayout(column)
        input_layout.addRow('Position [m]', pos_row)

        rpy_row = QHBoxLayout()
        for axis, box in zip(('Roll', 'Pitch', 'Yaw'), self._rpy_inputs):
            column = QVBoxLayout()
            label = QLabel(f'{axis} [deg]', self._widget)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            column.addWidget(box)
            rpy_row.addLayout(column)
        input_layout.addRow('Orientation', rpy_row)

        button_row = QHBoxLayout()
        button_row.addWidget(self._publish_button)
        button_row.addWidget(self._prepare_button)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._stop_button)
        input_layout.addRow(button_row)

        state_box = QGroupBox('Current State', self._widget)
        state_layout = QFormLayout(state_box)

        state_pos_row = QHBoxLayout()
        for lbl in self._position_labels:
            state_pos_row.addWidget(lbl)
        state_layout.addRow('Position [m]', state_pos_row)

        state_rpy_row = QHBoxLayout()
        for lbl in self._orientation_labels:
            state_rpy_row.addWidget(lbl)
        state_layout.addRow('Roll / Pitch / Yaw [deg]', state_rpy_row)

        layout.addWidget(topic_box)
        layout.addWidget(input_box)
        layout.addWidget(state_box)
        layout.addWidget(self._status_label)
        layout.addStretch()

    def _attempt_initial_connection(self) -> None:
        if self._connections_ready:
            return
        try:
            self._ensure_connections()
        except rospy.ROSException:
            self._set_status('Waiting for MPC topics/services...', error=True)
            QTimer.singleShot(self._connection_retry_ms, self._attempt_initial_connection)

    def _ensure_connections(self) -> None:
        if self._connections_ready:
            return

        setpoint_topic = self._setpoint_topic_input.text().strip() or '/mpc_controller/setpoint'
        odom_topic = self._odom_topic_input.text().strip() or '/hummingbird/ground_truth/odometry'
        prepare_srvs = self._prepare_service_input.text().strip() or '/mpc_controller_node/prepare_trajectory'
        start_srvs = self._start_service_input.text().strip() or '/mpc_controller_node/start_trajectory'

        if self._setpoint_pub is None or self._setpoint_topic_name != setpoint_topic:
            self._setpoint_pub = rospy.Publisher(setpoint_topic, PoseStamped, queue_size=1)
            self._setpoint_topic_name = setpoint_topic
        if self._odom_sub is None or self._odom_topic_name != odom_topic:
            if self._odom_sub is not None:
                self._odom_sub.unregister()
            self._odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=1)
            self._odom_topic_name = odom_topic
        if self._prepare_client is None or self._prepare_service_name != prepare_srvs:
            self._prepare_client = rospy.ServiceProxy(prepare_srvs, SetBool, persistent=True)
            self._prepare_service_name = prepare_srvs
        if self._start_client is None or self._start_service_name != start_srvs:
            self._start_client = rospy.ServiceProxy(start_srvs, SetBool, persistent=True)
            self._start_service_name = start_srvs

        try:
            rospy.wait_for_service(prepare_srvs, timeout=2.0)
            rospy.wait_for_service(start_srvs, timeout=2.0)
        except rospy.ROSException as exc:
            raise

        self._connections_ready = True
        self._set_status(f'Connected to {setpoint_topic}', error=False)

    def _odom_cb(self, msg: Odometry) -> None:
        roll, pitch, yaw = quaternion_to_euler(msg.pose.pose.orientation.x,
                                               msg.pose.pose.orientation.y,
                                               msg.pose.pose.orientation.z,
                                               msg.pose.pose.orientation.w)
        state = VehicleState(
            position=(msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z),
            orientation_rpy=(roll, pitch, yaw),
        )
        with self._state_lock:
            self._latest_state = state

    def _update_display(self) -> None:
        with self._state_lock:
            state = self._latest_state
        if state is None:
            return

        for lbl, value in zip(self._position_labels, state.position):
            lbl.setText(f'{value: .2f}')

        for lbl, value in zip(self._orientation_labels, state.orientation_rpy):
            degrees = math.degrees(value)
            lbl.setText(f'{degrees: .1f}')

    def _handle_publish(self) -> None:
        try:
            self._ensure_connections()
        except rospy.ROSException as exc:
            self._set_status(f'Failed to connect: {exc}', error=True)
            return

        if self._setpoint_pub is None:
            self._set_status('Setpoint publisher is not ready.', error=True)
            return

        position = [box.value() for box in self._pos_inputs]
        roll = math.radians(self._rpy_inputs[0].value())
        pitch = math.radians(self._rpy_inputs[1].value())
        yaw = math.radians(self._rpy_inputs[2].value())

        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._frame_id_input.text().strip() or 'world'
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self._setpoint_pub.publish(msg)
        self._set_status('Setpoint published.', error=False)

    def _handle_prepare(self) -> None:
        self._invoke_service('prepare', self._prepare_client, True)

    def _handle_start(self) -> None:
        self._invoke_service('start', self._start_client, True)

    def _handle_stop(self) -> None:
        self._invoke_service('stop', self._start_client, False)

    def _invoke_service(self, label: str, client: Optional[rospy.ServiceProxy], enable: bool) -> None:
        try:
            self._ensure_connections()
        except rospy.ROSException as exc:
            self._set_status(f'Failed to connect: {exc}', error=True)
            return

        if client is None:
            self._set_status(f'{label.title()} service is not ready.', error=True)
            return

        try:
            client.wait_for_service(timeout=1.0)
        except rospy.ROSException as exc:
            self._set_status(f'{label.title()} service unavailable: {exc}', error=True)
            return

        try:
            response = client(enable)
        except rospy.ServiceException as exc:
            self._set_status(f'{label.title()} call failed: {exc}', error=True)
            return

        suffix = f': {response.message}' if response.message else ''
        status = 'succeeded' if response.success else 'failed'
        self._set_status(f'{label.title()} {status}{suffix}', error=not response.success)

    def _set_status(self, message: str, *, error: bool) -> None:
        self._status_label.setText(message)
        color = '#b00020' if error else '#2b2b2b'
        self._status_label.setStyleSheet(f'color: {color};')

    # Plugin API -----------------------------------------------------------------

    def shutdown_plugin(self) -> None:
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        if self._odom_sub is not None:
            self._odom_sub.unregister()
            self._odom_sub = None
        self._odom_topic_name = None
        self._setpoint_pub = None
        self._setpoint_topic_name = None
        self._prepare_client = None
        self._start_client = None
        self._prepare_service_name = None
        self._start_service_name = None

    def save_settings(self, plugin_settings, instance_settings) -> None:
        instance_settings.set_value('frame_id', self._frame_id_input.text())
        instance_settings.set_value('setpoint_topic', self._setpoint_topic_input.text())
        instance_settings.set_value('odom_topic', self._odom_topic_input.text())
        instance_settings.set_value('prepare_service', self._prepare_service_input.text())
        instance_settings.set_value('start_service', self._start_service_input.text())
        instance_settings.set_value('pos_values', [box.value() for box in self._pos_inputs])
        instance_settings.set_value('rpy_values', [box.value() for box in self._rpy_inputs])

    def restore_settings(self, plugin_settings, instance_settings) -> None:
        if instance_settings.contains('frame_id'):
            self._frame_id_input.setText(instance_settings.value('frame_id'))
        if instance_settings.contains('setpoint_topic'):
            self._setpoint_topic_input.setText(instance_settings.value('setpoint_topic'))
        if instance_settings.contains('odom_topic'):
            self._odom_topic_input.setText(instance_settings.value('odom_topic'))
        if instance_settings.contains('prepare_service'):
            self._prepare_service_input.setText(instance_settings.value('prepare_service'))
        if instance_settings.contains('start_service'):
            self._start_service_input.setText(instance_settings.value('start_service'))
        if instance_settings.contains('pos_values'):
            values = instance_settings.value('pos_values')
            try:
                for box, value in zip(self._pos_inputs, values):
                    box.setValue(float(value))
            except (TypeError, ValueError):
                pass
        if instance_settings.contains('rpy_values'):
            values = instance_settings.value('rpy_values')
            try:
                for box, value in zip(self._rpy_inputs, values):
                    box.setValue(float(value))
            except (TypeError, ValueError):
                pass

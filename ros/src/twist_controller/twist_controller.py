from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
Kp = 2
Ki = 0.1
Kd = 0.05

class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.min_speed = 0
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']
        self.pid_controller = PID(Kp, Ki, Kd, mn=self.decel_limit, mx=self.accel_limit)
        self.yaw_controller = YawController(self.wheel_base,
                                            self.steer_ratio,
                                            self.min_speed,
                                            self.max_lat_accel,
                                            self.max_steer_angle)
        self.vehicle_mass = kwargs['vehicle_mass'] + kwargs['fuel_capacity'] * GAS_DENSITY


    def control(self, current_velocity, twist_cmd, dbw_enabled, sample_time):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        throttle = 0.0
        brake = 0.0
        steering = 0.0

        if not all((current_velocity, twist_cmd)):
            return throttle, brake, steering

        linear_velocity_error = twist_cmd.twist.linear.x - current_velocity.twist.linear.x

        throttle = self.pid_controller.step(linear_velocity_error, sample_time)

        throttle = max(0.0, min(1.0, throttle))
        if linear_velocity_error < 0:
            # brake value = weight of vehicle * deceleration (e.g. 4 m/s/s) * wheel has a radius
            brake = self.vehicle_mass * -linear_velocity_error/sample_time * self.wheel_radius
            throttle = 0.0
        else:
            brake = 0.0

        steering = self.yaw_controller.get_steering(twist_cmd.twist.linear.x,
                                                    twist_cmd.twist.angular.z,
                                                    current_velocity.twist.linear.x)

        return throttle, brake, steering

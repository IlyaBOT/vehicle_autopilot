import json
import time

calculate_cos_clock_delay = 0.1 # в секундах
dt = 0.01 # в секундах
hor_kp, hor_ki, hor_kd = 0.3, 0.5, 0.08
z_kp, z_ki, z_kd = 20.0, 5.0, 3.0

calculate_cos_clock = time.time()
pid_clock = time.time()
prev_cos = 0
prev_direction = 0
target_axis_y = 0
yaw_right, yaw_left = 0, 0
prev_pos = [0, 0, 0]
motorSpeed = [0 for _ in range(8)]

err_x, err_y = 0, 0
integral_x, prevErr_x = 0, 0
integral_y, prevErr_y = 0, 0

err_z = 0
integral_z, prevErr_z = 0, 0

def constrain(x, a, b):
    if x < a:
        return a
    if x > b:
        return b
    return x

def computePID_X(input, setpoint, kp, ki, kd, dt, minOut, maxOut):
    global err_x, integral_x, prevErr_x
    err_x = setpoint - input
    integral_x = constrain(integral_x + err_x * dt * ki, minOut, maxOut)
    D = (err_x - prevErr_x) / dt
    prevErr_x = err_x
    return constrain(err_x * kp + integral_x + D * kd, minOut, maxOut)

def computePID_Y(input, setpoint, kp, ki, kd, dt, minOut, maxOut):
    global err_y, integral_y, prevErr_y
    err_y = setpoint - input
    integral_y = constrain(integral_y + err_y * dt * ki, minOut, maxOut)
    D = (err_y - prevErr_y) / dt
    prevErr_y = err_y
    return constrain(err_y * kp + integral_y + D * kd, minOut, maxOut)

def computePID_Z(input, setpoint, kp, ki, kd, dt, minOut, maxOut):
    global err_z, integral_z, prevErr_z
    err_z = setpoint - input
    integral_z = constrain(integral_z + err_z * dt * ki, minOut, maxOut)
    D = (err_z - prevErr_z) / dt
    prevErr_z = err_z
    return constrain(err_z * kp + integral_z + D * kd, minOut, maxOut)


def get_clock(timer):
    """Возвращает время в секундах"""
    return time.time() - timer


def calculate_cos(data: dict):
    global prev_direction, calculate_cos_clock, calculate_cos_clock_delay, prev_cos, target_axis_y, yaw_right, yaw_left, prev_pos
    target_x, target_y = data["targetVector"]["z"], data["targetVector"]["x"]
    current_x, current_y = data["droneVector"]["z"], data["droneVector"]["x"]

    if (abs(target_x - current_x) >= 30 or abs(target_y - current_y) >= 30) and get_clock(calculate_cos_clock) > calculate_cos_clock_delay:
        cos = (target_x * (current_x - prev_pos[0]) + target_y * (current_y - prev_pos[1])) / ((target_x ** 2 + target_y ** 2) ** 0.5 * ((current_x - prev_pos[0]) ** 2 + (current_y - prev_pos[1]) ** 2) ** 0.5)

        #data["targetAxisRotation"]["x"] = -20

        if cos < 0.95:
            if prev_cos > cos:
                if prev_direction == 1 or prev_direction == 0:
                    yaw_right = 10
                    prev_direction = 2
                elif prev_direction == 2:
                    yaw_left = 10
                    prev_direction = 1
            else:
                if prev_direction == 1:
                    yaw_left = 10
                elif prev_direction == 2:
                    yaw_right = 10

        calculate_cos_clock = time.time()
        prev_pos = [current_x, current_y, data["droneVector"]["y"]]
        prev_cos = cos


def calculate_engine(data):
    global pid_clock, dt, motorSpeed

    axis_x, axis_y = float(data["droneAxisRotation"]["z"]), float(data["droneAxisRotation"]["x"])
    target_axis_x, target_axis_y = data["targetAxisRotation"]["z"], data["targetAxisRotation"]["x"]
    current_z, target_z = data["droneVector"]["y"], data["targetVector"]["y"]

    if get_clock(pid_clock) > dt:
        xSpeed = computePID_X(axis_x, target_axis_x, hor_kp, hor_ki, hor_kd, dt, -15, 15)
        ySpeed = computePID_Y(axis_y, target_axis_y, hor_kp, hor_ki, hor_kd, dt, -15, 15)
        speed = computePID_Z(current_z, target_z, z_kp, z_ki, z_kd, dt, 0, 80)

        motorSpeed[0] = speed + xSpeed + yaw_right
        motorSpeed[1] = speed + xSpeed + yaw_left
        motorSpeed[2] = speed + ySpeed + yaw_right
        motorSpeed[3] = speed + ySpeed + yaw_left
        motorSpeed[4] = speed - xSpeed + yaw_right
        motorSpeed[5] = speed - xSpeed + yaw_left
        motorSpeed[6] = speed - ySpeed + yaw_right
        motorSpeed[7] = speed - ySpeed + yaw_left

        pid_clock = time.time()

    return motorSpeed


def analyze(str_data: str):
    result = []
    for data in json.loads(str_data):
        calculate_cos(data)
        engines = calculate_engine(data)

        res = {"id": data["id"], "engines": {
            "fr": engines[1],
            "fl": engines[0],
            "br": engines[4],
            "bl": engines[5],
            "rf": engines[2],
            "rb": engines[3],
            "lf": engines[7],
            "lb": engines[6],
        }}

        result.append(res)

    return json.dumps(result)

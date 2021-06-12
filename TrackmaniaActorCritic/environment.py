import screenFormat
import ctypes
import InputController


class TrackmaniaEnv:
    speed_address = 0
    checkpoint_address = 0
    goal_address = 0

    def __init__(self, spd_address, checkpt_address, go_address):
        self.speed_address = spd_address
        self.checkpoint_address = checkpt_address
        self.goal_address = go_address

    def reset(self):
        # just return a screen capture
        # possible pause?
        screenshot = screenFormat.capturescreen()
        return screenshot

    def step(self, actions):
        InputController.play_function_keyboard(actions[0], actions[1])
        screenshot = screenFormat.capturescreen()
        current_speed = ctypes.c_int.from_address(self.speed_address)
        passed_checkpoints = ctypes.c_int.from_address(self.checkpoint_address)
        done = ctypes.c_int.from_address(self.goal_address)

        reward = current_speed.value + 10 * passed_checkpoints.value

        return screenshot, reward, done








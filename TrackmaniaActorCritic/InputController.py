import pyvjoy
import directKeys

MAX_VJOY = 32767
j = pyvjoy.VJoyDevice(1)

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20


def play_function(self, x, z=0):
    self.j.data.wAxisX = int(x * self.MAX_VJOY)
    # self.j.data.wAxisY = int(y * self.MAX_VJOY)
    self.j.data.wAxisZRot = int(z * self.MAX_VJOY)

    j.update()


def play_function_keyboard(x, z):
    if x != 0:
        if x > 0:
            directKeys.PressKey(D)
        else:
            directKeys.PressKey(A)
    if z > 0:
        directKeys.PressKey(W)


import subprocess
import ctypes
import time
import psutil
import win32con
import win32gui
import win32api
import win32process
import numpy as np
import pyautogui as pag
from PIL import Image, ImageGrab


def game_start():
    # # 打开游戏窗口
    # url = "D:/work_tool/PycharmProjects/tensorflowProject/DQN/game/Jump.exe"
    # subprocess.Popen(url)
    # time.sleep(3)
    # 获取游戏窗口的句柄
    hwnd = win32gui.FindWindow(None, '你要上天！')
    # 将窗口设置为可见
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    # 检查窗口是否有效
    if win32gui.IsWindow(hwnd):
        # 将窗口置于前台
        win32gui.SetForegroundWindow(hwnd)
    else:
        print("窗口无效")


def get_window_pos(name):
    name = name
    handle = win32gui.FindWindow(0, name)
    # 获取窗口句柄
    if handle == 0:
        return None
    else:
        # 返回坐标值和handle
        return win32gui.GetWindowRect(handle), handle


def fetch_image():
    (x1, y1, x2, y2), handle = get_window_pos('你要上天！')
    # 发送还原最小化窗口的信息
    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # 设为高亮
    win32gui.SetForegroundWindow(handle)
    # 截图
    grab_image = ImageGrab.grab((x1, y1, x2, y2))

    return grab_image


def find_process_id(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if process_name.lower() in proc.info['name'].lower():
            return proc.info['pid']


def frame_step(input_actions, score):
    # 获取游戏窗口的句柄
    hwnd = win32gui.FindWindow(None, '你要上天！')
    # print(hwnd)
    # 取进程ID
    # pid = win32process.GetWindowThreadProcessId(hwnd)[1]
    pid = find_process_id('javaw.exe')
    # print(pid)
    # 获取游戏进程的句柄
    phd = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, pid)
    # print(phd)
    # 获取游戏基址
    # game_base_address = ctypes.c_ulonglong(0)
    # ctypes.windll.psapi.GetModuleBaseNameA(phd, None, ctypes.byref(game_base_address), 8)
    # print(game_base_address)
    # 读取内存数据
    # offset = 0x00 # 偏移量
    score_address = 0x76B2C2B10
    death_address = 0x76B2C2B14
    buffer = ctypes.create_string_buffer(4)  # 读取4字节的数据
    # print(value1)
    # 奖励值
    reward = 0.1
    # 游戏是否结束
    terminal = False

    # 向上
    if input_actions[0] == 1:
        pag.press('Up')
        # time.sleep(2.5)

    # 向左走
    if input_actions[1] == 1:
        pag.press('Left')
        # time.sleep(0.4)

    # 向右走
    if input_actions[2] == 1:
        pag.press('Right')
        # time.sleep(0.4)

    # # 向左跳
    # if input_actions[3] == 1:
    #     pag.press('Up')
    #     pag.press('Left')
    #     time.sleep(2.5)
    #
    # # 向右跳
    # if input_actions[4] == 1:
    #     pag.press('Up')
    #     pag.press('Right')
    #     time.sleep(2.5)

    time.sleep(0.3)
    ctypes.windll.kernel32.ReadProcessMemory(phd, ctypes.c_void_p(score_address), buffer, 4, None)
    value2 = int.from_bytes(buffer.raw, byteorder='little')
    ctypes.windll.kernel32.ReadProcessMemory(phd, ctypes.c_void_p(death_address), buffer, 4, None)
    death = int.from_bytes(buffer.raw, byteorder='little')
    # print(value2)

    if value2 > score:
        reward = 1.0
        score = value2
        frequency = 0

    if death == 1:
        reward = -1.0
        score = 0
        terminal = True

    image_data = np.array(fetch_image())
    return image_data, reward, terminal, score



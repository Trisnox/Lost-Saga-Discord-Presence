import copy
import ctypes
import cv2
import itertools
import json
import multiprocessing
import numpy as np
import os
import pydirectinput
import pyautogui
import pynput
import pytesseract
import pypresence
import time
import random
import string
import tkinter as tk
import win32gui, win32com.client
from PIL import Image, ImageGrab, ImageTk
from tkinter import filedialog, ttk

def is_admin():
    try:
        is_admin = (os.getuid() == 0)
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    return is_admin

if not is_admin():
    tk.messagebox.showerror(title='Missing Perms', message='Program tidak di run menggunakan admin. Jika program tidak di run dengan admin, kemungkinan tidak bisa mengambil screenshot dari game.')
    exit()

input_process = None
main_process = None
toggle = False
pup_up = None
pynput_keyboard = None
file_name = None

with open('config.json', 'r') as f:
    config: dict = json.load(f)

def crop_center(img, w, h):
    im_w, im_h = img.size
    return img.crop(((im_w - w) // 2,
                    (im_h - h) // 2,
                    (im_w + w) // 2,
                    (im_h + h) // 2))

# def image_search(img, target, precision):
#     img_rgb = np.array(img)
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
#     template = cv2.imread(target, 0)
#     template.shape[::-1]

#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     img.close()
#     if max_val < precision:
#         return [-1, -1]
    
#     return max_loc

def image_search(img, target, precision):
    transparent = False

    img_rgb = np.array(img)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    if target.lower().endswith('png'):
        img = Image.open(target)
        if not img.info.get('transparency', None) is None:
            transparent = True
        elif img.mode == 'P':
            transparency = img.info.get('transparency', -1)
            for _, index in img.getcolors():
                if index == transparency:
                    transparent = True
        elif img.mode == "RGBA":
            extrema = img.getextrema()
            if extrema[3][0] < 255:
                transparent = True
    print(transparent)
    if transparent:
        input_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template_image = cv2.imread(target, cv2.IMREAD_UNCHANGED)
        gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        alpha_channel = template_image[:,:,3]
        _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(template_image[:,:,3], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        # hh, ww = mask.shape[:2]

        res = cv2.matchTemplate(input_image, gray_template, cv2.TM_CCORR_NORMED, mask=mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < precision:
            print(max_val, ' for: ', target)
            return None
        print(max_val, ' for: ', target)
        return max_loc
    else:
        template = cv2.imread(target, 0)
        template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        img.close()

        if max_val < precision:
            print(max_val, ' for: ',  target)
            return None
        print(max_val, ' for: ',  target)
        return max_loc

# if transparent:
#         target_image =  cv2.imread(target, cv2.IMREAD_UNCHANGED)
#         gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

#         alpha_channel = target_image[:,:,3]

#         _, partial = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

#         contour, _ = cv2.findContours(partial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         big_contour = max(contour, key=cv2.contourArea)

#         mask = np.zeros_like(target_image[:,:,3], dtype=np.uint8)
#         cv2.drawContours(mask, [big_contour], -1, 255, -1)
#         # hh, ww = mask.shape[:2]

#         res = cv2.matchTemplate(img_gray, gray_target, cv2.TM_CCOEFF_NORMED, mask=mask)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#         img.close()

#         if max_val < precision:
#             print(max_val, ' for: ',  target)
#             return None
#         print(max_val, ' for: ',  target)
#         return max_loc

# def test(img, target):
#     input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     template_image = cv2.imread(target, cv2.IMREAD_UNCHANGED)
#     gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
#     alpha_channel = template_image[:,:,3]
#     _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # contours = contours[0] if len(contours) == 2 else contours[1]
#     largest_contour = max(contours, key=cv2.contourArea)
#     mask = np.zeros_like(template_image[:,:,3], dtype=np.uint8)
#     cv2.drawContours(mask, [largest_contour], -1, 255, -1)
#     # hh, ww = mask.shape[:2]
#     print(type(input_image))
#     print(type(template_image))
#     print(type(mask))
#     cv2.imshow('input_image', input_image)
#     cv2.imshow('template_image', template_image)
#     cv2.imshow('mask', mask)
#     res = cv2.matchTemplate(input_image, gray_template, cv2.TM_CCORR_NORMED, mask=mask)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     return max_val, max_loc

def find_template_instances(img, template, precision):
    image = np.array(img)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
        
    match = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        
    locations = np.where(match >= precision)
    locations = list(zip(*locations[::-1]))
        
    return locations

def get_win_ss():
    toplist, winlist = [], []
    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(enum_cb, toplist)

    lso = [(hwnd, title) for hwnd, title in winlist if 'lost saga in time' in title.lower()]
    try:
        lso = lso[0]
        hwnd = lso[0]
    except IndexError:
        return False

    
    # fix fullscreen problem... I think?
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    
    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    time.sleep(0.4) # bug fix, it took a few ms to capture the game before the window swaps
    img = ImageGrab.grab(bbox)
    if img:
        return img
    else:
        return False

def time_set():
    return int(time.time())

def ocr_num(img):
    text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789: --psm 6 digits')
    return text

def ocr_text(img):
    text = pytesseract.image_to_string(img, config='--psm 6')
    return text

def main():
    interval = config['interval']
    username = config['username']
    precision = config['precision'] / 100
    time_running = None
    time_start = None
    last_known_round = 0
    rpc = None
    print("ready\n\n")
    while True:
        screenshot = get_win_ss()
        if not screenshot:
            time.sleep(interval)
            time_running = None
            if rpc:
                rpc.close()
            rpc = None
            continue
        
        acceptable_w = [800, 1024, 1280, 1600, 1680, 1920]
        acceptable_h = [600, 768, 720, 1024, 900, 1050, 1080]
        w, h = screenshot.size
        if not h in acceptable_h: # crop by 3, 26, 3, 3
            w = w-6
            h = h-29
            screenshot = screenshot.crop((3, 26, w, h+26))

        print(w, h)
        guide_dict = {
            1080: (220, 206),
            1050: (217, 203),
            1024: (214, 200),
            900: (202, 188),
            768: (188, 174),
            720: (184, 170),
            600: (172, 158)
            }
        
        corner_UI = (146, 137)

        if not time_running:
            time_running = time_set()

        if not rpc:
            # It is considered to check the folder location of the process, and then use regex to grab the filename
            # and check which lost saga does this belong.
            # If it doesn't have match, then this default presence name will be used instead
            rpc = pypresence.Presence("1100419868576202762")
            rpc.connect()
        
            rpc.update(
                start=time_running,
                large_image="ls_main",
                large_text="Lost Saga Origin",
                details=username,
                )
        
        # hold for 0.06 (0.07 rounded) for the box to appear.
        # or hold for 0.08 for failsafe
        # for 60 fps, it's 4/5 frames
        # for 30 fps, it's 3 frames

        while True:
            horizontal, vertical = guide_dict[h]
            splice = screenshot.crop((0, 0, 500, 500))
            pos = image_search(splice, 'img/lobby.png', precision)
            if pos:
                print('using lobby presence')
                time_start = None
                rpc.update(
                    start=time_running,
                    large_image="lobby",
                    large_text="Lobby",
                    small_image='ls_main',
                    small_text='Lost Saga Origin',
                    details=username,
                    state='Lobby'
                    )
                break
            
            pos = image_search(splice, 'img/room.png', precision)
            if pos:
                time_start = None
                break
            
            pydirectinput.keyDown('tab')
            time.sleep(0.08)
            screenshot = get_win_ss()
            if not screenshot:
                time.sleep(interval)
                time_running = None
                if rpc:
                    rpc.close()
                rpc = None
                continue

            w, h = screenshot.size
            if not h in acceptable_h: # crop by 3, 26, 3, 3
                w = w-6
                h = h-29
                screenshot = screenshot.crop((3, 26, w, h+26))

            room_info = crop_center(screenshot, 670, 366)
            pydirectinput.keyUp('tab')

            pos = image_search(room_info, 'img/room_indicator_1.png', precision)
            if not pos:
                pos = image_search(room_info, 'img/room_indicator_2.png', precision)
                if not pos:
                    break

            pos = image_search(room_info, 'img/prisoner.png', precision+0.07)
            if pos:
                print('using regular prisoner')
                timer_ui = screenshot.crop((horizontal-146, vertical-137, horizontal, vertical))
                raw_time = timer_ui.crop((64, 7, 64+23, 7+7))
                raw_time = raw_time.resize((25*12, 11*12), resample=Image.LANCZOS)
                time_end = ocr_num(raw_time)
                count_round = -1

                if not len(time_end) == 4:
                    time_end = None
                else:
                    time_end = int(time_end)
                # formula 1: 64, 7, 64+25, 7+7, and scale by 25*12, 10*12
                # another formula: 64 7, 64+25, 7+7 and scale by 25*12, 11*12
                if last_known_round >= count_round:
                    count_round = copy.deepcopy(last_known_round)
                    time_start = time_set()
                if type(time_end) == int:
                    if time_start:
                        time_end += time_start
                    else:
                        time_end += time.time()

                # I give up on using OCR for rounds, so I'll just use image instead... besides, they're only have 4 numbers either way
                # rounds = timer_ui.crop((33, 55, 80+33, 40+55))
                # rounds = rounds.resize(???)
                # Deathmatch? I'll just try to grab position from the tab instead... hope it doesn't get too distracting

                round1 = timer_ui.crop((30, 55, 36+30, 45+55))
                round2 = timer_ui.crop((78, 55, 36+78, 45+55))
                rounds = [] # separated, in case I thought of changing stuffs
                round_numbers = [('img/1.png', 1), ('img/2.png', 2), ('img/3.png', 3), (None, 0)]

                for x, y in round_numbers:
                    if x:
                        pos = image_search(round1, x, 0.8)
                        if pos:
                            rounds += [y]
                            break
                    else:
                        rounds += [0]

                for x, y in round_numbers:
                    if x:
                        pos = image_search(round2, x, 0.8)
                        if pos:
                            rounds += [y]
                            break
                    else:
                        rounds += [0]

                last_known_round = sum(rounds)
                match_round = f'Round: {sum(rounds) + 1} of 5'

                # party_size = (24, 134) for first half (343, 134) for second half (21, 168) for size
                w = 21
                h = 168
                party1 = room_info.crop((24, 134, w+24, h+134))
                party2 = room_info.crop((343, 134, w+343, h+134))

                party_instances1 = find_template_instances(party1, 'img/slot_empty.png', 0.9)
                party_instances2 = find_template_instances(party2, 'img/slot_empty.png', 0.9)

                color_pix = room_info.load()
                empty = (206, 78, 0, 255)
                locked = (167, 167, 167, 255)

                size = 16
                empty = 0
                locked = 0

                for x, y in party_instances1:
                    x = x + 24 + 4
                    y = y + 134 + 3
                    if color_pix[x, y] == empty:
                        empty += 1
                    else:
                        locked += 1
                for x, y in party_instances2:
                    x = x + 343 + 4
                    y = y + 134 + 3
                    if color_pix[x, y] == empty:
                        empty += 1
                    else:
                        locked += 1
                
                party_current = size - empty - locked
                party_max = size - locked

                room_name_raw = room_info.crop((140, 9, 405+140, 27+9))
                room_name = ocr_text(room_name_raw).strip()
                if room_name.endswith('(Pribadi)'):
                    room_name = room_name[:-9] + ' (Private Room)'
                print(time_start, time_end)
                # the time end is time start += end, with end being time remaining
                if type(time_start) != None and type(time_end) != None:
                    if time_start == time_end:
                        rpc.update( large_image='prisoner',
                                    large_text='Prisoner',
                                    small_image='ls_main',
                                    small_text=username,
                                    details=room_name,
                                    state=f'{match_round} (Death Time)',
                                    party_size=[party_current, party_max])
                    else:
                        rpc.update( start=time_start,
                                    end=time_end,
                                    large_image='prisoner',
                                    large_text='Prisoner',
                                    small_image='ls_main',
                                    small_text=username,
                                    details=room_name,
                                    state=f'{match_round}',
                                    party_size=[party_current, party_max])
                else:
                    rpc.update( start=time_start,
                                end=time_end,
                                large_image='prisoner',
                                large_text='Prisoner',
                                small_image='ls_main',
                                small_text=username,
                                details=room_name,
                                state=f'{match_round} (Time Unknown)',
                                party_size=[party_current, party_max])

                break

            pos = image_search(room_info, 'img/prisoner_user_mode.png', 0.77)
            if pos:
                print('using prisoner user mode')
                timer_ui = screenshot.crop((horizontal-146, vertical-137, horizontal, vertical))
                raw_time = timer_ui.crop((64, 7, 64+23, 7+7))
                raw_time = raw_time.resize((25*12, 11*12), resample=Image.LANCZOS)
                time_end = ocr_num(raw_time)
                count_round = -1

                if not len(time_end) == 4:
                    time_end = None
                else:
                    time_end = int(time_end)
                # formula 1: 64, 7, 64+25, 7+7, and scale by 25*12, 10*12
                # another formula: 64 7, 64+25, 7+7 and scale by 25*12, 11*12
                if last_known_round >= count_round:
                    count_round = copy.deepcopy(last_known_round)
                    time_start = time_set()
                if type(time_end) == int:
                    if time_start:
                        time_end += time_start
                    else:
                        time_end += time.time()

                # I give up on using OCR for rounds, so I'll just use image instead... besides, they're only have 4 numbers either way
                # rounds = timer_ui.crop((33, 55, 80+33, 40+55))
                # rounds = rounds.resize(???)
                # Deathmatch? I'll just try to grab position from the tab instead... hope it doesn't get too distracting

                round1 = timer_ui.crop((30, 55, 36+30, 45+55))
                round2 = timer_ui.crop((78, 55, 36+78, 45+55))
                rounds = [] # separated, in case I thought of changing stuffs
                round_numbers = [('img/1.png', 1), ('img/2.png', 2), ('img/3.png', 3), (None, 0)]

                for x, y in round_numbers:
                    if x:
                        pos = image_search(round1, x, 0.8)
                        if pos:
                            rounds += [y]
                            break
                    else:
                        rounds += [0]

                for x, y in round_numbers:
                    if x:
                        pos = image_search(round2, x, 0.8)
                        if pos:
                            rounds += [y]
                            break
                    else:
                        rounds += [0]

                last_known_round = sum(rounds)
                match_round = f'Round: {sum(rounds) + 1} of 5'

                # party_size = (24, 134) for first half (343, 134) for second half (21, 168) for size
                w = 21
                h = 168
                party1 = room_info.crop((24, 134, w+24, h+134))
                party2 = room_info.crop((343, 134, w+343, h+134))

                party_instances1 = find_template_instances(party1, 'img/slot_empty.png', 0.9)
                party_instances2 = find_template_instances(party2, 'img/slot_empty.png', 0.9)

                color_pix = room_info.load()
                empty = (206, 78, 0, 255)
                locked = (167, 167, 167, 255)

                size = 16
                empty = 0
                locked = 0

                for x, y in party_instances1:
                    x = x + 24 + 4
                    y = y + 134 + 3
                    if color_pix[x, y] == empty:
                        empty += 1
                    else:
                        locked += 1
                for x, y in party_instances2:
                    x = x + 343 + 4
                    y = y + 134 + 3
                    if color_pix[x, y] == empty:
                        empty += 1
                    else:
                        locked += 1
                
                party_current = size - empty - locked
                party_max = size - locked

                room_name_raw = room_info.crop((140, 9, 405+140, 27+9))
                room_name = ocr_text(room_name_raw).strip()
                if room_name.endswith('(Pribadi)'):
                    room_name = room_name[:-9] + ' (Private Room)'
                print(time_start, time_end)
                # the time end is time start += end, with end being time remaining
                if type(time_start) != None and type(time_end) != None:
                    if time_start == time_end:
                        rpc.update( large_image='user_mode',
                                    large_text='Prisoner User Mode',
                                    small_image='ls_main',
                                    small_text=username,
                                    details=room_name,
                                    state=f'{match_round} (Death Time)',
                                    party_size=[party_current, party_max])
                    else:
                        rpc.update( start=time_start,
                                    end=time_end,
                                    large_image='user_mode',
                                    large_text='Prisoner User Mode',
                                    small_image='ls_main',
                                    small_text=username,
                                    details=room_name,
                                    state=f'{match_round}',
                                    party_size=[party_current, party_max])
                else:
                    rpc.update( start=time_start,
                                end=time_end,
                                large_image='user_mode',
                                large_text='Prisoner',
                                small_image='ls_main',
                                small_text=username,
                                details=room_name,
                                state=f'{match_round} (Time Unknown)',
                                party_size=[party_current, party_max])

                break

                break
            break

        time.sleep(interval)

class ImageLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []

        try:
            for i in itertools.count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image="")
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

def math_loop():
    img = get_win_ss()
    img = pre_crop(img)
    
    res = image_search(img, './img/afk1.png')
    if not res:
        res = image_search(img, './img/afk2.png')
    
    if not res:
        return False
    fn = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(14))
    img.save(f'img/{fn}.png')
    img = math_crop(img)

    text = ocr()

    num = ocr_check(text)

    if not num:
        num = math_loop()
    
    return num

def pre_crop(img):
    global config
    if not config['window'] == 'full': # windowed mode
        w, h = img.size
        if config['resolution'] == '800x600':
            img = img.crop(((w - 255) // 2, 45, (w + 255) // 2, 171))
        elif config['resolution'] == '1024x768':
            img = img.crop(((w - 255) // 2, 50, (w + 255) // 2, 176))
        elif config['resolution'] == '1280x720':
            img = img.crop(((w - 255) // 2, 49, (w + 255) // 2, 151))
        elif config['resolution'] == '1280x1024':
            img = img.crop(((w - 255) // 2, 59, (w + 255) // 2, 185))
        elif config['resolution'] == '1600x900':
            img = img.crop(((w - 255) // 2, 55, (w + 255) // 2, 182))
        elif config['resolution'] == '1680x1050':
            img = img.crop(((w - 255) // 2, 60, (w + 255) // 2, 186))
        elif config['resolution'] == '1920x1080':
            img = img.crop(((w - 255) // 2, 61, (w + 255) // 2, 187))
        else:
            print('unknown resolution selected')
    else:
        w, h = img.size
        if config['resolution'] == '800x600':
            img = img.crop(((w - 255) // 2, 19, (w + 255) // 2, 145))
        elif config['resolution'] == '1024x768':
            img = img.crop(((w - 255) // 2, 24, (w + 255) // 2, 150))
        elif config['resolution'] == '1280x720':
            img = img.crop(((w - 255) // 2, 23, (w + 255) // 2, 149))
        elif config['resolution'] == '1280x1024':
            img = img.crop(((w - 255) // 2, 33, (w + 255) // 2, 159))
        elif config['resolution'] == '1600x900':
            img = img.crop(((w - 255) // 2, 29, (w + 255) // 2, 156))
        elif config['resolution'] == '1680x1050':
            img = img.crop(((w - 255) // 2, 34, (w + 255) // 2, 160))
        elif config['resolution'] == '1920x1080':
            img = img.crop(((w - 255) // 2, 35, (w + 255) // 2, 161))
        else:
            print('unknown resolution selected')
    
    return img

def math_crop(img):
    global file_name
    img = img.crop((40, 67, 41+40, 15+67))
    w, h = img.size
    img = img.resize((w*5, h*5), resample=Image.LANCZOS)
    file_name = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(14))
    img.save(f'./img/{file_name}.png')
    return img

def ocr():
    global file_name
    img = cv2.imread(f"img/{file_name}.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = im2[y:y + h, x:x + w]

        text = pytesseract.image_to_string(cropped, config='-c tessedit_char_whitelist=0123456789+- --psm 6 digits')
        print(text)

    return text

def ocr_check(result):
    global config
    global pynput_keyboard
    filtered = result.split('+')
    try:
        num1 = filtered[0]
        int(num1)
        num2 = filtered[1]
        int(num2)
    except (ValueError, IndexError):
        print('OCR gagal mengidentifikasi seluruh nomor, mencoba ulang')
        u_input = config['input']
        if u_input == 'directinput':
            pydirectinput.press('1')
            time.sleep(.1)
            pydirectinput.press('enter')
        else:
            pynput_keyboard.tap('1')
            time.sleep(.1)
            pynput_keyboard.tap(pynput.keyboard.Key.enter)
        return False

    if len(num1) >= 3:
        num1 = num1[2]
        return (num1, num2)
    else:
        return (num1, num2)

def toggle_script():
    global main_process
    global toggle
    global tesseract_loc
    global interval_input
    global fix_input
    global input_input
    global window_input
    global resolution_input
    global toggle_button

    global config

    if not toggle:
        tesseract_loc.set(tesseract_entry.get())
        interval_input.set(interval_entry.get())
        
        if not tesseract_loc.get():
            tk.messagebox.showerror(title='Invalid', message='Lokasi Tesseract tidak bisa kosong')
            return False
        
        try:
            float(interval_input.get())
        except ValueError:
            tk.messagebox.showerror(title='Invalid', message='Interval harus berupa int atau float')
            return False
        
        config['tesseract'] = tesseract_loc.get().replace('/', '\\')
        config['fix'] = fix_input.get()
        config['input'] = input_input.get()
        config['interval'] = float(interval_input.get())
        config['window'] = window_input.get()
        config['resolution'] = resolution_input.get()

        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        toggle = True
        toggle_button.config(text='Stop')
        main_process = multiprocessing.Process(target=main)
        main_process.start()
    else:
        main_process.terminate()
        toggle = False
        main_process = None
        print('Script telah dihentikan')
        toggle_button.config(text='Mulai')

def test_screenshot():
    tk.messagebox.showinfo(message='Setelah klik ok, fokuskan window ke Lost Saga lalu tunggu beberapa detik. Setelah lebih dari 5/6 detik, screenshot akan muncul di folder img')
    time.sleep(5)

    img = get_win_ss()

    if img:
        img.save('./img/ss.png')
    else:
        tk.messagebox.showwarning(title='Warning', message='Tidak dapat screenshot Lost Saga karena proses tidak ditemukan')

def tesseract_set():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select A File",filetype = (("exe","*.exe"),("All Files","*.*")))
    if filename:
        tesseract_loc.set(filename)

if __name__ == '__main__':
    main()
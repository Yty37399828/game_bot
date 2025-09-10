import cv2
import numpy as np
import pygetwindow as gw
import mss
import time
import random
import pyautogui
import threading  # 新增：用于多线程同步按键
from pynput import keyboard
from ultralytics import YOLO
from threading import Thread

# -------------------------- 配置参数 --------------------------
GAME_WINDOW_TITLE = "MapleStory Worlds-Artale (繁體中文版)"
YOLO_MODEL_PATH = "runs4/train/exp/weights/best.pt"
CONF_THRESHOLD = 0.6
USE_GPU = True  # GPU加速开关
PLAYER_WIDTH_RATIO = 4
PLAYER_HEIGHT_RATIO = 0.8
HUMAN_ACTION_PROB = {
    "empty_skill": 0.02,
    "mistake_jump": 0.02
}
KEY_PRESS_DURATION = (0.08, 0.2)
# 爬绳索相关配置
ROPE_CLIMB_INTERVAL = 600  # 强制爬绳索间隔（10分钟）
ROPE_PLAYER_X_RANGE = 5  # 角色与绳索的水平允许偏差
ROPE_CLIMB_DURATION = 3.0  # 攀爬时长（长按上键时间）
ROPE_JUMP_DELAY = 0.3  # 跳跃+上键同时按下的持续时间（确保接触绳索）
MOVE_INTERVAL = (2.0, 3.0)
END_KEY_INTERVAL = 300  # 定时按buff键的间隔（5分钟）
# --------------------------------------------------------------

# 全局变量
is_paused = False  # 暂停标志
game_window = None  # 游戏窗口对象
model = None  # YOLO模型对象
sct = mss.mss()  # 屏幕捕获对象
last_rope_climb_time = time.time()  # 上次爬绳索时间戳
last_end_key_time = time.time()  # 上次按buff键时间戳

# 键盘按键映射
KEYS = {
    "left": "left",
    "right": "right",
    "jump": "alt",  # 跳跃键
    "attack": "ctrl",  # 攻击键
    "climb": "up",  # 上键（攀爬用）
    "end": "a"  # 定时按键（A键）
}


def init_game_window():
    """初始化游戏窗口捕获，激活并准备好窗口用于后续截图"""
    global game_window
    try:
        game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
        if not game_window.isActive:
            game_window.activate()  # 激活窗口
        if game_window.isMinimized:
            game_window.restore()  # 还原窗口（如果最小化）
        time.sleep(1)
        print(f"成功找到游戏窗口：{GAME_WINDOW_TITLE}")
        print(f"窗口位置：({game_window.left}, {game_window.top}) 大小：{game_window.width}x{game_window.height}")
        return True
    except IndexError:
        print(f"错误：未找到标题为'{GAME_WINDOW_TITLE}'的窗口")
        return False


def init_yolo_model():
    """初始化YOLO模型，自动检测并使用GPU（如果可用）"""
    global model
    try:
        model = YOLO(YOLO_MODEL_PATH)  # 加载模型

        # 尝试使用GPU
        if USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    model.to('cuda')  # 切换到GPU
                    # 验证GPU是否工作
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    model(dummy_img, verbose=False)
                    print(f"成功加载YOLO模型并使用GPU加速：{YOLO_MODEL_PATH}")
                    return True
                else:
                    print("CUDA不可用，自动切换到CPU模式")
            except ImportError:
                print("未安装PyTorch，自动使用CPU模式")

        # 若GPU不可用则使用CPU
        model.to('cpu')
        print(f"成功加载YOLO模型（CPU模式）：{YOLO_MODEL_PATH}")
        return True

    except Exception as e:
        print(f"加载YOLO模型失败：{str(e)}")
        return False


def capture_game_screen():
    """捕获游戏窗口区域的图像，用于后续目标检测"""
    if not game_window:
        return None

    # 定义游戏窗口的捕获区域
    monitor = {
        "top": game_window.top,
        "left": game_window.left,
        "width": game_window.width,
        "height": game_window.height
    }

    try:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # 转换颜色格式
        return img
    except Exception as e:
        print(f"屏幕捕获失败：{str(e)}")
        return None


def detect_objects(img):
    """使用YOLO模型检测图像中的目标（玩家、怪物、绳索、金币）"""
    if not model or img is None:
        return {"player": [], "boar": [], "rope": [], "coin": []}

    # 执行目标检测（使用模型已设置的设备：GPU/CPU）
    results = model(img, conf=CONF_THRESHOLD, classes=[0, 1, 2, 3], verbose=False)
    detections = {"player": [], "boar": [], "rope": [], "coin": []}

    # 解析检测结果
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # 根据类别存储检测结果
            if cls_id == 1:  # 玩家
                detections["player"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 0:  # 怪物
                detections["boar"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 2:  # 绳索
                detections["rope"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 3:  # 金币
                detections["coin"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})

    return detections


def press_key(key, duration=None):
    """模拟按键操作，duration为按键持续时间"""
    if key not in KEYS:
        return

    actual_key = KEYS[key]
    try:
        press_time = duration if duration else random.uniform(*KEY_PRESS_DURATION)
        pyautogui.keyDown(actual_key)
        time.sleep(press_time)
        pyautogui.keyUp(actual_key)
        time.sleep(random.uniform(0.03, 0.1))  # 按键后小停顿
    except Exception as e:
        print(f"按键操作失败（{key}）：{str(e)}")


def hold_key(key, hold_time):
    """模拟长按按键（用于需要持续按住的操作）"""
    if key not in KEYS:
        return

    actual_key = KEYS[key]
    try:
        print(f"[爬绳索] 长按{key}键{hold_time}秒")
        pyautogui.keyDown(actual_key)
        time.sleep(hold_time)
        pyautogui.keyUp(actual_key)
        time.sleep(random.uniform(0.1, 0.2))
    except Exception as e:
        print(f"长按按键失败（{key}）：{str(e)}")


def simulate_human_behavior():
    """模拟人类的随机操作行为，增加自动化的真实性"""
    # 随机空放技能（2%概率）
    if random.random() < HUMAN_ACTION_PROB["empty_skill"]:
        print("[人类模拟] 空放技能")
        press_key("attack")

    # 随机误碰跳跃（2%概率）
    if random.random() < HUMAN_ACTION_PROB["mistake_jump"]:
        print("[人类模拟] 误碰跳跃")
        press_key("jump")


def get_player_state(detections):
    """获取玩家状态：位置、尺寸和所在平台层级"""
    if not detections["player"]:
        return None

    player = detections["player"][0]
    # 判断是否在下层平台（y坐标超过窗口高度的一半）
    is_lower_platform = player["y"] > (game_window.height / 2)

    return {
        "x": player["x"],
        "y": player["y"],
        "width": player["w"],
        "is_lower": is_lower_platform
    }


def get_boar_positions(detections, player_x, player_y):
    """获取怪物位置，并分类为前方/后方怪物（相对于玩家）"""
    boars = detections["boar"]
    try:
        player = detections["player"][0]
        player_w = player["w"]
        player_h = player["h"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"获取玩家信息失败: {e}")
        return {"front": [], "back": [], "all": []}

    if not boars or player_x is None:
        return {"front": [], "back": [], "all": []}

    # 计算有效检测范围（基于玩家身位）
    valid_x_range = PLAYER_WIDTH_RATIO * player_w
    valid_y_range = PLAYER_HEIGHT_RATIO * player_h
    front_boars = []
    back_boars = []

    for boar in boars:
        x_distance = abs(boar["x"] - player_x)
        y_distance = abs(boar["y"] - player_y)
        # 只处理玩家附近范围内的怪物
        if x_distance <= valid_x_range and y_distance <= valid_y_range:
            if boar["x"] > player_x:
                front_boars.append(boar)
            else:
                back_boars.append(boar)

    print(f"[怪物检测] 前方{len(front_boars)}只，后方{len(back_boars)}只，总计{len(front_boars + back_boars)}只")
    return {
        "front": front_boars,
        "back": back_boars,
        "all": front_boars + back_boars
    }


def attack_boar(boar_type, boar_list, img, detections):
    """攻击指定方向的怪物，支持多怪物批量攻击"""
    if not boar_list:
        print(f"[战斗] {boar_type}方向无怪物，停止攻击")
        return

    print(f"[战斗] 开始攻击{boar_type}方向{len(boar_list)}只怪物")
    start_time = time.time()
    max_attack_time = 1.0  # 攻击超时时间

    while not is_paused:
        # 核心优化4：复用传入的检测结果，而非每次循环都重新检测
        player = get_player_state(detections)
        current_boars = get_boar_positions(detections, player["x"], player["y"])[boar_type]
        if not player:
            print("[战斗] 丢失玩家位置，停止攻击")
            break

        # 重新获取当前方向的怪物
        current_boars = get_boar_positions(detections, player["x"], player["y"])[boar_type]
        if not current_boars or (time.time() - start_time) > max_attack_time:
            end_reason = "所有怪物已消灭" if not current_boars else "攻击超时"
            print(f"[战斗] {end_reason}（{boar_type}方向）")
            break

        # 微调方向以覆盖所有怪物
        avg_boar_x = sum(boar["x"] for boar in current_boars) / len(current_boars)
        if boar_type == "front" and player["x"] < avg_boar_x:
            press_key("right", duration=0.1)  # 右微调
        elif boar_type == "back" and player["x"] > avg_boar_x:
            press_key("left", duration=0.1)  # 左微调

        # 攻击时长随怪物数量动态调整
        attack_duration = random.uniform(0.4, 0.8) * (1 + len(current_boars) * 0.1)
        press_key("attack", duration=attack_duration)
        time.sleep(random.uniform(0.05, 0.15))


def handle_combat(detections, player, img):
    """战斗总调度，优先清理前方怪物，再处理后方怪物"""
    if not player:
        return False

    boars = get_boar_positions(detections, player["x"], player["y"])
    if not boars["all"]:
        print("[战斗] 无符合范围的怪物")
        return False

    print(f"[战斗] 进入战斗模式（前方{len(boars['front'])}只，后方{len(boars['back'])}只）")

    # 1. 优先清理前方所有怪物
    if boars["front"]:
        # 修复：补充传递img和detections参数
        attack_boar("front", boars["front"], img, detections)
        # 重新检测，防止后方怪物移动到前方
        img = capture_game_screen()
        detections = detect_objects(img)
        boars = get_boar_positions(detections, player["x"], player["y"])
        if is_paused:
            return False

    # 2. 清理后方所有怪物
    if boars["back"]:
        print(f"[战斗] 前方怪物已清，转向攻击后方{len(boars['back'])}只怪物")
        # 转身面向后方怪物
        avg_back_x = sum(boar["x"] for boar in boars["back"]) / len(boars["back"])
        turn_dir = "left" if player["x"] > avg_back_x else "right"
        press_key(turn_dir, duration=0.2)  # 短时间转身
        time.sleep(0.1)
        # 修复：补充传递img和detections参数
        attack_boar("back", boars["back"], img, detections)

    print("[战斗] 所有怪物处理完成，退出战斗模式")
    return True


def random_movement(player, img, detections):
    """随机移动函数（复用检测结果，仅在移动后更新）

    参数:
        player: 玩家状态字典（位置、尺寸等）
        img: 当前游戏画面截图（复用主循环的截图）
        detections: 当前帧的目标检测结果（复用主循环的检测结果）
    """
    # 若未找到玩家，执行定位移动
    if player is None:
        print("[移动] 未找到玩家，执行定位移动")
        # 随机选择方向短距离移动，尝试重新定位玩家
        direction = random.choice(["left", "right"])
        press_key(direction, duration=random.uniform(0.5, 1.0))
        return

    # 仅在下层平台执行随机移动
    if not player["is_lower"]:
        print("[移动] 玩家在上层平台，不执行随机移动")
        return

    # 移动参数初始化
    total_move_time = random.uniform(2.0, 3.0)  # 总移动时长
    move_fragment = 0.8  # 每次移动片段时长（提升响应灵敏度）
    remaining_time = total_move_time  # 剩余移动时间
    direction = random.choice(["left", "right"])  # 随机初始方向
    print(f"[移动] 开始随机移动（方向：{direction}，总时长：{round(total_move_time, 1)}秒）")

    # 分段移动主循环
    while not is_paused and remaining_time > 0:
        # 执行当前片段移动
        current_move = min(move_fragment, remaining_time)
        print(f"[移动] 移动片段：{round(current_move, 1)}秒（剩余：{round(remaining_time, 1)}秒）")
        press_key(direction, duration=current_move)
        remaining_time -= current_move

        # 移动后必须更新检测结果（场景已变化）
        print("[移动] 移动完成，更新画面和检测结果")
        img = capture_game_screen()  # 重新截图
        if img is None:
            print("[移动] 截图失败，终止移动")
            break

        detections = detect_objects(img)  # 重新检测目标
        player = get_player_state(detections)  # 更新玩家状态

        # 检查玩家是否丢失
        if not player:
            print("[移动] 丢失玩家位置，终止移动")
            break

        # 优先处理爬绳索（高优先级动作）
        if check_rope_priority(detections, player):
            print("[移动] 检测到绳索优先条件，中断移动转爬绳索")
            execute_rope_climb()
            return

        # 检测到怪物则转战斗模式
        boars = get_boar_positions(detections, player["x"], player["y"])
        if boars["all"]:
            print(f"[移动] 检测到{len(boars['all'])}只怪物，中断移动转战斗")
            handle_combat(detections, player, img)
            return

        # 随机切换方向（增加移动随机性）
        if random.random() < 0.3:  # 30%概率切换方向
            direction = "left" if direction == "right" else "right"
            print(f"[移动] 随机切换方向至：{direction}")

        # 模拟人类操作延迟
        simulate_human_behavior()
        time.sleep(random.uniform(0.1, 0.2))

    # 移动结束处理
    print(f"[移动] 随机移动结束（总耗时：{round(total_move_time - remaining_time, 1)}秒）")
    # 移动后小停顿，避免连续操作
    time.sleep(random.uniform(0.5, 1.0))


def check_rope_priority(detections, player):
    """检查是否满足爬绳索的优先条件"""
    global last_rope_climb_time

    # 条件1：距离上次爬绳索已超过设定间隔（10分钟）
    time_since_last_climb = time.time() - last_rope_climb_time
    if time_since_last_climb < ROPE_CLIMB_INTERVAL:
        return False

    # 条件2：检测到绳索
    if not detections["rope"]:
        return False

    # 条件3：角色位于绳索正下方
    rope = detections["rope"][0]
    player_x = player["x"]
    player_y = player["y"]

    # 水平方向判定：角色在绳索x范围±允许偏差内
    rope_x_min = rope["x"] - (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    rope_x_max = rope["x"] + (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    is_x_match = rope_x_min <= player_x <= rope_x_max

    # 垂直方向判定：角色在绳索下方
    rope_bottom_y = rope["y"] - (rope["h"] / 2)
    is_y_below = player_y > rope_bottom_y

    if is_x_match and is_y_below:
        print(f"[爬绳索] 满足优先条件（{round(time_since_last_climb / 60, 1)}分钟未爬），角色在绳索下方")
        return True
    return False


def execute_rope_climb():
    """
    执行爬绳索动作（优化版）
    核心改进：使用多线程实现跳跃键和上键几乎同时按下，确保抓住绳索
    """
    global last_rope_climb_time

    if is_paused:
        return

    print("[爬绳索] 开始执行爬绳索动作（优化版：上键和跳跃键同步按下）")

    try:
        # 获取实际按键映射（从配置中读取）
        actual_jump_key = KEYS["jump"]
        actual_climb_key = KEYS["climb"]

        # 定义两个线程函数，分别负责按下跳跃键和上键
        def press_jump_key():
            """按下跳跃键并保持"""
            pyautogui.keyDown(actual_jump_key)

        def press_climb_key():
            """按下上键并保持"""
            pyautogui.keyDown(actual_climb_key)

        # 创建线程对象
        jump_thread = threading.Thread(target=press_jump_key)
        climb_thread = threading.Thread(target=press_climb_key)

        # 启动线程，实现两个键几乎同时按下
        print("[爬绳索] 同步按下跳跃键和上键以接触绳索")
        jump_thread.start()
        climb_thread.start()

        # 等待两个线程执行完毕（确保按键已按下）
        jump_thread.join()
        climb_thread.join()

        # 保持两个键同时按下状态（确保角色接触并抓住绳索）
        print(f"[爬绳索] 保持按键{ROPE_JUMP_DELAY}秒，确保接触绳索")
        time.sleep(ROPE_JUMP_DELAY)  # 默认0.3秒

        # 释放跳跃键（此时已抓住绳索，只需上键继续攀爬）
        pyautogui.keyUp(actual_jump_key)
        print("[爬绳索] 已抓住绳索，释放跳跃键，继续攀爬")

        # 继续长按上键攀爬指定时长
        time.sleep(ROPE_CLIMB_DURATION)  # 默认1.5秒
        pyautogui.keyUp(actual_climb_key)
        print("[爬绳索] 攀爬完成，释放上键")

        # 更新爬绳索时间戳（避免短时间内重复执行）
        if not is_paused:
            last_rope_climb_time = time.time()
            print(f"[爬绳索] 动作完成，更新上次时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")

        # 攀爬后停顿调整（模拟角色稳定）
        time.sleep(random.uniform(0.2, 0.5))

    except Exception as e:
        print(f"[爬绳索] 动作执行失败：{str(e)}")
    finally:
        # 无论是否发生异常，确保释放所有按键（防止持续按键）
        pyautogui.keyUp(KEYS["jump"])
        pyautogui.keyUp(KEYS["climb"])


def on_key_press(key):
    """键盘热键监听：F12控制暂停/继续"""
    global is_paused
    try:
        if key == keyboard.Key.f12:
            is_paused = not is_paused
            status = "暂停" if is_paused else "继续"
            print(f"\n[系统] 已{status}运行（按F12切换）")
    except AttributeError:
        pass


def start_hotkey_listener():
    """启动热键监听线程"""
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    print("[系统] 热键监听已启动（F12暂停/继续）")
    return listener


def end_key_timer():
    """定时按buff键的独立线程，每5分钟执行一次"""
    global last_end_key_time
    print(f"[定时任务] 每{END_KEY_INTERVAL // 60}分钟按buff键的任务已启动")

    while True:
        # 暂停时不执行
        while is_paused:
            time.sleep(1)

        current_time = time.time()
        # 检查是否达到触发间隔
        if current_time - last_end_key_time >= END_KEY_INTERVAL:
            print(
                f"[定时任务] 触发每{END_KEY_INTERVAL // 60}分钟按buff键（上次执行：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}）")
            press_key("end", duration=0.3)  # 按A键，时长0.3秒
            last_end_key_time = current_time

        time.sleep(30)  # 每30秒检查一次


def main_loop():
    """主循环：按优先级处理爬绳索、战斗、移动（新增检测频率控制）"""
    global is_paused, last_end_key_time, last_rope_climb_time
    print("\n[系统] 开始自动化运行（按F12暂停）")
    print(f"[爬绳索] 初始上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")
    print(f"[定时任务] 初始上次按buff键时间：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}")

    # 检测频率控制参数（新增）
    DETECTION_INTERVAL = 0.5  # 固定检测间隔（秒），可根据需求修改，x秒内检测1次

    while True:
        # 暂停检查
        if is_paused:
            time.sleep(0.5)  # 暂停时降低循环频率
            continue

        # 记录循环开始时间（用于频率控制）
        start_time = time.time()

        # 确保游戏窗口处于激活状态
        if not game_window.isActive:
            print("窗口已失去焦点，重新激活...")
            game_window.activate()

        # 1. 捕获画面与检测目标
        img = capture_game_screen()
        if img is None:
            time.sleep(1)
            # 计算耗时并补足间隔（新增）
            elapsed = time.time() - start_time
            if elapsed < DETECTION_INTERVAL:
                time.sleep(DETECTION_INTERVAL - elapsed)
            continue

        detections = detect_objects(img)
        player = get_player_state(detections)  # 复用检测结果

        if not player:
            print("[警告] 未检测到玩家，重试...")
            random_movement(player, img, detections)
            continue
        else:
            # 2. 优先处理爬绳索（最高优先级）
            if check_rope_priority(detections, player):
                execute_rope_climb()
                time.sleep(0.8)  # 等待角色稳定
            else:
                # 3. 处理战斗
                in_combat = handle_combat(detections, player, img)
                if in_combat:
                    time.sleep(0.3)
                else:
                    # 4. 无战斗则随机移动（仅下层平台）
                    if player["is_lower"]:
                        random_movement(player, img, detections)
                    else:
                        print("[移动] 玩家在上层平台，等待返回下层...")
                        time.sleep(0.5)

        # 计算耗时，补足间隔时间（确保每次检测间隔固定）
        elapsed = time.time() - start_time
        if elapsed < DETECTION_INTERVAL:
            time.sleep(DETECTION_INTERVAL - elapsed)

        # 其他定时任务（如按buff键）
        if time.time() - last_end_key_time > END_KEY_INTERVAL:
            press_key("end")
            last_end_key_time = time.time()


if __name__ == "__main__":
    print("=" * 50)
    print("挂机启动（含优先爬绳索+定时按buff键功能）")
    print("=" * 50)

    # 初始化流程
    if not init_yolo_model():
        exit(1)
    if not init_game_window():
        exit(1)
    hotkey_listener = start_hotkey_listener()

    # 启动定时按buff键的独立线程
    end_key_thread = Thread(target=end_key_timer, daemon=True)
    end_key_thread.start()

    # 启动主循环
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[系统] 手动终止运行")
    finally:
        hotkey_listener.stop()
        sct.close()
        print(
            f"[系统] 资源已释放，脚本退出。\n"
            f"上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}\n"
            f"上次按buff键时间：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}"
        )

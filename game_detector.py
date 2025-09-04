import cv2
import numpy as np
import pygetwindow as gw
import mss
import time
import random
import pyautogui
from pynput import keyboard
from ultralytics import YOLO
from threading import Thread

# -------------------------- 配置参数 --------------------------
GAME_WINDOW_TITLE = "MapleStory Worlds-Artale (繁體中文版)"  # 游戏窗口标题（需根据实际游戏修改）
YOLO_MODEL_PATH = "runs2/train/exp/weights/best.pt"  # 训练好的YOLOv8n模型路径
CONF_THRESHOLD = 0.5  # 目标检测置信度阈值
PLAYER_WIDTH_RATIO = 3  # 定义"三个身位"为角色宽度的3倍
HUMAN_ACTION_PROB = {
    "empty_skill": 0.05,  # 空放技能概率（5%）
    "mistake_jump": 0.03  # 误碰跳跃概率（3%）
}
KEY_PRESS_DURATION = (0.1, 0.3)  # 按键持续时间范围
MOVE_INTERVAL = (1.0, 3.0)  # 随机移动方向切换间隔
# 新增：爬绳索相关配置
ROPE_CLIMB_INTERVAL = 600  # 10分钟（600秒）未爬绳索则优先执行
ROPE_PLAYER_X_RANGE = 30  # 角色x坐标与绳索x坐标的允许偏差（判定是否在下方）
ROPE_CLIMB_DURATION = 1.5  # 爬绳索总时长（持续按上键的时间）
ROPE_JUMP_DELAY = 0.3  # 跳跃后等待接触绳索的延迟时间
# 新增：攻击相关配置（解决攻击欲望低、时长短问题）
ATTACK_DURATION = (0.3, 0.5)  # 技能按键持续时间（延长至0.3-0.5秒）
ATTACK_INTERVAL = (0.03, 0.08)  # 攻击间隔（缩短至0.03-0.08秒，提升频率）
COMBAT_FORCE_ATTACK = True  # 战斗中强制高频攻击（不插入随机移动）
# --------------------------------------------------------------

# 全局变量
is_paused = False  # 暂停标志
game_window = None  # 游戏窗口对象
model = None  # YOLO模型对象
sct = mss.mss()  # 屏幕捕获对象
last_rope_climb_time = time.time()  # 新增：上次爬绳索时间戳（初始为当前时间）

# 键盘按键映射
KEYS = {
    "left": "left",
    "right": "right",
    "jump": "alt",
    "attack": "ctrl",
    "climb": "up"
}


def init_game_window():
    """初始化游戏窗口捕获"""
    global game_window
    try:
        game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
        game_window.activate()
        time.sleep(1)
        print(f"成功找到游戏窗口：{GAME_WINDOW_TITLE}")
        print(f"窗口位置：({game_window.left}, {game_window.top}) 大小：{game_window.width}x{game_window.height}")
        return True
    except IndexError:
        print(f"错误：未找到标题为'{GAME_WINDOW_TITLE}'的窗口，请检查窗口标题是否正确")
        return False


def init_yolo_model():
    """初始化YOLOv8模型"""
    global model
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"成功加载YOLO模型：{YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"加载YOLO模型失败：{str(e)}")
        return False


def capture_game_screen():
    """捕获游戏窗口区域图像"""
    if not game_window:
        return None

    monitor = {
        "top": game_window.top,
        "left": game_window.left,
        "width": game_window.width,
        "height": game_window.height
    }

    try:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img
    except Exception as e:
        print(f"屏幕捕获失败：{str(e)}")
        return None


def detect_objects(img):
    """YOLO目标检测，返回检测结果（分类、坐标、置信度）"""
    if not model or img is None:
        return {"player": [], "boar": [], "rope": [], "coin": []}

    results = model(img, conf=CONF_THRESHOLD, classes=[0, 1, 2, 3])  # 需与模型标签顺序一致
    detections = {"player": [], "boar": [], "rope": [], "coin": []}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # 根据类别分类存储（需确认模型实际cls_id与标签对应关系）
            if cls_id == 1:  # player
                detections["player"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 0:  # boar
                detections["boar"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 2:  # rope
                detections["rope"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})
            elif cls_id == 3:  # coin
                detections["coin"].append({"x": x_center, "y": y_center, "w": width, "h": height, "conf": conf})

    # 筛选每个类别置信度最高的目标
    for cls in detections:
        if len(detections[cls]) > 1:
            detections[cls].sort(key=lambda x: x["conf"], reverse=True)
            detections[cls] = [detections[cls][0]]

    return detections


def press_key(key, duration=None):
    """模拟按键操作"""
    if not key in KEYS:
        return

    actual_key = KEYS[key]
    try:
        press_time = duration if duration else random.uniform(*KEY_PRESS_DURATION)
        pyautogui.keyDown(actual_key)
        time.sleep(press_time)
        pyautogui.keyUp(actual_key)
        time.sleep(random.uniform(0.05, 0.15))  # 按键后小停顿
    except Exception as e:
        print(f"按键操作失败（{key}）：{str(e)}")


def hold_key(key, hold_time):
    """模拟长按按键（用于爬绳索持续按上键）"""
    if not key in KEYS:
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
    """模拟人类行为：空放技能、误碰跳跃"""
    # 随机空放技能（无怪物时）
    if random.random() < HUMAN_ACTION_PROB["empty_skill"]:
        print("[人类模拟] 空放技能")
        press_key("attack")

    # 随机误碰跳跃
    if random.random() < HUMAN_ACTION_PROB["mistake_jump"]:
        print("[人类模拟] 误碰跳跃")
        press_key("jump")


def get_player_state(detections):
    """获取玩家状态：位置、宽度、是否在下层平台"""
    if not detections["player"]:
        return None

    player = detections["player"][0]
    # 判定是否在下层平台（可根据游戏实际地图调整y坐标阈值）
    is_lower_platform = player["y"] > (game_window.height / 2)

    return {
        "x": player["x"],
        "y": player["y"],
        "width": player["w"],
        "is_lower": is_lower_platform
    }


def get_boar_positions(detections, player_x):
    """获取怪物位置，并分类为前方/后方怪物"""
    boars = detections["boar"]
    if not boars or player_x is None:
        return {"front": [], "back": [], "all": []}

    three_body_width = PLAYER_WIDTH_RATIO * detections["player"][0]["w"]  # 三个身位像素距离
    front_boars = []
    back_boars = []

    for boar in boars:
        distance = abs(boar["x"] - player_x)
        if boar["x"] > player_x and distance <= three_body_width:
            front_boars.append(boar)
        elif boar["x"] < player_x and distance <= three_body_width:
            back_boars.append(boar)

    return {
        "front": front_boars,
        "back": back_boars,
        "all": front_boars + back_boars
    }


def attack_boar(boar_type="front"):
    """攻击怪物（持续攻击直到怪物消失）- 优化：提升频率+延长技能时长"""
    print(f"[战斗] 开始攻击{boar_type}怪物")
    start_time = time.time()

    while not is_paused:
        img = capture_game_screen()
        if img is None:
            time.sleep(0.1)  # 画面捕获失败时小停顿重试
            continue

        detections = detect_objects(img)
        player = get_player_state(detections)
        boars = get_boar_positions(detections, player["x"] if player else None)

        # 终止攻击的条件：怪物消失/超时（10秒）/玩家丢失
        if (not boars[boar_type]) or (time.time() - start_time) > 10 or (not player):
            end_reason = "怪物已消灭" if not boars[boar_type] else \
                "攻击超时" if (time.time() - start_time) > 10 else "玩家丢失"
            print(f"[战斗] {end_reason}，停止攻击{boar_type}怪物")
            break

        # 高频攻击：使用新配置的攻击时长和间隔
        press_key("attack", duration=random.uniform(*ATTACK_DURATION))
        time.sleep(random.uniform(*ATTACK_INTERVAL))


def handle_combat(detections, player):
    """战斗逻辑处理：判断怪物位置并执行攻击策略 - 优化：补充怪物消失后补刀"""
    if not player:
        return False

    boars = get_boar_positions(detections, player["x"])
    combat_triggered = False  # 标记是否触发过战斗

    # 1. 前后均有怪物：先灭前方再灭后方
    if boars["front"] and boars["back"]:
        print("[战斗] 前后均有怪物，先攻击前方")
        attack_boar("front")
        if not is_paused and boars["back"]:  # 前方打完后检查后方是否仍有怪物
            press_key("left" if player["x"] > boars["back"][0]["x"] else "right", duration=0.2)
            attack_boar("back")
        combat_triggered = True

    # 2. 仅前方有怪物
    elif boars["front"]:
        attack_boar("front")
        combat_triggered = True

    # 3. 仅后方有怪物：转身攻击
    elif boars["back"]:
        print("[战斗] 后方有怪物，转身攻击")
        press_key("left" if player["x"] > boars["back"][0]["x"] else "right", duration=0.2)
        attack_boar("back")
        combat_triggered = True

    # 4. 新增：战斗刚结束时补刀1-2次（提升攻击欲望）
    if combat_triggered and not (boars["front"] or boars["back"]):
        print("[战斗] 怪物消失，补刀确认")
        for _ in range(random.randint(1, 2)):  # 补1-2次攻击
            if is_paused:
                break
            press_key("attack", duration=random.uniform(*ATTACK_DURATION))
            time.sleep(random.uniform(*ATTACK_INTERVAL))

    return combat_triggered


def random_movement(player):
    """一般状态：下层平台随机左右移动 - 优化：战斗中立即停止移动"""
    if not player or not player["is_lower"]:
        return

    direction = random.choice(["left", "right"])
    move_duration = random.uniform(*MOVE_INTERVAL)
    start_time = time.time()

    print(f"[移动] 开始向{direction}移动（持续{round(move_duration, 1)}秒）")

    while not is_paused and (time.time() - start_time) < move_duration:
        # 移动中检测战斗需求：一旦触发战斗，立即停止移动并专注攻击
        img = capture_game_screen()
        detections = detect_objects(img)
        if handle_combat(detections, player):
            print(f"[移动] 检测到战斗，停止向{direction}移动")
            return  # 战斗后不继续移动，直接返回主循环

        # 移动中检测爬绳索需求（若满足则中断移动）
        if check_rope_priority(detections, player):
            execute_rope_climb()
            return

        # 非战斗时正常移动（不影响攻击逻辑）
        press_key(direction, duration=random.uniform(0.2, 0.4))
        time.sleep(random.uniform(0.1, 0.3))
        simulate_human_behavior()


# 新增：检查是否满足优先爬绳索条件
def check_rope_priority(detections, player):
    """
    判定是否优先爬绳索：
    1. 距离上次爬绳索超过10分钟
    2. 检测到绳索
    3. 角色在绳索正下方（x坐标偏差在允许范围内，y坐标在绳索下方）
    """
    global last_rope_climb_time

    # 条件1：10分钟未爬绳索
    time_since_last_climb = time.time() - last_rope_climb_time
    if time_since_last_climb < ROPE_CLIMB_INTERVAL:
        # 调试信息：显示剩余时间（可选）
        # print(f"[爬绳索] 未达优先条件，剩余时间：{round((ROPE_CLIMB_INTERVAL - time_since_last_climb)/60,1)}分钟")
        return False

    # 条件2：检测到绳索
    if not detections["rope"]:
        # print("[爬绳索] 未检测到绳索")
        return False

    # 条件3：角色在绳索下方
    rope = detections["rope"][0]
    player_x = player["x"]
    player_y = player["y"]

    # x坐标判定：角色在绳索x范围±允许偏差内
    rope_x_min = rope["x"] - (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    rope_x_max = rope["x"] + (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    is_x_match = rope_x_min <= player_x <= rope_x_max

    # y坐标判定：角色在绳索下方（绳索y中心 - 绳索高度/2 > 角色y中心）
    rope_bottom_y = rope["y"] - (rope["h"] / 2)  # 绳索底部y坐标
    is_y_below = player_y > rope_bottom_y

    if is_x_match and is_y_below:
        print(f"[爬绳索] 满足优先条件（{round(time_since_last_climb / 60, 1)}分钟未爬），角色在绳索下方")
        return True
    else:
        # print(f"[爬绳索] 角色不在绳索下方（x匹配：{is_x_match}，y下方：{is_y_below}）")
        return False


# 新增：执行爬绳索动作（跳跃+长按上键）
def execute_rope_climb():
    """执行爬绳索动作：跳跃→等待接触→长按上键爬至上层"""
    global last_rope_climb_time

    if is_paused:
        return

    print("[爬绳索] 开始执行爬绳索动作")

    # 1. 跳跃（确保接触绳索）
    print("[爬绳索] 跳跃以接触绳索")
    press_key("jump", duration=random.uniform(0.2, 0.3))
    time.sleep(ROPE_JUMP_DELAY)  # 等待角色上升接触绳索

    # 2. 长按上键爬绳索（持续指定时长）
    if not is_paused:
        hold_key("climb", ROPE_CLIMB_DURATION)

    # 3. 爬绳索后更新时间戳
    if not is_paused:
        last_rope_climb_time = time.time()
        print(f"[爬绳索] 动作完成，更新上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")

    # 4. 爬上上层后小停顿（模拟角色调整）
    time.sleep(random.uniform(1.0, 2.0))


def on_key_press(key):
    """键盘热键监听：F12暂停/继续"""
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


def main_loop():
    """主循环：优先爬绳索→战斗→随机移动 - 优化：新增强制战斗攻击逻辑"""
    print("\n[系统] 开始自动化运行（按F12暂停）")
    print(f"[爬绳索] 初始上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")

    while True:
        # 暂停检查
        while is_paused:
            time.sleep(0.5)

        # 1. 捕获画面与检测目标
        img = capture_game_screen()
        if img is None:
            time.sleep(1)
            continue
        detections = detect_objects(img)
        player = get_player_state(detections)

        if not player:
            print("[警告] 未检测到玩家，重试...")
            time.sleep(1)
            continue

        # 2. 优先处理：10分钟未爬绳索且在绳索下方
        if check_rope_priority(detections, player):
            execute_rope_climb()
            # 爬上上层后，等待角色稳定再继续循环
            time.sleep(2)
            continue

        # 3. 新增：强制战斗攻击（即使handle_combat未触发，也检查怪物并攻击）
        boars = get_boar_positions(detections, player["x"])
        if COMBAT_FORCE_ATTACK and (boars["front"] or boars["back"]):
            # 直接判断怪物位置并攻击，跳过中间逻辑
            target_type = "front" if boars["front"] else "back"
            attack_boar(target_type)
            time.sleep(0.5)
            continue

        # 4. 原有战斗逻辑（兼容历史逻辑）
        in_combat = handle_combat(detections, player)
        if in_combat:
            time.sleep(0.5)
            continue

        # 5. 无战斗则随机移动（仅下层平台）
        if player["is_lower"]:
            random_movement(player)
        else:
            print("[移动] 玩家在上层平台，等待返回下层...")
            # 上层平台可扩展：检测是否需要爬绳索返回下层（可选）
            time.sleep(2)


if __name__ == "__main__":
    print("=" * 50)
    print("2D游戏自动化脚本启动（含优先爬绳索功能）")
    print("=" * 50)

    # 初始化流程
    if not init_yolo_model():
        exit(1)
    if not init_game_window():
        exit(1)
    hotkey_listener = start_hotkey_listener()

    # 启动主循环
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[系统] 手动终止运行")
    finally:
        hotkey_listener.stop()
        sct.close()
        print(
            f"[系统] 资源已释放，脚本退出。上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")
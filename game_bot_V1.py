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
CONF_THRESHOLD = 0.6  # 目标检测置信度阈值
PLAYER_WIDTH_RATIO = 4  # 怪物检测x范围（玩家身位，覆盖多怪物）
PLAYER_HEIGHT_RATIO = 0.8  # 怪物检测y范围（玩家身高，过滤偏移）
HUMAN_ACTION_PROB = {
    "empty_skill": 0.05,  # 空放技能概率（5%）
    "mistake_jump": 0.03  # 误碰跳跃概率（3%）
}
KEY_PRESS_DURATION = (0.08, 0.2)  # 按键持续时间范围
# 新增：爬绳索相关配置
ROPE_CLIMB_INTERVAL = 600  # 10分钟（600秒）未爬绳索则优先执行
ROPE_PLAYER_X_RANGE = 30  # 角色x坐标与绳索x坐标的允许偏差（判定是否在下方）
ROPE_CLIMB_DURATION = 1.5  # 爬绳索总时长（持续按上键的时间）
ROPE_JUMP_DELAY = 0.3  # 跳跃后等待接触绳索的延迟时间
MOVE_INTERVAL = (2.0, 3.0)  # 随机移动方向切换间隔
# 新增：定时按A键配置
END_KEY_INTERVAL = 300  # 每5分钟（300秒）按一次A键
# --------------------------------------------------------------

# 全局变量
is_paused = False  # 暂停标志
game_window = None  # 游戏窗口对象
model = None  # YOLO模型对象
sct = mss.mss()  # 屏幕捕获对象
last_rope_climb_time = time.time()  # 上次爬绳索时间戳（初始为当前时间）
last_end_key_time = time.time()  # 上次按end键时间戳（初始为当前时间）

# 键盘按键映射
KEYS = {
    "left": "left",
    "right": "right",
    "jump": "alt",
    "attack": "ctrl",
    "climb": "up",
    "end": "end"  # 新增end键映射
}


def init_game_window():
    """初始化游戏窗口捕获"""
    global game_window
    try:
        game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
        if not game_window.isActive:
            game_window.activate()  # 激活窗口
        if game_window.isMinimized:
            game_window.restore()  # 若最小化则还原
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

    results = model(img, conf=CONF_THRESHOLD, classes=[0, 1, 2, 3], verbose=False)  # 需与模型标签顺序一致
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

    return detections


def press_key(key, duration=None):
    """模拟按键操作"""
    if key not in KEYS:
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


def get_boar_positions(detections, player_x, player_y):
    """获取怪物位置，并分类为前方/后方怪物"""
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


def attack_boar(boar_type="front", boar_list=None):
    """
    攻击指定方向怪物
    优化点：1. 支持多怪物批量攻击 2. 动态微调方向覆盖所有怪物 3. 缩短超时时间适配怪物移动
    """
    if not boar_list:
        print(f"[战斗] {boar_type}方向无怪物，停止攻击")
        return

    print(f"[战斗] 开始攻击{boar_type}方向{len(boar_list)}只怪物")
    start_time = time.time()
    max_attack_time = 4.0  # 多怪物超时延长至4秒

    while not is_paused:
        # 实时更新怪物位置（防止多怪物移动后漏检）
        img = capture_game_screen()
        detections = detect_objects(img)
        player = get_player_state(detections)
        if not player:
            print("[战斗] 丢失玩家位置，停止攻击")
            break

        # 重新获取当前方向怪物（避免攻击已离开的怪物）
        current_boars = get_boar_positions(detections, player["x"], player["y"])[boar_type]
        if not current_boars or (time.time() - start_time) > max_attack_time:
            end_reason = "所有怪物已消灭" if not current_boars else "攻击超时"
            print(f"[战斗] {end_reason}（{boar_type}方向）")
            break

        # 多怪物方向微调：按怪物平均x坐标方向，覆盖所有怪物
        avg_boar_x = sum(boar["x"] for boar in current_boars) / len(current_boars)
        if boar_type == "front" and player["x"] < avg_boar_x:
            press_key("right", duration=0.1)  # 右微调
        elif boar_type == "back" and player["x"] > avg_boar_x:
            press_key("left", duration=0.1)  # 左微调

        # 攻击时长随怪物数量动态调整
        attack_duration = random.uniform(0.4, 0.8) * (1 + len(current_boars) * 0.1)
        press_key("attack", duration=attack_duration)
        time.sleep(random.uniform(0.05, 0.15))


def handle_combat(detections, player):
    """
    战斗总调度
    优化点：1. 分批次清场（前方优先） 2. 战斗中实时新怪物位置 3. 增加转身适配
    """
    if not player:
        return False

    boars = get_boar_positions(detections, player["x"], player["y"])
    if not boars["all"]:
        print("[战斗] 无符合范围的怪物")
        return False

    print(f"[战斗] 进入战斗模式（前方{len(boars['front'])}只，后方{len(boars['back'])}只）")

    # 1. 优先清前方所有怪物（威胁更高）
    if boars["front"]:
        attack_boar("front", boars["front"])
        # 攻击后重新检测：防止后方怪物移动到前方
        img = capture_game_screen()
        detections = detect_objects(img)
        boars = get_boar_positions(detections, player["x"], player["y"])
        if is_paused:
            return False

    # 2. 再清后方所有怪物（前方清完后转场）
    if boars["back"]:
        print(f"[战斗] 前方怪物已清，转向攻击后方{len(boars['back'])}只怪物")
        # 按后方怪物平均位置转身（确保面向怪物）
        avg_back_x = sum(boar["x"] for boar in boars["back"]) / len(boars["back"])
        turn_dir = "left" if player["x"] > avg_back_x else "right"
        press_key(turn_dir, duration=0.2)  # 短时间转身到位
        time.sleep(0.1)
        attack_boar("back", boars["back"])

    print("[战斗] 所有怪物处理完成，退出战斗模式")
    return True


def random_movement(player):
    """
    下层平台随机移动
    优化点：1. 长时移动拆分为片段 2. 移除递归改用迭代 3. 移动后立即检测多怪物
    """
    direction = random.choice(["left", "right"])
    move_duration = random.uniform(*MOVE_INTERVAL)
    start_time = time.time()
    while player is None:
        # 角色卡墙角移动
        img = capture_game_screen()
        detections = detect_objects(img)
        player = get_player_state(detections)
        print("detection or player not found continue")
        print(f"[移动] 未找到角色，开始向{direction}移动（持续{round(move_duration, 1)}秒）")
        press_key(random.choice(["left", "right"]), duration=move_duration)  # 没人移动
        print(f"[移动] 移动完成")
        continue

    if not player or not player["is_lower"]:
        return

    # 单次移动总时长2-3秒，拆分为0.8秒片段（提升响应）
    total_move_time = random.uniform(2.0, 3.0)
    move_fragment = 0.8  # 每个移动片段时长
    remaining_time = total_move_time
    direction = random.choice(["left", "right"])
    print(f"[移动] 开始向{direction}移动（总时长{round(total_move_time, 1)}秒，分段执行）")

    # 迭代执行“移动-检测”
    while not is_paused and remaining_time > 0:
        # 执行短时移动（取剩余时间和片段时长的较小值）
        current_move = min(move_fragment, remaining_time)
        press_key(direction, duration=current_move)
        remaining_time -= current_move

        # 移动后立即检测
        img = capture_game_screen()
        detections = detect_objects(img)
        player = get_player_state(detections)
        if not player:
            print("[移动] 丢失玩家位置，停止移动")
            break

        # 优先处理爬绳索（高优先级保留）
        if check_rope_priority(detections, player):
            execute_rope_climb()
            return

        # 检测到任何怪物立即中断移动
        boars = get_boar_positions(detections, player["x"], player["y"])
        if boars["all"]:
            print(f"[移动] 检测到{len(boars['all'])}只怪物，中断移动转攻击")
            handle_combat(detections, player)
            return  # 攻击后重新进入主循环

        simulate_human_behavior()
        time.sleep(random.uniform(0.1, 0.2))

    print(f"[移动] 完成向{direction}的移动（剩余时间{round(remaining_time, 1)}秒）")


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
        return False

    # 条件2：检测到绳索
    if not detections["rope"]:
        return False

    # 条件3：角色在绳索下方
    rope = detections["rope"][0]
    player_x = player["x"]
    player_y = player["y"]

    # x坐标判定：角色在绳索x范围±允许偏差内
    rope_x_min = rope["x"] - (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    rope_x_max = rope["x"] + (rope["w"] / 2 + ROPE_PLAYER_X_RANGE)
    is_x_match = rope_x_min <= player_x <= rope_x_max

    # y坐标判定：角色在绳索下方（绳索底部y坐标 < 角色y中心）
    rope_bottom_y = rope["y"] - (rope["h"] / 2)
    is_y_below = player_y > rope_bottom_y

    if is_x_match and is_y_below:
        print(f"[爬绳索] 满足优先条件（{round(time_since_last_climb / 60, 1)}分钟未爬），角色在绳索下方")
        return True
    return False


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


def end_key_timer():
    """定时按end键的独立线程函数：每5分钟执行一次"""
    global last_end_key_time
    print(f"[定时任务] 每{END_KEY_INTERVAL // 60}分钟按end键的任务已启动")

    while True:
        # 暂停时不执行按键
        while is_paused:
            time.sleep(1)

        # 检查是否到达按键间隔
        current_time = time.time()
        if current_time - last_end_key_time >= END_KEY_INTERVAL:
            print(
                f"[定时任务] 触发每{END_KEY_INTERVAL // 60}分钟按end键（上次执行：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}）")
            press_key("A", duration=0.1)  # 按A键，固定0.1秒时长
            last_end_key_time = current_time

        time.sleep(30)  # 每30秒检查一次，避免频繁占用资源


def main_loop():
    """主循环：优先爬绳索→战斗→随机移动"""
    print("\n[系统] 开始自动化运行（按F12暂停）")
    print(f"[爬绳索] 初始上次爬绳索时间：{time.strftime('%H:%M:%S', time.localtime(last_rope_climb_time))}")
    print(f"[定时任务] 初始上次按A键时间：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}")

    while True:
        if not game_window.isActive:
            print("窗口已失去焦点，重新激活...")
            game_window.activate()

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
            random_movement(player)
            continue

        # 2. 优先处理：10分钟未爬绳索且在绳索下方
        if check_rope_priority(detections, player):
            execute_rope_climb()
            time.sleep(2)  # 等待角色稳定
            continue

        # 3. 处理战斗
        in_combat = handle_combat(detections, player)
        if in_combat:
            time.sleep(0.5)
            continue

        # 4. 无战斗则随机移动（仅下层平台）
        if player["is_lower"]:
            random_movement(player)
        else:
            print("[移动] 玩家在上层平台，等待返回下层...")
            time.sleep(2)


if __name__ == "__main__":
    print("=" * 50)
    print("挂机启动（含优先爬绳索+定时按A键功能）")
    print("=" * 50)

    # 初始化流程
    if not init_yolo_model():
        exit(1)
    if not init_game_window():
        exit(1)
    hotkey_listener = start_hotkey_listener()

    # 启动定时按A键的独立线程（不阻塞主循环）
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
            f"上次按end键时间：{time.strftime('%H:%M:%S', time.localtime(last_end_key_time))}"
        )

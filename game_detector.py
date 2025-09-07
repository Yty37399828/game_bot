import cv2
import numpy as np
import pygetwindow as gw
from mss import mss
from ultralytics import YOLO
import time


def detect_game_window(model_path, window_title_keyword):
    # 加载训练好的模型
    model = YOLO(model_path)

    # 查找目标窗口
    windows = gw.getWindowsWithTitle(window_title_keyword)
    if not windows:
        raise ValueError(f"未找到标题包含 '{window_title_keyword}' 的窗口")

    # 选择第一个匹配的窗口
    game_window = windows[0]
    if game_window.isMinimized:
        game_window.restore()  # 恢复窗口
    game_window.activate()  # 激活窗口
    time.sleep(1)  # 等待窗口激活

    # 配置屏幕捕获区域
    monitor = {
        "top": game_window.top,
        "left": game_window.left,
        "width": game_window.width,
        "height": game_window.height
    }

    # 初始化屏幕捕获
    sct = mss()

    # 检测控制变量
    running = True  # 是否运行检测
    paused = False  # 是否暂停
    frame_count = 0  # 帧计数器
    window_scale = 1.0  # 窗口缩放比例，初始为100%

    # 检测频率设置 (每隔多少帧检测一次)
    freq_options = {
        'high': 1,  # 高频率：每帧都检测
        'medium': 3,  # 中频率：每3帧检测一次
        'low': 6  # 低频率：每6帧检测一次
    }
    current_freq = 'high'  # 默认高频率
    freq_step = freq_options[current_freq]

    # 创建显示窗口并设置为可调整大小
    cv2.namedWindow("游戏窗口目标检测", cv2.WINDOW_NORMAL)
    # 设置初始窗口大小为游戏窗口的80%
    init_width = int(game_window.width * 0.8)
    init_height = int(game_window.height * 0.8)
    cv2.resizeWindow("游戏窗口目标检测", init_width, init_height)

    # 精简的状态显示信息
    status_text = "P-pause, Q-quit, +/- scale"

    try:
        print("控制说明:")
        print("  P键: 开始/暂停检测 | Q键: 退出程序")
        print("  1/2/3键: 切换高/中/低频率 | +/-键: 调整窗口大小")

        while running:
            # 捕获窗口画面
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换为OpenCV格式

            # 根据状态和频率决定是否进行检测
            detected_frame = frame.copy()
            if not paused:
                frame_count += 1
                if frame_count % freq_step == 0:
                    # 模型推理
                    results = model(frame, conf=0.5)  # 置信度阈值
                    detected_frame = results[0].plot(
                        conf=True,  # 显示置信度
                        labels=True,  # 显示标签
                        line_width=2  # 边框线宽
                    )
                    frame_count = 0  # 重置计数器
            else:
                # 暂停状态显示提示
                cv2.putText(detected_frame, "Paused",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)   #fontScale为字体大小调整

            # 应用窗口缩放
            if window_scale != 1.0:
                new_width = int(detected_frame.shape[1] * window_scale)
                new_height = int(detected_frame.shape[0] * window_scale)
                detected_frame = cv2.resize(detected_frame, (new_width, new_height))

            # 显示精简状态信息（底部居中）
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (detected_frame.shape[1] - text_size[0]) // 2
            cv2.putText(detected_frame, status_text,
                        (text_x, detected_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示结果窗口
            cv2.imshow("游戏窗口目标检测", detected_frame)

            # 键盘事件处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                running = False
            elif key == ord('p'):  # 暂停/继续
                paused = not paused
            elif key == ord('1'):  # 高频率
                current_freq = 'high'
                freq_step = freq_options[current_freq]
            elif key == ord('2'):  # 中频率
                current_freq = 'medium'
                freq_step = freq_options[current_freq]
            elif key == ord('3'):  # 低频率
                current_freq = 'low'
                freq_step = freq_options[current_freq]
            elif key == ord('+') or key == ord('='):  # 增大窗口
                window_scale = min(2.0, window_scale + 0.1)
                new_width = int(frame.shape[1] * window_scale)
                new_height = int(frame.shape[0] * window_scale)
                cv2.resizeWindow("游戏窗口目标检测", new_width, new_height)
            elif key == ord('-'):  # 减小窗口
                window_scale = max(0.2, window_scale - 0.1)
                new_width = int(frame.shape[1] * window_scale)
                new_height = int(frame.shape[0] * window_scale)
                cv2.resizeWindow("游戏窗口目标检测", new_width, new_height)

    except Exception as e:
        print(f"检测过程中出错: {e}")
    finally:
        # 释放资源
        sct.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_PATH = "runs2/train/exp/weights/best.pt"
    # 替换为步骤1中记录的完整标题（如“植物大战僵尸 中文版 v1.0”）
    FULL_WINDOW_TITLE = "MapleStory Worlds-Artale (繁體中文版)"
    detect_game_window(MODEL_PATH, FULL_WINDOW_TITLE)

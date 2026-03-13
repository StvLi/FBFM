import base64
import io
import json
import time
import os
import sys
import numpy as np
import cv2
import datetime
from typing import Tuple, Union, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# 从新机器人SDK导入
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera

# 添加路径以导入 WebSocket 客户端
lingbot_va_root = Path(__file__).parent.parent
sys.path.insert(0, str(lingbot_va_root))
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

# --- 全局配置 ---
CONTROL_MODE = 'eepose'  
# WebSocket 服务器配置
WS_HOST = "127.0.0.1"  # 根据实际情况修改
WS_PORT = 29056  # 根据实际情况修改，对应 launch_server.sh 中的 START_PORT
LOG_IMAGE_DIR = "./Log_New"  # 保存周期性图像日志的文件夹

CAMERA_MAPPING = {
    "cam_head": "head",
    # "cam_high_fisheye": "head_center_fisheye",
    "cam_left_wrist": "hand_left",
    "cam_right_wrist": "hand_right",
}

# 相机名称映射：从 a2d_client 的命名到 server 期望的命名
CAMERA_NAME_MAPPING = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
INSTRUCTION = "pick up the first cup from left to right and place it in the appropriate position and pour the coffee into the cup"
class PrettyActionLogger:
    def __init__(self, ins, save_dir="./action_logs"):
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        # 按时间生成新文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(save_dir, f"actions_{ins}_{timestamp}.jsonl")
        # 自动累积序号
        self.counter = 0
        print(f"[INIT] Pretty action log → {self.filepath}")
    def append_action(self, actions):
        arr = np.array(actions)
        # if arr.shape not in [(9, 30), (14, 30)]:
        #     raise ValueError(f"shape must be (9,30) or (14,30), got {arr.shape}")
        self.counter += 1
        rows = [
            " ".join(map(str, row))     # 例如 "1 2 3 4 ... 30"
            for row in arr
        ]
        record = {
            "index": self.counter,
            "shape": list(arr.shape),
            "rows": rows
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"[OK] appended action #{self.counter}")

def clear_log_directory():
    """清空 LOG_IMAGE_DIR 文件夹中的所有文件"""
    if os.path.exists(LOG_IMAGE_DIR):
        for filename in os.listdir(LOG_IMAGE_DIR):
            file_path = os.path.join(LOG_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"警告：无法删除文件 {file_path}: {e}")
    else:
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 已创建文件夹: {LOG_IMAGE_DIR}")

def get_image(camera: CosineCamera, cam_sdk_name: str) -> Optional[np.ndarray]:
    """
    从指定的摄像头获取图像，返回 numpy 数组 (H, W, 3) uint8 RGB 格式。
    """
    try:
        img, _ = camera.get_latest_image(cam_sdk_name)
        if img is not None and img.size > 0:
            # 确保是 RGB 格式，OpenCV 默认是 BGR
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 调整大小（可选，服务器会自己 resize）
            # img = cv2.resize(img, (320, 256))  # 根据 config 中的 width, height
            return img.astype(np.uint8)
        else:
            print(f"警告：无法获取 {cam_sdk_name} 的图像，或者图像为空。")
            return None
    except Exception as e:
        print(f"警告：获取 {cam_sdk_name} 图像时发生异常: {e}")
        return None

def format_obs(images_dict: dict) -> dict:
    """
    将图像格式化为服务器期望的格式。
    
    Args:
        images_dict: 字典，key 为 cam_head, cam_left_wrist, cam_right_wrist
    
    Returns:
        格式化的观测字典，包含：
        - observation.images.cam_high
        - observation.images.cam_left_wrist
        - observation.images.cam_right_wrist
    """
    formatted = {}
    
    # 添加相机图像
    for client_key, server_key in CAMERA_NAME_MAPPING.items():
        if client_key in images_dict and images_dict[client_key] is not None:
            formatted[server_key] = images_dict[client_key]
        else:
            print(f"警告：缺少相机图像 {client_key}")
    
    return formatted

def main():
    """主程序：连接机器人，并进入主控制循环"""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"📂 已创建文件夹: {LOG_IMAGE_DIR}")
        # 清空 Log 文件夹
    print("🧹 清空日志文件夹...")
    clear_log_directory()

    robot_dds = None
    robot_controller = None
    camera = None

    try:
        # --- 1. 初始化机器人和相机 ---
        print("🤖 初始化机器人系统...")
        robot_dds = RobotDds()
        robot_controller = RobotController()
        
        # 从 CAMERA_MAPPING 获取所有需要使用的摄像头SDK名称
        camera_sdk_names = list(CAMERA_MAPPING.values())
        print(f"📷 正在初始化摄像头: {camera_sdk_names}")
        camera = CosineCamera(camera_sdk_names)
        # arm_initial_joint_position=[-1.63665295,  0.78416812,  0.61188424, -0.70639342,  1.04935575,
        # 1.44077671,  0.72583276,  1.77511871, -0.99129957, -1.53809536,
        # 0.63575584, -0.19526549, -1.14260852, -0.98163235]
        arm_initial_joint_position=[-1.075, 0.6108, 0.279, -1.284, 0.731, 1.495, -0.188,
                                    1.075, -0.6108, -0.279, 1.284, -0.731, -1.495, 0.188] ##和数采位置对齐
        
        robot_dds.reset(arm_positions=arm_initial_joint_position,
                        gripper_positions=[0.0,0.0],
                        hand_positions=robot_dds.hand_initial_joint_position,
                        waist_positions=robot_dds.waist_initial_joint_position,
                        head_positions=robot_dds.head_initial_joint_position
                        )
        print("✅ 系统初始化完成！")
        print("🚀 连接 WebSocket 推理服务器...")
        
        # --- 初始化 WebSocket 客户端 ---
        try:
            model = WebsocketClientPolicy(host=WS_HOST, port=WS_PORT)
            print(f"✅ WebSocket 连接成功: ws://{WS_HOST}:{WS_PORT}")
        except Exception as e:
            print(f"❌ WebSocket 连接失败: {e}")
            raise

        instruction = INSTRUCTION
        logger = PrettyActionLogger(instruction)
        
        # --- 发送 Reset 请求 ---
        print(f"📝 设置任务指令: {instruction}")
        try:
            ret = model.infer(dict(reset=True, prompt=instruction))
            print("✅ 服务器已重置并设置指令")
        except Exception as e:
            print(f"❌ Reset 请求失败: {e}")
            raise

        # 用于保存历史观测（用于 KV cache）
        obs_history = []
        first_inference = True

        # --- 2. 主控制循环 ---
        while True:
            time.sleep(1)
            print("\n" + "="*50)
            
            # --- 2.1. 获取状态和图像 ---
            try:
                # 获取手臂末端执行器位姿, 并整理为 "右+左" 顺序
                motion_status = robot_controller.get_motion_status()
                left_cartesian = motion_status["frames"]["arm_left_link7"]
                right_cartesian = motion_status["frames"]["arm_right_link7"]
                #a2d_sdk.gripper_states() 返回 ([左爪状态, 右爪状态], [时间戳]) 待确认
                gripper_states_raw, _ = robot_dds.gripper_states() 
                left_gripper_state = gripper_states_raw[0]
                right_gripper_state = gripper_states_raw[1]

                eef_pose_state = [
                    right_cartesian["position"]["x"], right_cartesian["position"]["y"], right_cartesian["position"]["z"],
                    right_cartesian["orientation"]["quaternion"]["x"], right_cartesian["orientation"]["quaternion"]["y"],right_cartesian["orientation"]["quaternion"]["z"],right_cartesian["orientation"]["quaternion"]["w"],right_gripper_state,
                    left_cartesian["position"]["x"], left_cartesian["position"]["y"],  left_cartesian["position"]["z"],
                    left_cartesian["orientation"]["quaternion"]["x"], left_cartesian["orientation"]["quaternion"]["y"], left_cartesian["orientation"]["quaternion"]["z"], left_cartesian["orientation"]["quaternion"]["w"],left_gripper_state
                ]
                

            except (KeyError, IndexError) as e:
                 print(f"❌ 获取机器人状态失败: {e}。跳过本轮循环。")
                 continue
            except Exception as e:
                 print(f"❌ 获取机器人状态时发生未知错误: {e}。跳过本轮循环。")
                 continue
            
            # 获取图像
            images_dict = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, sdk_name in CAMERA_MAPPING.items():
                raw_img = get_image(camera, sdk_name)
                if raw_img is not None:
                    images_dict[server_name] = raw_img
                    print(f"✅ 已获取图像: {server_name}, shape: {raw_img.shape}")
                    # 保存图像日志
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    # 保存时需要转换回 BGR
                    img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(log_path, img_bgr)
                    latest_path = f"./{server_name}_latest.png"
                    cv2.imwrite(latest_path, img_bgr)
                    print(f"[Saved] {latest_path}")
                else:
                    print(f"⚠️ 无法获取图像: {server_name}")

            # 检查是否所有图像都获取成功
            if len(images_dict) < len(CAMERA_MAPPING):
                print(f"⚠️ 部分图像获取失败，跳过本轮推理")
                continue

            # --- 2.2. 格式化观测数据 ---
            formatted_obs = format_obs(images_dict)
            
            # --- 2.3. 发送推理请求 ---
            try:
                print(f"🚀 正在发送推理请求...")
                # 发送单帧观测（服务器内部会处理为 list）
                # 注意：prompt 只在 reset 时使用，正常推理时使用 reset 时设置的 prompt
                ret = model.infer(dict(obs=formatted_obs))
                
                if "action" not in ret:
                    print(f"[!] 服务器未返回 action，响应: {ret}")
                    continue
                    
                action = ret['action']
                print(f"✅ 收到 action, shape: {action.shape if hasattr(action, 'shape') else type(action)}")
                
            except Exception as e:
                print(f"[!] 推理请求失败: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                continue

            # --- 2.4. 解析并执行动作 ---
            if CONTROL_MODE == 'eepose':
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                
                logger.append_action(action)
                
                # 根据 eval_polict_client_openpi.py 的处理方式
                # action 的 shape 是 (C, F, H)
                # C: len(used_action_channel_ids) = 14
                # F: frame_chunk_size = 2
                # H: action_per_frame = 16
                print(f"Action shape: {action.shape}")
                
                if len(action.shape) != 3:
                    print(f"[!] 不支持的 action shape: {action.shape}，应为 (C, F, H)")
                    continue
                
                action_dim, num_frames, num_steps = action.shape
                
                # 根据 eval 代码，action_per_frame = num_steps // 4
                assert num_steps % 4 == 0, f"num_steps ({num_steps}) 必须能被 4 整除"
                action_per_frame = num_steps // 4
                
                # 保存初始位姿（用于相对动作）
                initial_eef_pose = np.array(eef_pose_state, dtype=np.float64)
                
                # 初始化 key_frame_list，用于 KV cache
                key_frame_list = []
                
                # 遍历每个 frame（从第二个 frame 开始，第一个 frame 用于初始化）
                # 第一次推理时跳过第一个 frame，后续推理从第一个 frame 开始
                start_idx = 1 if first_inference else 0
                for i in range(start_idx, num_frames):
                    # 遍历每个 step
                    for j in range(num_steps):
                        # 提取当前动作: (action_dim,)
                        ee_action = action[:, i, j]
                        
                        # 根据 action_dim 处理
                        if action_dim == 14:
                            # 14维：需要转换为16维（左右各8维：xyz+quat+gripper）
                            # 格式：[right_xyz(3), right_euler(3), right_gripper(1), 
                            #        left_xyz(3), left_euler(3), left_gripper(1)]
                            # 需要转换为 quaternion
                            
                            # 右臂
                            right_xyz = ee_action[:3]
                            right_euler = ee_action[3:6]
                            right_gripper = ee_action[6:7]
                            right_quat = R.from_euler('xyz', right_euler).as_quat()  # [x, y, z, w]
                            
                            # 左臂
                            left_xyz = ee_action[7:10]
                            left_euler = ee_action[10:13]
                            left_gripper = ee_action[13:14]
                            left_quat = R.from_euler('xyz', left_euler).as_quat()
                            
                            # 转换为绝对位姿（相对于初始位姿）
                            # 这里简化处理，直接使用相对动作
                            right_pose_abs = np.concatenate([
                                initial_eef_pose[:3] + right_xyz,  # 右臂位置
                                right_quat,  # 右臂旋转
                                initial_eef_pose[7:8] + right_gripper  # 右爪
                            ])
                            left_pose_abs = np.concatenate([
                                initial_eef_pose[8:11] + left_xyz,  # 左臂位置
                                left_quat,  # 左臂旋转
                                initial_eef_pose[15:16] + left_gripper  # 左爪
                            ])
                            
                            ee_action_16 = np.concatenate([right_pose_abs, left_pose_abs])
                            
                        elif action_dim == 16:
                            # 16维：已经是正确格式，但需要转换为绝对位姿
                            # 格式：[right_xyz(3), right_quat(4), right_gripper(1),
                            #        left_xyz(3), left_quat(4), left_gripper(1)]
                            # 转换为绝对位姿
                            ee_action_16 = np.concatenate([
                                initial_eef_pose[:3] + ee_action[:3],  # 右臂位置
                                ee_action[3:7] / np.linalg.norm(ee_action[3:7]),  # 右臂旋转（归一化）
                                initial_eef_pose[7:8] + ee_action[7:8],  # 右爪
                                initial_eef_pose[8:11] + ee_action[8:11],  # 左臂位置
                                ee_action[11:15] / np.linalg.norm(ee_action[11:15]),  # 左臂旋转（归一化）
                                initial_eef_pose[15:16] + ee_action[15:16]  # 左爪
                            ])
                        else:
                            print(f"[!] 动作维度不正确 (应为14或16)，当前: {action_dim}")
                            continue
                        
                        if len(ee_action_16) != 16:
                            print(f"[!] 转换后动作维度不正确 (应为16)，当前: {len(ee_action_16)}")
                            continue

                        right_pose_array, left_pose_array = ee_action_16[0:8], ee_action_16[8:16]
                        gripper_states = [left_pose_array[7].item(), right_pose_array[7].item()]
                        right_pose_dict = { 
                            "x": right_pose_array[0].item(), 
                            "y": right_pose_array[1].item(), 
                            "z": right_pose_array[2].item(), 
                            "qx": right_pose_array[3].item(), 
                            "qy": right_pose_array[4].item(), 
                            "qz": right_pose_array[5].item(), 
                            "qw": right_pose_array[6].item() 
                        }
                        left_pose_dict = { 
                            "x": left_pose_array[0].item(), 
                            "y": left_pose_array[1].item(), 
                            "z": left_pose_array[2].item(), 
                            "qx": left_pose_array[3].item(), 
                            "qy": left_pose_array[4].item(), 
                            "qz": left_pose_array[5].item(), 
                            "qw": left_pose_array[6].item() 
                        }
                        
                        print(f"[→ Frame {i+1}/{num_frames}, Step {j+1}/{num_steps}] 执行动作...")
                        robot_controller.set_end_effector_pose_control(
                            lifetime=1.0,
                            control_group=["dual_arm"],
                            left_pose=left_pose_dict,
                            right_pose=right_pose_dict,
                        )
                        robot_dds.move_gripper(gripper_states)
                        
                        # 控制执行频率
                        if j % 4 == 0:
                            time.sleep(0.05)
                        
                        # 每 action_per_frame 个 step 获取一次观测（用于 KV cache）
                        if (j + 1) % action_per_frame == 0:
                            try:
                                # 获取当前图像（简化：只获取图像，不获取状态）
                                current_images_dict = {}
                                for server_name, sdk_name in CAMERA_MAPPING.items():
                                    current_img = get_image(camera, sdk_name)
                                    if current_img is not None:
                                        current_images_dict[server_name] = current_img
                                
                                # 格式化观测并添加到 key_frame_list
                                if len(current_images_dict) == len(CAMERA_MAPPING):
                                    current_formatted_obs = format_obs(current_images_dict)
                                    key_frame_list.append(current_formatted_obs)
                                    print(f"✅ 已收集 key frame (Frame {i+1}, Step {j+1})")
                                else:
                                    print(f"⚠️ 无法获取完整图像，跳过此 key frame")
                                    
                            except Exception as e:
                                print(f"⚠️ 获取观测失败，跳过此 key frame: {e}")
                                import traceback
                                traceback.print_exc()
                
                print("✅ 动作序列执行完毕。")
                
                # 发送 KV cache 请求（如果有收集到 key frames）
                if len(key_frame_list) > 0:
                    try:
                        print(f"🔄 发送 KV cache 请求，key_frame_list 长度: {len(key_frame_list)}")
                        model.infer(dict(
                            obs=key_frame_list,
                            compute_kv_cache=True,
                            state=action
                        ))
                        print("✅ KV cache 更新完成")
                    except Exception as e:
                        print(f"⚠️ KV cache 请求失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 第一次推理后，后续推理从第一个 frame 开始
                if first_inference:
                    first_inference = False
            else:
                print(f"[!] 当前不支持的控制模式: {CONTROL_MODE}")
                break

    except KeyboardInterrupt:
        print("\n[Main] 用户手动中断程序。")
    except Exception as e:
        print(f"\n[Main] ❌ 程序执行时发生严重错误: {e}")
    finally:
        # --- 3. 安全关闭 ---
        if robot_dds:
            print("\n[Main] 重置机器人到安全位置...")
            robot_dds.reset()
            time.sleep(2)
            robot_dds.shutdown()
        if camera:
            camera.close()
        print("[Main] 程序已安全退出。")

if __name__ == "__main__":
    main()

      
"""
POST /infer 输入样例：
{
  "qpos": [[0.1, 0.2, ..., 0.3]],  # shape: [B, state_dim]，可为一维或二维数组
  "eef_pose": [[0.1, 0.2, ..., 0.3]],  # shape: [B, action_dim]，可为一维或二维数组
  "instruction": "请让机器人前进并避开障碍物",
  "images": [
    {
      "base_0_rgb": "<base64字符串>",
      "left_wrist_0_rgb": "<base64字符串>"
    }
    # 可以有多个样本，每个样本是一个相机名到base64图片的字典
  ],
}
"""
import os
import sys
import torch
import h5py
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import numpy as np
import time
import traceback
import json
import cv2
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("/mnt/data/real_eval")
sys.path.append("/mnt/data/RoboTwin/eval_vla_robotwin")

from pose_transform import add_delta_to_quat_pose
from action_token.action_chunk_to_fast_token import ActionChunkProcessor

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Return a cached ActionChunkProcessor (one per process).
    每个 Ray worker 进程各自维护 _TOKENIZER_CACHE，首次调用时才实例化。
    """
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
    return tok


def norm_transform(x, q01, q99):
    q_range = q99 - q01
    if not isinstance(q_range, np.ndarray):
        q_range = np.array(q_range)

    q_range[q_range == 0.0] = 1.0

    x = 2 * ((x - q01) / q_range) - 1
    return scaled_x


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 启用CORS，允许跨域请求
CORS(app)

# 服务配置
SERVICE_CONFIG = {
    'host': '0.0.0.0',  # 监听所有网络接口
    'port': 5004,       # 服务端口
    'debug': False,     # 生产环境设为False
    'threaded': True,   # 启用多线程
    'max_content_length': 16 * 1024 * 1024  # 最大请求大小16MB
}

# 加载模型
# MODEL_PATH = '/share/project/jiyuheng/ckpt/robotics_pretrain_modeltp1pp1_S6_20'
# MODEL_PATH = '/share/project/jiyuheng/ckpt/robotics_pretrain_modeltp1pp1_S6_23'
# MODEL_PATH = '/mnt/data/ckpts/a2d_S10_model/robotics_pretrain_modeltp1pp1_S10_A2D_pour_coffee_10epoch'
# MODEL_PATH = '/mnt/data/ckpts/a2d_S8_demo/train_qwen2_5_vl_3b_action_S10_A2D_open_close_10epoch'
# MODEL_PATH = '/mnt/data/ckpts/a2d_S7_model/train_3b_action_S7_A2D_open_close_640_480'
# MODEL_PATH = '/mnt/data/ckpts/a2d_X0_model/X0_A2D/train_3b_action_S7_subtask_A2D_pour_coffee_640_480'
# MODEL_PATH = '/mnt/data/ckpts/a2d_X0_model/X0_A2D/train_3b_action_X0_A2D_pour_coffee_320_240'
# MODEL_PATH = '/mnt/data/ckpts/a2d_X0_model/X0_A2D/train_3b_action_S7_subtask_A2D_open_close_640_480'
# MODEL_PATH = '/mnt/data/ckpts/a2d_X0_model/X0_A2D/train_3b_action_X0_A2D_open_close_320_240'
MODEL_PATH = '/mnt/data/ckpts/robotics_pretrain_modeltp1pp1_3b_action_S7_Subtask_X0_Pro_A2D_20Skill_640_480_merged_5epoch'
# MODEL_PATH = '/mnt/data/ckpts/robotics_pretrain_modeltp1pp1_3b_action_S7_Subtask_X0_Pro_A2D_20Skill_640_480_merged_1epoch'
# /mnt/data/ckpts/checkpoint-8603


# MODEL_PATH = '/mnt/data/ckpts/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_1epoch/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_1epoch'
# MODEL_PATH = '/mnt/data/ckpts/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_3epoch/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_3epoch'
# MODEL_PATH = '/mnt/data/ckpts/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_5epoch/robotics_pretrain_modeltp1pp1_Alpha_600h_pour_coffee_5epoch'





CONFIG_PATH = MODEL_PATH
DEBUG=False
# 全局模型变量
model = None
action_tokenizer = get_tokenizer(max_len=256)

# 加载norm参数
# with open("/mnt/data/real_eval/norms/a2d_norm_open_close_212.json", 'r') as f:
# with open("/mnt/data/real_eval/norms/a2d_norm_pour_coffice.json", 'r') as f:
with open("/mnt/data/real_eval/norms/a2d_norm_skill_20.json", 'r') as f:
    action_quantiles_low = json.load(f) 

def load_model():
    """加载模型"""
    global model
    global processor
    try:
        logger.info("开始加载模型...")
        # config = RoboBrainRoboticsConfig.from_pretrained(CONFIG_PATH)
        # config.training = False
        devide_id = os.environ.get("EGL_DEVICE_ID", "0")
        device = f"cuda:{devide_id}" if torch.cuda.is_available() else "cpu"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            padding_side='left'
        )
        model.eval()
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"模型已加载到GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("模型已加载到CPU")
            
        logger.info("模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False

def transform(x, scale, offset, clip=True):
    x_norm = x * scale + offset
    if clip:
        np.clip(x_norm, -1, 1, out=x_norm)  
    return x_norm

def inverse_transform(x_norm, scale, offset):
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale


import base64
from PIL import Image
import io


def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    """
    将欧拉角(roll, pitch, yaw)转换为四元数
    
    参数:
        roll: 绕X轴旋转角度 (翻滚角)
        pitch: 绕Y轴旋转角度 (俯仰角)
        yaw: 绕Z轴旋转角度 (偏航角)
        degrees: 输入是否为角度 (默认True)
    
    返回:
        四元数 [x, y, z, w]
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_quat()

def decode_image_base64(image_base64):
    """解码base64图片"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('BGR')
        image = np.array(image).astype(np.float32) / 255.0  # 归一化到0-1
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
        return image
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError(f"图片解码失败: {e}")

def decode_image_base64_to_pil(image_base64):
    """解码base64图片"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_cv = np.array(image)
        image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
        return image
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError(f"图片解码失败: {e}")

def process_images(images_dict):
    """处理图片列表"""
    processed = []
    processed_list = []

    try:
        sample_dict = {}
        # for k in ['cam_high_realsense', 'cam_right_wrist', 'cam_left_wrist']:
        for k in ['cam_head', 'cam_right_wrist', 'cam_left_wrist']:
            sample_dict[k] = decode_image_base64_to_pil(images_dict[k])
            processed_list.append(sample_dict[k].resize((640,480)))
            # processed_list.append(sample_dict[k])
        processed.append(sample_dict)
    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        raise ValueError(f"处理图片失败: {e}")

    return processed_list

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 检查模型是否已加载
        if model is None:
            return jsonify({
                "status": "error",
                "message": "模型未加载",
                "timestamp": time.time()
            }), 503
        
        # 检查GPU状态
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            }
        else:
            gpu_info = {"available": False}
        
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "gpu_info": gpu_info,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/info', methods=['GET'])
def service_info():
    """服务信息端点"""
    return jsonify({
        "service_name": "RoboBrain Robotics API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info", 
            "infer": "/infer"
        },
        "model_info": {
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "action_dim": getattr(model.config, 'action_dim', 'unknown') if model else 'unknown',
            "action_horizon": getattr(model.config, 'action_horizon', 'unknown') if model else 'unknown'
        },
        "timestamp": time.time()
    })


@app.route('/infer', methods=['POST'])
def infer_api():
    """推理API端点"""
    start_time = time.time()
    
    try:
        # 检查模型是否已加载
        if model is None:
            print("400: 模型未加载，请检查服务状态")
            return jsonify({
                "success": False,
                "error": "模型未加载，请检查服务状态"
            }), 503
        
        # 解析请求数据
        data = request.get_json()
        if not data:
            print("400: 请求数据为空或格式错误!")
            return jsonify({
                "success": False,
                "error": "请求数据为空或格式错误"
            }), 400

        if 'eef_pose' not in data:
            print("400: 缺少必需字段: eef_pose!")
            return jsonify({
                "success": False,
                "error": "缺少必需字段: eef_pose"
            }), 400

        instruction = data.get('instruction')[0]
        images = data.get('images')
        eef_pose = np.array(data['eef_pose'])

        # 验证指令参数
        if instruction is None:
            print("400: 必须提供 instruction")
            return jsonify({
                "success": False,
                "error": "必须提供 instruction"
            }), 400

        # 处理图片数据
        images_tensor = None
        if images is not None:
            try:
                images_tensor = process_images(images)
            except Exception as e:
                print(f"400: 错误：{e}")
                return jsonify({
                    "success": False,
                    "error": f"图片处理失败: {e}"
                }), 400


        # 执行推理
        logger.info(f"开始推理，状态维度: state_tensor.shape, 指令: {instruction}, 图片数量: {len(images_tensor) if images_tensor else 0}")
        with torch.no_grad():
            ####
            prefix_prompt = [
                "You are controlling an A2D dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. ",
                "You need to output control tokens that can be decoded into a 30×14 action sequence. ",
                "The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEF, and the last 7 dimensions control the left arm EEF. ",
                "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
                "Your current visual inputs are: "
            ]

            content = [
                {"type": "text", "text": "".join(prefix_prompt)}, 
                {"type": "text", "text": "robot front image"}, 
                {"type": "image", "image": f"data:image;base64,{images['cam_head']}"},
                {"type": "text", "text": ", right wrist image"},
                {"type": "image", "image": f"data:image;base64,{images['cam_right_wrist']}"},
                {"type": "text", "text": " and left wrist image"},
                {"type": "image", "image": f"data:image;base64,{images['cam_left_wrist']}"},
                # {"type": "text", "text": f"\nYour overall task is: {instruction.lower()}.  Currently, focus on completing the subtask: {instruction.lower()}."}, #
                {"type": "text", "text": f"\nYour overall task is: {instruction.lower()}."}, 
            ]
            keys = ['cam_head', 'cam_right_wrist', 'cam_left_wrist']
            log_time = time.time()
            for cam_key, cam_image in zip(keys, images_tensor):
                cam_image.save(f'./image_log/{cam_key}.png')

            messages = [{"role": "user", "content": content}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)   
            inputs = processor(text=[text_prompt], images=images_tensor, padding=True, return_tensors="pt").to(model.device)   

            gen_kwargs = {
                "max_new_tokens": 768,
                "do_sample": False,
                "temperature": 0.0,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "repetition_penalty": 1.0,  # 不惩罚重复，因为action可能重复
                "use_cache": True,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)

            generated_tokens = output.sequences
            input_length = inputs.input_ids.shape[1]
            output_ids = generated_tokens[:, input_length:]
            output_ids = output_ids[0].detach().cpu().tolist()

            try:
                end_index = output_ids.index(151667)  # for 2 action tokens
                output_ids = output_ids[:end_index]
            except ValueError:
                logger.warning("未找到结束token 151667，使用完整输出")

            action_ids = [int(_-149595) for _ in output_ids if _ >= 149595 and _ < 151643]
            print(f"action_ids: {action_ids}")
            actions, output_dims = action_tokenizer._extract_actions_from_tokens([action_ids], action_horizon=30, action_dim=14)
            delta_actions = actions[0]

        # 处理动作数据
        if delta_actions is not None:
            try:
                scale = np.array(action_quantiles_low['action.eepose']['scale_'])
                offset = np.array(action_quantiles_low['action.eepose']['offset_'])
                delta_actions_denorm = inverse_transform(np.array(delta_actions), scale, offset)
 
                final_ee_actions_pos = []
                final_qpos = []
                current_eef_pose = eef_pose
                for i in range(30):
                    if DEBUG:  ipdb.set_trace()
                    current_eef_pose[:3] += delta_actions_denorm[i][:3]
                    current_eef_pose[3:7] = add_delta_to_quat_pose(current_eef_pose[3:7], delta_actions_denorm[i][3:6])
                    current_eef_pose[8:11] += delta_actions_denorm[i][7:10]
                    current_eef_pose[11:15] = add_delta_to_quat_pose(current_eef_pose[11:15], delta_actions_denorm[i][10:13])
                    # gripper 原本为0关1开，这里需要做一次变换
                    current_eef_pose[7] = np.clip(1-delta_actions_denorm[i][6], 0, 1)
                    current_eef_pose[-1] = np.clip(1-delta_actions_denorm[i][-1], 0, 1)
                    # 转四元数
                    current_eef_quaternion_pose = current_eef_pose
                    final_ee_actions_pos.append(current_eef_quaternion_pose.tolist())
     
            except Exception as e:
                logger.error(f"动作数据反归一化失败: {e}")
                # 不返回错误，保持原始数据

        # 添加处理时间信息
        processing_time = time.time() - start_time
        if DEBUG:  ipdb.set_trace()
        #with open(f'/share/project/dumengfei/code/real_eval/action_log/action.json', 'w') as f:
        #    json.dump(current_eef_pose.tolist(), f)
        logger.info(f"推理完成，耗时: {processing_time:.2f}秒")
        print(f"Final eef_action_pos shape: ({len(final_ee_actions_pos)} x {len(final_ee_actions_pos[0])})")
        print(f"Final delta eef_action_pos[0]: {delta_actions_denorm[0]}")
        print(f"Final delta eef_action_pos[1]: {delta_actions_denorm[1]}")
        print(f"Final eef_action_pos[0]: {final_ee_actions_pos[0]}")
        print(f"Final eef_action_pos[-1]: {final_ee_actions_pos[-1]}")

        if DEBUG: ipdb.set_trace()
        return jsonify({
            "success": True, 
            "eepose": final_ee_actions_pos,   # default: []
            "qpos": final_qpos,
            "processing_time": processing_time
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "success": False,
        "error": "接口不存在",
        "available_endpoints": ["/health", "/info", "/infer"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500

if __name__ == '__main__':
    # 加载模型
    if not load_model():
        logger.error("模型加载失败，服务无法启动")
        sys.exit(1)
    
    # 打印服务信息
    logger.info(f"RoboBrain Robotics API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"可用端点:")
    logger.info(f"  - GET  /health  - 健康检查")
    logger.info(f"  - GET  /info    - 服务信息")
    logger.info(f"  - POST /infer   - 推理接口")
    
    # 启动服务
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    ) 

    

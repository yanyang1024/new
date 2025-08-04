import asyncio
import threading
import numpy as np
import pyaudio
import webrtcvad
import queue
import time
from datetime import datetime
import torch
import soundfile as sf
import os
from collections import deque
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

from ..config.audio_config import *
from ..config.translation_config import *
from ..utils.audio_utils import *
from ..utils.translation_utils import *

class AudioTranslator:
    def __init__(self, test_mode=True):
        try:
            import transformers
            if transformers.__version__ < "4.37.2":
                print("Warning: Please upgrade transformers to >= 4.37.2")
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        # 初始化音频处理组件
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_MODE)
        
        # 初始化调试参数
        self.test_mode = test_mode
        self.debug_info = {
            'vad_detections': 0,
            'speech_segments': 0,
            'recognition_attempts': 0,
            'recognition_success': 0,
            'translation_attempts': 0
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_real_translator()

        # 初始化缓冲区和状态
        self.audio_buffer = deque(maxlen=50)  # 约1.5秒的音频
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = []
        self.translation_queue = queue.Queue()
        self.current_translation = ""
        
        # 新增的属性
        self.previous_translations = []  # 存储最近的翻译结果
        self.partial_translation = ""    # 存储部分翻译结果
        self.last_speech_time = time.time()  # 上次检测到语音的时间
        self.energy_threshold = ENERGY_THRESHOLD     # 语音能量阈值，可以动态调整
        
        self.vad_window = deque(maxlen=5)  # VAD结果的滑动窗口
        self.last_chinese_text = ""      # 上一次识别的中文文本
        self.consecutive_silence = 0      # 连续静音帧计数
        
        self.latest_partial = ""  # 新增：最新部分翻译
        self.last_update_time = 0  # 新增：最后更新时间
        self.translation_history = []  # 新增：翻译历史记录

        self.running = False

        # 启动处理线程
        self.running = True
        self.translation_thread = threading.Thread(target=self._translation_worker)
        self.translation_thread.start()

        # 在AudioTranslator类中添加新的属性
        self.pending_translation = ""  # 待处理的翻译文本
        self.sentence_buffer = []      # 句子缓冲区
        self.last_sentence_end = time.time()  # 上一句结束时间
        self.punctuation_threshold = PUNCTUATION_THRESHOLD  # 句子间隔阈值（秒）

        # 新增属性: 输入模式，默认为None，表示未设置
        self.input_mode = None  # 'file' or 'mic'
        self.latest_complete_translation = ""  # 新增:存储最新的完整翻译
        self.all_translations = []  # 新增:存储所有翻译结果
        self.file_path = None  # 新增:存储动态文件路径

        # 音频流缓存优化
        self.audio_cache = deque(maxlen=MAX_CACHE_SIZE)  # 音频缓存，最多保存200帧
        self.cache_lock = threading.Lock()  # 缓存锁
        self.max_cache_size = MAX_CACHE_SIZE  # 最大缓存大小
        self.cache_cleanup_interval = CACHE_CLEANUP_INTERVAL  # 每1000帧清理一次缓存
        self.frame_counter = 0  # 帧计数器

        self.websocket_server = None
        self.udp_server = None
        self.input_source = None  # 'file', 'mic', or 'websocket'
        
        # 新增：Gradio Audio流式处理相关属性
        self.gradio_audio_stream = None  # Gradio音频流状态
        self.gradio_audio_buffer = deque(maxlen=100)  # Gradio音频缓冲区
        self.gradio_processing = False  # Gradio处理状态标志
        self.gradio_audio_lock = threading.Lock()  # Gradio音频处理锁

    def _setup_real_translator(self):
        """设置实际的翻译器，优化GPU性能"""
        # 设置GPU性能优化
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"[INFO] 使用设备: {self.device}")
        print(f"[INFO] 当前CUDA设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        
        # 加载模型和处理器
        print("[INFO] 开始加载模型...")
        self.processor = AutoProcessor.from_pretrained(PATH_FOR_SEAMLESS_M4T, device_map="auto")
        
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            PATH_FOR_SEAMLESS_M4T,
            device_map="auto",
            torch_dtype=torch.float16
        ).to(self.device).eval()
        print("[INFO] 模型加载完成")

    def _is_speech(self, audio_chunk):
        """检测是否为语音"""
        try:
            # 统一将数据转换为float32格式
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            else:
                audio_data = audio_chunk
            
            # 检查音频数据是否有效
            if audio_data is None or len(audio_data) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空，跳过VAD检测")
                return False
            
            # 确保数据类型为float32
            audio_data = audio_data.astype(np.float32)
            
            # 处理异常数据范围 - 改进的NaN值处理
            max_abs = np.max(np.abs(audio_data))
            if max_abs > MAX_AMPLITUDE:  # 如果数据范围过大，进行软裁剪
                if self.test_mode:
                    self._append_debug(f"音频数据范围异常: {max_abs:.2f}，进行软裁剪")
                # 软裁剪到合理范围
                audio_data = np.clip(audio_data, -MAX_AMPLITUDE, MAX_AMPLITUDE)
                # 重新归一化
                max_abs = np.max(np.abs(audio_data))
                if max_abs > 0:
                    audio_data = audio_data / max_abs
            
            # 检查数据是否包含无效值
            if np.isnan(audio_data).any():
                if self.test_mode:
                    self._append_debug(f"音频数据包含NaN值，尝试修复。数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
                # 尝试修复NaN值
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug(f"音频数据包含Inf值，尝试修复。数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
                # 尝试修复Inf值
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 如果是不完整的帧，用0填充到标准长度
            if len(audio_data) < CHUNK:
                pad_length = CHUNK - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_length), 'constant', constant_values=0)
            elif len(audio_data) > CHUNK:
                audio_data = audio_data[:CHUNK]
            
            # 最终检查处理后的数据
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug("音频数据处理后仍包含无效值，跳过VAD检测")
                return False
            
            # 改进的音频能量计算 - 防止overflow
            energy = calculate_audio_energy(audio_data, self.test_mode)
            
            # 动态调整能量阈值
            if energy > self.energy_threshold * 2:
                self.energy_threshold = min(self.energy_threshold * 1.1, 0.02)
            elif energy < self.energy_threshold / 2:
                self.energy_threshold = max(self.energy_threshold * 0.9, 0.005)
            
            # 准备VAD检测的数据
            # 确保音频数据在合理范围内
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 最终检查
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                if self.test_mode:
                    self._append_debug("音频数据裁剪后仍包含无效值，跳过VAD检测")
                return False
            
            vad_data = (audio_data * NORMALIZATION_FACTOR).astype(np.int16)
            
            # 检查转换是否成功
            if np.isnan(vad_data).any() or np.isinf(vad_data).any():
                if self.test_mode:
                    self._append_debug("VAD数据转换失败，包含无效值")
                return False
            
            frame_bytes = vad_data.tobytes()
            
            if len(frame_bytes) % 2 != 0:
                if self.test_mode:
                    self._append_debug("跳过VAD检测：帧字节数不是2的倍数")
                return False
            
            # VAD检测
            vad_result = self.vad.is_speech(frame_bytes, RATE)
            self.vad_window.append(vad_result)
            
            # 使用滑动窗口平均值进行平滑
            vad_ratio = sum(self.vad_window) / len(self.vad_window) if self.vad_window else 0
            
            # 综合判断是否为语音
            is_speech = (energy > self.energy_threshold and vad_ratio > 0.4)
            
            if self.test_mode:
                self.debug_info['vad_detections'] += 1
                self._append_debug(f"VAD检测: {'语音' if is_speech else '静音'} "
                                f"(能量={energy:.4f}, 阈值={self.energy_threshold:.4f}, "
                                f"VAD比例={vad_ratio:.2f})")
            return is_speech
            
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"VAD检测异常: {str(e)}")
            return False

    def _append_debug(self, msg):
        """添加调试信息"""
        if not hasattr(self, 'debug_msgs'):
            self.debug_msgs = []
        self.debug_msgs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # 限制debug信息长度
        if len(self.debug_msgs) > MAX_DEBUG_MESSAGES:
            self.debug_msgs = self.debug_msgs[-MAX_DEBUG_MESSAGES:]

    def get_debug_info(self):
        """获取调试信息"""
        return '\n'.join(getattr(self, 'debug_msgs', []))

    def get_all_translations(self):
        """获取所有翻译记录的汇总"""
        return "\n".join(self.all_translations)

    def set_input_mode(self, mode, file_path=None):
        """设置输入模式"""
        self.input_mode = mode
        if file_path:
            self.file_path = file_path
        if self.test_mode:
            self._append_debug(f"设置输入模式: {mode}" + (f", 文件: {file_path}" if file_path else ""))
        
        # 如果是Gradio麦克风模式，重置相关状态
        if mode == 'gradio_mic':
            self.reset_gradio_state()

    def close(self):
        """改进的关闭方法"""
        print("[INFO] 开始关闭AudioTranslator...")
        self.running = False
        
        try:
            # 停止翻译线程
            if hasattr(self, 'translation_thread') and self.translation_thread.is_alive():
                print("[INFO] 停止翻译线程...")
                self.translation_queue.put(("", False))  # 发送空消息触发退出
                self.translation_thread.join(timeout=5)
                if self.translation_thread.is_alive():
                    print("[WARNING] 翻译线程未能在5秒内停止")
                
            # 关闭音频设备
            if hasattr(self, 'audio'):
                print("[INFO] 关闭音频设备...")
                self.audio.terminate()
                
            # 清理状态
            self.speech_frames = []
            self.audio_buffer.clear()
            self.translation_queue = queue.Queue()
            self.current_translation = ""
            self.partial_translation = ""
            self.latest_complete_translation = ""
            self.all_translations = []
            self.translation_history = []
                
            if self.test_mode:
                print("\nDebug Information:")
                for key, value in self.debug_info.items():
                    print(f"{key}: {value}")
                
        except Exception as e:
            print(f"[ERROR] 关闭异常: {e}")
        
        print("[INFO] AudioTranslator关闭完成") 
import asyncio
import threading
import numpy as np
import gradio as gr
import pyaudio
import wave
import webrtcvad
import queue
import time
from datetime import datetime
import speech_recognition as sr
from transformers import AutoProcessor, SeamlessM4Tv2Model
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from pydub import AudioSegment
from collections import deque
import torch
import soundfile as sf
import os
from mock_translator import MockProcessor, MockSeamlessModel
import json

# 音频配置
RATE = 16000
CHANNELS = 1
CHUNK = 320
FORMAT = pyaudio.paFloat32
VAD_MODE = 2
PARTIAL_UPDATE_FRAMES = 40  # 增加到800ms进行一次部分翻译
SILENCE_THRESHOLD_SHORT = 25  # 增加到500ms静音
SILENCE_THRESHOLD_LONG = 35  # 增加到700ms静音
MIN_SPEECH_FRAMES = 20  # 至少需要400ms的语音

path_for_seamless_m4t = "/data/checkpoints/seamless-m4t-v2-large"



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
        # self.recognizer = sr.Recognizer()
        
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

        test_mode = True
        self.device = "cuda"
        
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
        self.energy_threshold = 0.01     # 语音能量阈值，可以动态调整
        
        
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
        self.punctuation_threshold = 2.0  # 句子间隔阈值（秒）

        # 新增属性: 输入模式，默认为None，表示未设置
        self.input_mode = None  # 'file' or 'mic'
        self.latest_complete_translation = ""  # 新增:存储最新的完整翻译
        self.all_translations = []  # 新增:存储所有翻译结果
        self.file_path = None  # 新增:存储动态文件路径

        # 音频流缓存优化
        self.audio_cache = deque(maxlen=200)  # 音频缓存，最多保存200帧
        self.cache_lock = threading.Lock()  # 缓存锁
        self.max_cache_size = 200  # 最大缓存大小
        self.cache_cleanup_interval = 1000  # 每1000帧清理一次缓存
        self.frame_counter = 0  # 帧计数器

        self.websocket_server = None
        self.udp_server = None
        self.input_source = None  # 'file', 'mic', or 'websocket'
        
        # 新增：Gradio Audio流式处理相关属性
        self.gradio_audio_stream = None  # Gradio音频流状态
        self.gradio_audio_buffer = deque(maxlen=100)  # Gradio音频缓冲区
        self.gradio_processing = False  # Gradio处理状态标志
        self.gradio_audio_lock = threading.Lock()  # Gradio音频处理锁
        
    def _setup_mock_translator(self):
        """设置模拟翻译器，直接使用mock_translator.py中的MockProcessor和MockSeamlessModel"""
        self.processor = MockProcessor()
        self.model = MockSeamlessModel()

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
        self.processor = AutoProcessor.from_pretrained(path_for_seamless_m4t, device_map="auto")
        
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            path_for_seamless_m4t,
            device_map="auto",
            torch_dtype=torch.float16
        ).to(self.device).eval()
        print("[INFO] 模型加载完成")

    def _setup_websocket_server(self, host="0.0.0.0", port=8765):
        """设置 WebSocket 服务器"""
        from websocket_server import AudioWebSocketServer
        import asyncio
        
        self.websocket_server = AudioWebSocketServer(
            host=host,
            port=port,
            audio_callback=self._process_audio_chunk
        )
        
        # 在新线程中启动 WebSocket 服务器
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.websocket_server.start())
            loop.run_forever()
            
        self.websocket_thread = threading.Thread(target=run_server, daemon=True)
        self.websocket_thread.start()
        
    def _setup_udp_server(self, host="0.0.0.0", port=12345):
        """设置 UDP 服务器"""
        from udp_server import AudioUDPServer
        
        self.udp_server = AudioUDPServer(
            host=host,
            port=port,
            audio_callback=self._process_audio_chunk
        )
        self.udp_server.start()
        
    def _get_audio_input_stream(self):
        """根据输入模式返回相应的音频流"""
        if self.input_mode == 'file':
            return self._get_file_stream()
        elif self.input_mode == 'mic':
            # 麦克风模式 - 使用 UDP
            self._setup_udp_server()
            return None  # UDP 模式下不需要返回流
        elif self.input_mode == 'gradio_mic':
            # Gradio麦克风模式 - 使用Gradio Audio流式处理
            return None  # Gradio模式下不需要返回流，直接处理
        else:
            raise ValueError(f"不支持的输入模式: {self.input_mode}")
    
    def _get_file_stream(self):
        """获取文件音频流"""
        self.test_mode = True
        if self.test_mode:
            import wave
            import threading
            import numpy as np
            from pydub import AudioSegment

            class FileStream:
                def __init__(self, wav_path, chunk):
                    try:
                        # 使用pydub加载音频文件并转换为16位PCM编码
                        audio = AudioSegment.from_file(wav_path)
                        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                        audio_raw_data = audio.raw_data
                        
                        self.framerate = audio.frame_rate
                        self.nchannels = audio.channels
                        self.sampwidth = audio.sample_width
                        
                        # 转换为numpy数组
                        audio_array = np.frombuffer(audio_raw_data, dtype=np.int16)
                        
                        # 转换为float32并归一化
                        self.audio_data = audio_array.astype(np.float32) / 32768.0
                        
                        self.chunk = chunk
                        self.position = 0
                        self.lock = threading.Lock()
                        print(f"[DEBUG] 成功加载音频文件: {wav_path}")
                        print(f"[DEBUG] 音频长度: {len(self.audio_data)} 采样点")
                        print(f"[DEBUG] 采样率: {self.framerate} Hz")
                        print(f"[DEBUG] 声道数: {self.nchannels}")
                    except Exception as e:
                        print(f"[ERROR] 处理音频文件 {wav_path} 失败: {e}")
                        raise

                def read(self, chunk, exception_on_overflow=False):
                    with self.lock:
                        if self.position >= len(self.audio_data):
                            # 测试模式下，读到文件末尾就停止，不重复播放
                            return b''
                        
                        end_pos = min(self.position + chunk, len(self.audio_data))
                        data = self.audio_data[self.position:end_pos]
                        self.position += chunk
                        
                        # 转换为bytes
                        return data.tobytes()

                def stop_stream(self): 
                    self.position = len(self.audio_data)  # 强制结束读取

                def close(self):
                    # self.running = False
                    self.position = len(self.audio_data)
                    print("[DEBUG] 关闭音频文件流")

            return FileStream(self.file_path, CHUNK)
        else:
            try:
                return self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            except Exception as e:
                print(f"[ERROR] 打开麦克风音频流失败: {e}")
                raise


    def _is_speech(self, audio_chunk):
        # 主要用VAD检查语音人声存在
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
            if max_abs > 1000.0:  # 如果数据范围过大，进行软裁剪
                if self.test_mode:
                    self._append_debug(f"音频数据范围异常: {max_abs:.2f}，进行软裁剪")
                # 软裁剪到合理范围
                audio_data = np.clip(audio_data, -1000.0, 1000.0)
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
            try:
                # 使用更安全的能量计算方法
                abs_audio = np.abs(audio_data)
                # 检查是否有异常大的值
                if np.max(abs_audio) > 1e6:  # 如果最大值超过1e6，说明有异常
                    if self.test_mode:
                        self._append_debug(f"音频数据异常大: {np.max(abs_audio):.2e}，进行裁剪")
                    abs_audio = np.clip(abs_audio, 0, 1e6)
                
                # 使用更稳定的平均值计算
                energy = np.mean(abs_audio)
                
                # 检查能量值是否合理
                if np.isnan(energy) or np.isinf(energy):
                    if self.test_mode:
                        self._append_debug(f"能量计算异常: {energy}，使用默认值")
                    energy = 0.0
                
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"能量计算异常: {str(e)}，使用默认值")
                energy = 0.0
            
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
            
            vad_data = (audio_data * 32768.0).astype(np.int16)
            
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



    def _process_audio_chunk(self, audio_chunk):
        """同步处理音频块的完整逻辑"""
        # VAD检测 First 翻译处理 Next

        try:
            if len(audio_chunk) == 0:
                if self.test_mode:
                    self._append_debug("收到空音频帧，跳过处理")
                return
            
            # 添加调试信息，特别是在麦克风模式下
            if self.test_mode and (self.input_mode == 'mic' or self.input_mode == 'gradio_mic'):
                self._append_debug(f"处理音频块: {len(audio_chunk)} 字节, 类型: {type(audio_chunk)}, 模式: {self.input_mode}")
            
            # 管理音频缓存
            self._manage_audio_cache(audio_chunk)
            self.audio_buffer.append(audio_chunk)
            
            # VAD检测
            is_speech = self._is_speech(audio_chunk)
            current_time = time.time()
            
            if self.test_mode:
                self._append_debug(f"VAD检测结果: {'语音' if is_speech else '静音'}")
            
            if is_speech:
                self.consecutive_silence = 0
                self.speech_frames.append(audio_chunk)
                self.last_speech_time = current_time
                
                if not self.is_speaking:
                    self.is_speaking = True # 设置标志位，表示正在说话 （也会传递到下一处理状态）
                    if self.test_mode:
                        self._append_debug("检测到语音 - 开始录音")
                
                # 实时处理：更频繁地进行部分处理以减少延迟
                if len(self.speech_frames) >= PARTIAL_UPDATE_FRAMES:  # 400ms 就进行一次处理
                    if self.test_mode:
                        self._append_debug(f"达到部分处理阈值: {len(self.speech_frames)} 帧")
                    # 保留更长的上下文
                    audio_data = b"".join(self.speech_frames)
                    if self.test_mode:
                        self._append_debug(f"开始处理语音片段: {len(audio_data)} 字节")
                    self.process_speech_segment(audio_data, partial=True) # partial=True 表明 进行部分处理

                    self.speech_frames = self.speech_frames[-10:]  # 保留200ms上文
            else:
                # 空音频/非人声处理
                self.consecutive_silence += 1
                
                if self.is_speaking:
                    silence_threshold = SILENCE_THRESHOLD_LONG if len(self.speech_frames) > 80 else SILENCE_THRESHOLD_SHORT
                    
                    if self.consecutive_silence >= silence_threshold:
                        if len(self.speech_frames) > MIN_SPEECH_FRAMES:
                            if self.test_mode:
                                self._append_debug(f"静音达到阈值，处理语音片段: {len(self.speech_frames)} 帧")
                            audio_data = b"".join(self.speech_frames)
                            if self.test_mode:
                                self._append_debug(f"开始处理完整语音片段: {len(audio_data)} 字节")
                            self.process_speech_segment(audio_data, partial=False)
                            
                        self.speech_frames = []
                        self.is_speaking = False
                        if self.test_mode:
                            self._append_debug("停止录音")
                
                # 长时间静音，清理状态
                if current_time - self.last_speech_time > 2.0:  # 2 秒无语音
                    # 无语音先处理之前的
                    if len(self.speech_frames) > MIN_SPEECH_FRAMES:
                        if self.test_mode:
                            self._append_debug(f"长时间静音，处理剩余语音片段: {len(self.speech_frames)} 帧")
                        audio_data = b"".join(self.speech_frames)
                        if self.test_mode:
                            self._append_debug(f"开始处理剩余语音片段: {len(audio_data)} 字节")
                        self.process_speech_segment(audio_data, partial=False)
                        self.speech_frames = []
                

                    # 再清空翻译流
                    self.speech_frames = []
                    self.is_speaking = False
                    if self.test_mode:
                        self._append_debug("长时间静音，清理状态")
                    
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"音频帧处理异常: {str(e)}")

   

    def process_speech_segment(self, audio_data, partial=False):
        try:
            if self.test_mode:
                self.debug_info['recognition_attempts'] += 1
                self._append_debug("开始处理语音片段" + (" (部分识别)" if partial else ""))

            # 将原始字节数据转换为 numpy 数组
            if isinstance(audio_data, bytes):
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                if self.test_mode:
                    self._append_debug(f"音频数据转换: bytes -> numpy, 长度={len(audio_np)}")
            else:
                audio_np = audio_data
                if self.test_mode:
                    self._append_debug(f"音频数据已经是numpy格式, 长度={len(audio_np)}")

            # 检查音频数据是否有效
            if len(audio_np) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空，跳过处理")
                return

            # 标准化音频数据（确保在-1 到1 之间）
            max_abs = np.max(np.abs(audio_np))
            if max_abs > 0:
                audio_np = audio_np / max_abs
                if self.test_mode:
                    self._append_debug(f"音频数据标准化: 最大值={max_abs:.4f} -> 1.0")
            else:
                if self.test_mode:
                    self._append_debug("音频数据为静音，跳过标准化")

            # 检查标准化后的数据
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                if self.test_mode:
                    self._append_debug("标准化后音频数据包含无效值，跳过处理")
                return

            try:
                if self.test_mode:
                    self._append_debug("开始SeamlessM4T处理...")
                
                # 使用 SeamlessM4T 进行端到端处理（语音->英文文本）
                audio_inputs = self.processor(
                    audios=audio_np,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=16000
                ).to(self.device) 

                if self.test_mode:
                    self._append_debug("音频预处理完成，开始生成翻译...")

                # 生成翻译（直接输出英文）
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    output_tokens = self.model.generate(
                        **audio_inputs,
                        tgt_lang="eng",
                        max_new_tokens=200,  # 使用 max_new_tokens 替代 max_length
                        num_beams=3,
                        early_stopping=True
                    )

                if self.test_mode:
                    self._append_debug("翻译生成完成，开始解码...")

                # 解码结果
                translated_text = self.processor.decode(
                    output_tokens[0].cpu().numpy().squeeze(), 
                    skip_special_tokens=True
                )

                if self.test_mode:
                    self.debug_info['recognition_success'] += 1
                    self._append_debug(f"识别翻译结果: {translated_text}")

                # 检查翻译结果是否有效
                if not translated_text or translated_text.strip() == "":
                    if self.test_mode:
                        self._append_debug("翻译结果为空，跳过处理")
                    return

                # 处理部分结果，翻译结果入队 结果包含翻译文本和是否为完整结果的flag,True表示为部分结果，False表示为完整结果
                if partial: 
                    # 部分结果需要进行部分处理 - 合并与更新翻译文本
                    # 将self.last_chinese_text 与 translated_text 进行比较，如果 translated_text 以 self.last_chinese_text 开头，则删除 self.last_chinese_text 的部分内容，并更新 self.last_chinese_text。
                    if self.last_chinese_text and translated_text.startswith(self.last_chinese_text):
                        translated_text = translated_text[len(self.last_chinese_text):].strip()
                        if self.test_mode:
                            self._append_debug(f"部分翻译去重: {translated_text}")
                    self.last_chinese_text = translated_text
                    self.translation_queue.put((translated_text, True))
                    if self.test_mode:
                        self._append_debug("部分翻译结果已入队")
                else:
                    self.translation_queue.put((translated_text, False))
                    if self.test_mode:
                        self._append_debug("完整翻译结果已入队")



            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"SeamlessM4T 处理异常: {str(e)}")
                translated_text = f"Translation Error: {str(e)}"

            # 处理临时文件（仅测试模式保留）
            if self.test_mode:
                wav_path = f"test_segments/segment_{int(time.time())}.wav"
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                sf.write(wav_path, audio_np, RATE)
                if self.test_mode:
                    self._append_debug(f"保存测试音频片段: {wav_path}")

                translated_text_path = f"test_segments_translated/segment_{int(time.time())}.txt"
                os.makedirs(os.path.dirname(translated_text_path), exist_ok=True)
                with open(translated_text_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)

        except Exception as e:
            if self.test_mode:
                self._append_debug(f"语音分段处理异常: {e}")
            translated_text = f"System Error: {str(e)}"

    def _format_translation(self, text, is_partial=False):
        """格式化翻译文本，添加标点符号"""
        # 移除已有的结束标点
        text = text.rstrip('.!?')
        
        # 对部分翻译结果的处理
        if is_partial:
            # 如果是不完整的句子，添加省略号
            return text + "..."
            
        # 对完整翻译结果的处理
        words = text.split()
        if len(words) < 3:  # 短句
            return text + "."
            
        # 根据关键词添加适当的标点
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        if words[0].lower() in question_words:
            return text + "?"
        elif any(word.lower() in {'can', 'could', 'would', 'will', 'shall'} for word in words[:2]):
            return text + "?"
        else:
            return text + "."

    def _should_merge_translations(self, new_text, prev_text):
        """增强的翻译合并判断逻辑"""
        if not prev_text or not new_text:
            return False
            
        # 检查时间间隔
        current_time = time.time()
        if current_time - self.last_sentence_end > self.punctuation_threshold:
            return False
            
        # 去除标点后再判断
        prev_clean = prev_text.rstrip('.!?').strip()
        new_clean = new_text.strip()
        
        # 检查句子完整性
        if len(prev_clean.split()) < 3:  # 前一句过短，可能不完整
            return True
            
        # 计算词重叠
        prev_words = set(prev_clean.lower().split())
        new_words = set(new_clean.lower().split())
        overlap = len(prev_words & new_words)
        
        # 主语连贯性检查
        prev_first = prev_clean.split()[0].lower()
        new_first = new_clean.split()[0].lower()
        subject_continuous = prev_first in {'i', 'you', 'he', 'she', 'it', 'we', 'they'} and \
                           new_first in {'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        return overlap >= 1 or subject_continuous

    def _save_translation_log(self, text, is_partial=False):
        """增强的日志保存功能"""
        try:
            log_dir = os.path.join(os.getcwd(), "translation_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 日志文件路径
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(log_dir, f"translation_{date_str}.log")
            stats_file = os.path.join(log_dir, f"stats_{date_str}.json")
            
            # 保存翻译日志
            with open(log_file, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {'(Partial) ' if is_partial else ''}{text}\n")
            
            # 更新统计信息
            if not is_partial:
                try:
                    stats = {}
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                    
                    stats['total_translations'] = stats.get('total_translations', 0) + 1
                    stats['avg_length'] = (stats.get('avg_length', 0) * (stats.get('total_translations', 1) - 1) + 
                                         len(text.split())) / stats['total_translations']
                    
                    with open(stats_file, 'w') as f:
                        json.dump(stats, f, indent=2)
                except Exception as e:
                    print(f"[WARNING] 统计信息更新失败: {e}")
                
            print(f"[DEBUG] 已保存日志到: {log_file}")
        except Exception as e:
            print(f"[ERROR] 保存日志失败: {e}")

    def _is_complete_sentence(self, text):
        """判断文本是否为完整句子"""
        # 移除空格和标点
        text = text.strip().rstrip('.!?')
        words = text.split()
        
        if len(words) < 3:  # 过短的句子视为不完整
            return False
            
        # 检查基本句子结构
        first_word = words[0].lower()
        
        # 1. 检查是否以常见主语开头
        common_subjects = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'there', 'this', 'that'}
        if first_word in common_subjects:
            return True
            
        # 2. 检查是否为疑问句
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        if first_word in question_words:
            return True
            
        # 3. 检查是否包含谓语动词
        common_verbs = {'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would'}
        if any(verb in words[1:3] for verb in common_verbs):
            return True
            
        return False

    def _translation_worker(self):
        while self.running:
            try:
                chinese_text, is_partial = self.translation_queue.get(timeout=1)
                if not self.running:
                    break
                
                if chinese_text:
                    if self.test_mode:
                        self.debug_info['translation_attempts'] += 1
                        self._append_debug(f"翻译中: {chinese_text} ({'部分' if is_partial else '完整'})")
                    
                    # 确定是否需要合并翻译
                    should_merge = self._should_merge_translations(
                        chinese_text, 
                        self.partial_translation if is_partial else self.current_translation
                    )
                    
                    if should_merge and self.test_mode:
                        self._append_debug("检测到语义连贯，将合并翻译")

                    translated_text = chinese_text
                    
                    # 判断句子完整性
                    is_complete = self._is_complete_sentence(translated_text)
                    if is_complete and is_partial:
                        # 如果检测到句子完整，更新部分翻译标志
                        is_partial = False
                        if self.test_mode:
                            self._append_debug("检测到完整句子，更新为完整翻译")
                    
                    if is_partial:  # 处理部分翻译
                        if should_merge and self.partial_translation:
                            self.partial_translation = f"{self.partial_translation} {translated_text}"
                            if self.test_mode:
                                self._append_debug(f"合并部分翻译: {self.partial_translation}")
                        else:
                            self.partial_translation = translated_text
                            if self.test_mode:
                                self._append_debug(f"设置部分翻译: {self.partial_translation}")
                        self.current_translation = f"{self.partial_translation}..."

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translation_history.append(f"[{timestamp}] (部分) {translated_text}")

                    else:  # 处理完整翻译
                        formatted_text = self._format_translation(translated_text, False)
                        if should_merge and self.current_translation:
                            merged_text = f"{self.current_translation.rstrip('.')} {formatted_text}"
                            self.current_translation = merged_text
                            if self.test_mode:
                                self._append_debug(f"合并完整翻译: {merged_text}")
                        else:
                            self.previous_translations.append(self.current_translation)
                            self.previous_translations = self.previous_translations[-5:]
                            self.current_translation = formatted_text
                            if self.test_mode:
                                self._append_debug(f"设置完整翻译: {formatted_text}")
                        self.partial_translation = ""

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.translation_history.append(f"[{timestamp}] {formatted_text}")
                        self.latest_partial = formatted_text
                        self.last_update_time = time.time()
                        
                        # 更新最新完整翻译和所有翻译记录
                        self.latest_complete_translation = formatted_text
                        self.all_translations.append(formatted_text)
                        self.all_translations = self.all_translations[-20:]  # 最多保留20条
                        
                        # 保存完整翻译到日志
                        self._save_translation_log(formatted_text, False)
                        
                        if self.test_mode:
                            self._append_debug(f"完整翻译已保存: {formatted_text}")
                    
                    if self.test_mode:
                        self._append_debug(f"翻译结果: {self.current_translation} ({'部分' if is_partial else '完整'})")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"翻译异常: {str(e)}")
                self.current_translation = f"翻译出错: {str(e)}"


    def run(self):
        """修改后的运行方法"""
        try:
            if self.input_mode == 'file':
                # 文件模式使用原有逻辑
                stream = self._get_audio_input_stream()
                self._run_file_mode(stream)
            elif self.input_mode == 'mic':
                # 麦克风模式等待 WebSocket 连接
                self._run_mic_mode()
            elif self.input_mode == 'gradio_mic':
                # Gradio麦克风模式 - 不需要额外处理，由Gradio界面直接调用
                self._run_gradio_mic_mode()
                
        except Exception as e:
            print(f"[ERROR] run 方法异常: {e}")
            raise
        finally:
            try:
                stream.stop_stream()
                stream.close()
                print("[DEBUG] 关闭音频输入流")
            except Exception as e:
                print(f"[ERROR] 关闭音频输入流异常: {e}")
            # 确保状态重置
            self.running = False

    def _run_file_mode(self, stream):
        """文件模式处理逻辑"""
        try:
            empty_chunks = 0  # 计数连续的空帧
            while self.running:  # 使用运行状态标志控制循环
                try:
                    audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                    chunk_len = len(audio_chunk)
                    
                    if chunk_len == 0:
                        empty_chunks += 1
                        if empty_chunks >= 100000:  # 连续 几个空帧就退出 一帧大概0.02秒
                            print("[DEBUG] 检测到连续空帧，音频可能已结束")
                            break
                    else:
                        empty_chunks = 0  # 重置空帧计数
                        
                    self._process_audio_chunk(audio_chunk)
                    
                    # 输出更多调试信息
                    if self.test_mode:
                        debug_info = self.get_debug_info()
                        if debug_info:
                            print(f"[DEBUG] 调试信息:\n{debug_info}")
                        if self.current_translation:
                            print(f"[DEBUG] 当前翻译: {self.current_translation}")
                    
                except Exception as e:
                    print(f"[ERROR] 主循环内异常: {e}")
                    break
        except Exception as e:
            print(f"[ERROR] run 方法异常: {e}")
            raise
        finally:
            try:
                stream.stop_stream()
                stream.close()
                print("[DEBUG] 关闭音频输入流")
            except Exception as e:
                print(f"[ERROR] 关闭音频输入流异常: {e}")
            # 确保状态重置
            self.running = False

    def _run_mic_mode(self):
        """麦克风模式处理逻辑"""
        print("[INFO] 等待客户端连接...")
        
        # 启动UDP服务器
        if not self.udp_server:
            self._setup_udp_server()
        
        # 等待音频数据，UDP服务器会自动调用回调函数
        while self.running:
            try:
                # 只需要等待，UDP服务器会自动调用audio_callback
                time.sleep(0.01)  # 短暂等待，减少CPU占用
                    
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"麦克风模式处理异常: {str(e)}")
                time.sleep(0.1)

    def _run_gradio_mic_mode(self):
        """Gradio麦克风模式处理逻辑"""
        print("[INFO] Gradio麦克风模式已启动...")
        
        # Gradio模式下不需要额外的处理循环，音频处理由Gradio界面直接调用
        while self.running:
            try:
                # 检查Gradio处理状态
                status = self.get_gradio_status()
                if self.test_mode and status['processing']:
                    self._append_debug(f"Gradio处理状态: {status}")
                
                time.sleep(0.1)  # 短暂等待，减少CPU占用
                    
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"Gradio麦克风模式处理异常: {str(e)}")
                time.sleep(0.1)
            
    def close(self):
        """改进的关闭方法"""
        print("[INFO] 开始关闭AudioTranslator...")
        self.running = False
        
        # 关闭 UDP 服务器
        if self.udp_server:
            print("[INFO] 关闭UDP服务器...")
            self.udp_server.stop()
            self.udp_server = None
            
        # 关闭 WebSocket 服务器
        if self.websocket_server:
            print("[INFO] 关闭WebSocket服务器...")
            try:
                asyncio.run(self.websocket_server.stop())
            except Exception as e:
                print(f"[WARNING] 关闭WebSocket服务器异常: {e}")
            self.websocket_server = None
            
        # 重置Gradio状态
        if self.input_mode == 'gradio_mic':
            print("[INFO] 重置Gradio状态...")
            self.reset_gradio_state()
            
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

    def _append_debug(self, msg):
        if not hasattr(self, 'debug_msgs'):
            self.debug_msgs = []
        self.debug_msgs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # 限制debug信息长度
        if len(self.debug_msgs) > 30:
            self.debug_msgs = self.debug_msgs[-30:]

    def get_debug_info(self):
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

    def _manage_audio_cache(self, audio_data):
        """管理音频缓存"""
        with self.cache_lock:
            self.audio_cache.append(audio_data)
            self.frame_counter += 1
            
            # 定期清理缓存
            if self.frame_counter % self.cache_cleanup_interval == 0:
                # 保留最近的帧，清理旧的
                while len(self.audio_cache) > self.max_cache_size // 2:
                    self.audio_cache.popleft()
                
                if self.test_mode:
                    self._append_debug(f"音频缓存清理: 当前大小={len(self.audio_cache)}")

    def _get_cached_audio(self, num_frames=10):
        """获取缓存的音频数据"""
        with self.cache_lock:
            if len(self.audio_cache) >= num_frames:
                return list(self.audio_cache)[-num_frames:]
            return list(self.audio_cache)

    def process_gradio_audio_stream(self, stream, new_chunk):
        """处理Gradio Audio流式音频数据"""
        try:
            if new_chunk is None:
                return stream, ""
            
            sr, y = new_chunk
            
            if self.test_mode:
                self._append_debug(f"Gradio音频流处理: sr={sr}, y_shape={y.shape}, y_dtype={y.dtype}")
            
            # 验证音频数据
            if not self._validate_audio_data(sr, y):
                return stream, ""
            
            # 转换为单声道
            if y.ndim > 1:
                y = y.mean(axis=1)
                if self.test_mode:
                    self._append_debug(f"转换为单声道: 新形状={y.shape}")
            
            # 改进的音频数据标准化处理
            # 根据Gradio文档，音频数据是16位int数组，范围从-32768到32767
            # 需要转换为float32并归一化到[-1, 1]范围，与文件模式保持一致
            y = y.astype(np.float32)
            
            if self.test_mode:
                self._append_debug(f"原始数据范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 使用与文件模式相同的归一化方式
            # 文件模式使用: audio_array.astype(np.float32) / 32768.0
            y = y / 32768.0
            
            if self.test_mode:
                self._append_debug(f"归一化后范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
            
            # 检查归一化后的数据范围
            max_amplitude = np.max(np.abs(y))
            if max_amplitude > 1.0:
                if self.test_mode:
                    self._append_debug(f"归一化后数据范围异常: {max_amplitude:.4f}，进行裁剪")
                # 裁剪到[-1, 1]范围
                y = np.clip(y, -1.0, 1.0)
            
            if self.test_mode:
                self._append_debug(f"音频数据归一化完成: 最大值={np.max(np.abs(y)):.4f}")
            
            # 重采样到16kHz（如果需要）
            if sr != RATE:
                y = self._resample_audio(y, sr, RATE)
            
            # 累积音频流
            if stream is not None:
                stream = np.concatenate([stream, y])
            else:
                stream = y
            
            # 限制音频流长度（避免内存溢出）
            max_length = RATE * 30  # 最多30秒
            if len(stream) > max_length:
                # 保留最新的音频数据
                stream = stream[-max_length:]
                if self.test_mode:
                    self._append_debug(f"音频流长度限制: 保留最新{max_length}采样点")
            
            # 处理音频块
            self._process_gradio_audio_chunk(y)
            
            return stream, ""
            
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"Gradio音频处理异常: {str(e)}")
            return stream, ""

    def _validate_audio_data(self, sr, y):
        """验证音频数据的有效性"""
        try:
            if y is None or len(y) == 0:
                if self.test_mode:
                    self._append_debug("音频数据为空")
                return False
            
            if sr <= 0:
                if self.test_mode:
                    self._append_debug(f"采样率无效: {sr}")
                return False
            
            if np.isnan(y).any() or np.isinf(y).any():
                if self.test_mode:
                    self._append_debug("音频数据包含NaN或Inf值")
                return False
            
            # 放宽音频数据范围检查，麦克风数据可能超出[-1, 1]范围
            max_amplitude = np.max(np.abs(y))
            if max_amplitude > 10.0:  # 允许更大的范围，但记录警告
                if self.test_mode:
                    self._append_debug(f"音频数据超出正常范围: {max_amplitude:.2f}，将进行归一化处理")
            
            return True
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"音频数据验证异常: {str(e)}")
            return False

    def _resample_audio(self, audio, original_sr, target_sr):
        """重采样音频数据"""
        try:
            if original_sr == target_sr:
                return audio
            
            # 检查输入音频数据是否有效
            if np.isnan(audio).any() or np.isinf(audio).any():
                if self.test_mode:
                    self._append_debug("重采样输入包含无效值，返回原音频")
                return audio
            
            # 简单的线性插值重采样
            ratio = target_sr / original_sr
            new_length = int(len(audio) * ratio)
            
            # 确保新长度不为0
            if new_length <= 0:
                if self.test_mode:
                    self._append_debug(f"重采样长度异常: {new_length}")
                return audio
            
            # 使用更安全的插值方法
            try:
                indices = np.linspace(0, len(audio) - 1, new_length)
                resampled = np.interp(indices, np.arange(len(audio)), audio)
                
                # 检查重采样结果是否有效
                if np.isnan(resampled).any() or np.isinf(resampled).any():
                    if self.test_mode:
                        self._append_debug("重采样结果包含无效值，返回原音频")
                    return audio
                
                if self.test_mode:
                    self._append_debug(f"重采样: {original_sr}Hz -> {target_sr}Hz, {len(audio)} -> {len(resampled)}")
                
                return resampled
                
            except Exception as e:
                if self.test_mode:
                    self._append_debug(f"重采样插值异常: {str(e)}")
                return audio
                
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"重采样异常: {str(e)}")
            return audio

    def _normalize_audio_data(self, audio_data):
        """统一音频数据格式处理，确保与文件模式一致"""
        try:
            if audio_data is None or len(audio_data) == 0:
                return None
            
            # 确保数据类型为float32
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            else:
                audio_data = audio_data.astype(np.float32)
            
            # 处理音频数据范围
            max_amplitude = np.max(np.abs(audio_data))
            
            if max_amplitude > 0:
                # 如果数据范围过大，进行软裁剪
                if max_amplitude > 10.0:
                    audio_data = np.clip(audio_data, -10.0, 10.0)
                    max_amplitude = 10.0
                    if self.test_mode:
                        self._append_debug(f"音频数据软裁剪: 最大值={max_amplitude:.2f}")
                
                # 归一化到[-1, 1]范围
                audio_data = audio_data / max_amplitude
            else:
                # 静音帧，保持原样
                if self.test_mode:
                    self._append_debug("检测到静音帧")
            
            return audio_data
            
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"音频数据归一化异常: {str(e)}")
            return None

    def _process_gradio_audio_chunk(self, audio_chunk):
        """处理Gradio音频块"""
        try:
            with self.gradio_audio_lock:
                # 设置处理状态
                self.gradio_processing = True
                
                # 添加到Gradio音频缓冲区
                self.gradio_audio_buffer.append(audio_chunk)
                
                if self.test_mode:
                    self._append_debug(f"Gradio音频块处理开始: 长度={len(audio_chunk)}, 类型={type(audio_chunk)}")
                
                # 改进的音频数据转换逻辑
                # Gradio音频已经是float32格式且已归一化到[-1, 1]，需要转换为int16字节格式
                try:
                    # 确保音频数据是numpy数组
                    if not isinstance(audio_chunk, np.ndarray):
                        if self.test_mode:
                            self._append_debug(f"音频数据类型错误: {type(audio_chunk)}，尝试转换")
                        audio_chunk = np.array(audio_chunk, dtype=np.float32)
                    
                    # 检查音频数据是否有效并尝试修复
                    if np.isnan(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug(f"音频数据包含NaN值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复NaN值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if np.isinf(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug(f"音频数据包含Inf值，尝试修复。数据范围: [{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                        # 尝试修复Inf值
                        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # 最终检查修复后的数据
                    if np.isnan(audio_chunk).any() or np.isinf(audio_chunk).any():
                        if self.test_mode:
                            self._append_debug("音频数据修复后仍包含无效值，跳过处理")
                        return
                    
                    # 确保音频数据在合理范围内
                    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                    
                    if self.test_mode:
                        self._append_debug(f"音频数据预处理完成: 范围=[{np.min(audio_chunk):.4f}, {np.max(audio_chunk):.4f}]")
                    
                    # 转换为int16格式，与文件模式保持一致
                    # 文件模式使用: data.tobytes()，其中data是float32格式
                    # 我们需要将float32转换为int16，然后转换为bytes
                    audio_int16 = (audio_chunk * 32768.0).astype(np.int16)
                    
                    # 检查转换是否成功
                    if np.isnan(audio_int16).any() or np.isinf(audio_int16).any():
                        if self.test_mode:
                            self._append_debug("int16转换失败，包含无效值")
                        return
                    
                    audio_bytes = audio_int16.tobytes()
                    
                    if self.test_mode:
                        self._append_debug(f"Gradio音频转换成功: 长度={len(audio_chunk)}, 最大值={np.max(np.abs(audio_chunk)):.4f}")
                    
                    # 使用现有的音频处理逻辑
                    self._process_audio_chunk(audio_bytes)
                    
                except Exception as e:
                    if self.test_mode:
                        self._append_debug(f"Gradio音频转换异常: {str(e)}")
                    # 如果转换失败，尝试直接处理float32数据
                    try:
                        # 直接使用float32数据，但需要确保格式正确
                        if isinstance(audio_chunk, np.ndarray):
                            # 将float32数据转换为bytes格式
                            audio_bytes = audio_chunk.tobytes()
                            if self.test_mode:
                                self._append_debug("使用备用方法处理float32数据")
                            self._process_audio_chunk(audio_bytes)
                        else:
                            if self.test_mode:
                                self._append_debug("音频数据类型不支持")
                    except Exception as e2:
                        if self.test_mode:
                            self._append_debug(f"备用音频处理也失败: {str(e2)}")
                
        except Exception as e:
            if self.test_mode:
                self._append_debug(f"Gradio音频块处理异常: {str(e)}")
        finally:
            with self.gradio_audio_lock:
                self.gradio_processing = False

    def reset_gradio_state(self):
        """重置Gradio相关状态"""
        with self.gradio_audio_lock:
            self.gradio_audio_buffer.clear()
            self.gradio_processing = False
            self.gradio_audio_stream = None

    def get_gradio_status(self):
        """获取Gradio处理状态"""
        with self.gradio_audio_lock:
            return {
                'processing': self.gradio_processing,
                'buffer_size': len(self.gradio_audio_buffer),
                'stream_active': self.gradio_audio_stream is not None
            }

def create_translation_interface():
    translator = AudioTranslator(test_mode=True)
    
    # 添加全局停止机制
    def cleanup_on_exit():
        """应用退出时的清理函数"""
        try:
            if translator.running:
                translator.close()
                print("[INFO] 应用退出时清理完成")
        except Exception as e:
            print(f"[ERROR] 应用退出时清理异常: {e}")
    
    # 注册退出时的清理函数
    import atexit
    atexit.register(cleanup_on_exit)
    
    with gr.Blocks(title="实时语音翻译系统") as demo:
        # 修改Timer配置
        update_timer = gr.Timer(value=0.3, active=False)
        
        gr.Markdown("## 实时中英语音翻译系统 (v6 Gradio优化版)")
        
        with gr.Row():
            with gr.Column(scale=1):
                status_indicator = gr.Label("系统状态: 等待启动", label="状态指示")
                with gr.Row():
                    file_btn = gr.Button("从音频文件翻译", variant="primary")
                    mic_btn = gr.Button("从麦克风翻译", variant="primary")
                    stop_btn = gr.Button("停止翻译", variant="secondary")
                
                # 添加文件上传组件
                audio_file = gr.Audio(
                    type="filepath",
                    label="选择音频文件",
                    visible=False
                )
                
                # 新增：Gradio流式麦克风组件
                gradio_mic = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    label="实时麦克风输入",
                    visible=False
                )
            
            with gr.Column(scale=3):
                latest_translation = gr.Textbox(
                    label="最新完整翻译",
                    placeholder="等待翻译...",
                    lines=1,
                    interactive=False
                )
                
                translation_output = gr.Textbox(
                    label="实时英文字幕",
                    placeholder="等待语音输入...",
                    lines=3,
                    interactive=False
                )
                
                history_output = gr.Textbox(
                    label="翻译历史",
                    lines=5,
                    interactive=False
                )
        
        debug_output = gr.Textbox(
            label="调试日志",
            lines=10,
            interactive=False
        )

        # 状态管理
        running = gr.State(False)

        def update_components():
            """获取最新翻译数据"""
            try:
                if running.value and translator.running:
                    current_text = translator.current_translation
                    latest_complete = translator.latest_complete_translation or "等待完整翻译..."
                    
                    # 构建历史记录
                    history = []
                    if translator.latest_complete_translation:
                        history.append(translator.latest_complete_translation)
                    history.extend(translator.translation_history[-4:])  # 只取最近4条
                    history_text = "\n".join(filter(None, history)) or "等待翻译..."
                    
                    # 获取调试信息
                    debug_text = translator.get_debug_info() or "等待日志..."
                    
                    # 根据输入模式显示不同状态
                    status_text = "系统状态: 运行中"
                    if translator.input_mode == 'file':
                        status_text = "系统状态: 运行中(文件模式)"
                    elif translator.input_mode == 'mic':
                        status_text = "系统状态: 运行中(UDP麦克风模式)"
                    elif translator.input_mode == 'gradio_mic':
                        status_text = "系统状态: 运行中(Gradio麦克风模式)"
                        # 获取Gradio处理状态
                        gradio_status = translator.get_gradio_status()
                        if gradio_status['processing']:
                            status_text += f" - 处理中(缓冲区:{gradio_status['buffer_size']})"
                        elif gradio_status['buffer_size'] > 0:
                            status_text += f" - 等待处理(缓冲区:{gradio_status['buffer_size']})"
                    
                    return (
                        latest_complete,
                        current_text or "等待输入...",
                        history_text,
                        debug_text,
                        status_text
                    )
                return (
                    "翻译已停止",
                    "翻译已停止",
                    translator.get_all_translations() or "无历史记录",
                    translator.get_debug_info() or "无日志",
                    "系统状态: 已停止"
                )
            except Exception as e:
                print(f"[ERROR] 更新异常: {e}")
                return (
                    "更新出错",
                    "更新出错",
                    "更新出错",
                    f"异常: {str(e)}",
                    "系统状态: 错误"
                )

        def start_translation_from_file(file_path):
            """从文件启动翻译"""
            try:
                if not running.value:
                    translator.set_input_mode('file', file_path)
                    running.value = True
                    translator.running = True
                    translator.translation_history = []
                    translator.all_translations = []
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(文件模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=False)
                    ]
                else:
                    # 如果已经在运行，先停止再重新启动
                    translator.close()
                    running.value = False
                    time.sleep(0.5)  # 等待资源释放
                    
                    translator.set_input_mode('file', file_path)
                    running.value = True
                    translator.running = True
                    translator.translation_history = []
                    translator.all_translations = []
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(文件模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=False)
                    ]
            except Exception as e:
                print(f"[ERROR] 启动失败: {e}")
                return [
                    False,
                    "启动失败",
                    "启动失败",
                    "启动失败",
                    f"异常: {str(e)}",
                    "系统状态: 错误",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=True)
                ]

        def start_translation_from_mic():
            """从麦克风启动翻译（Gradio流式模式）"""
            try:
                if not running.value:
                    translator.set_input_mode('gradio_mic')
                    running.value = True
                    translator.running = True
                    translator.translation_history = []
                    translator.all_translations = []
                    translator.reset_gradio_state()  # 重置Gradio状态
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(Gradio麦克风模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=False),
                        gr.update(visible=True)  # 显示Gradio麦克风组件
                    ]
                else:
                    # 如果已经在运行，先停止再重新启动
                    translator.close()
                    running.value = False
                    time.sleep(0.5)  # 等待资源释放
                    
                    translator.set_input_mode('gradio_mic')
                    running.value = True
                    translator.running = True
                    translator.translation_history = []
                    translator.all_translations = []
                    translator.reset_gradio_state()  # 重置Gradio状态
                    
                    threading.Thread(
                        target=translator.run,
                        daemon=True
                    ).start()
                    
                    return [
                        True,
                        "等待输入...",
                        "等待输入...",
                        "无历史记录",
                        translator.get_debug_info(),
                        "系统状态: 运行中(Gradio麦克风模式)",
                        gr.Timer(value=0.3, active=True),
                        gr.update(visible=False),
                        gr.update(visible=True)  # 显示Gradio麦克风组件
                    ]
            except Exception as e:
                print(f"[ERROR] 启动失败: {e}")
                return [
                    False,
                    "启动失败",
                    "启动失败",
                    "启动失败",
                    f"异常: {str(e)}",
                    "系统状态: 错误",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                ]

        def process_gradio_mic_stream(stream, new_chunk):
            """处理Gradio麦克风流式音频"""
            try:
                if not running.value or not translator.running:
                    return stream, ""
                
                # 调用translator的Gradio音频处理方法
                updated_stream, _ = translator.process_gradio_audio_stream(stream, new_chunk)
                return updated_stream, ""
                
            except Exception as e:
                print(f"[ERROR] Gradio麦克风处理异常: {e}")
                return stream, ""

        def stop_translation():
            """停止翻译流程"""
            try:
                running.value = False
                translator.running = False
                translator.close()
                translator.reset_gradio_state()  # 重置Gradio状态
                summary = translator.get_all_translations()
                return [
                    "翻译已停止",
                    "翻译已停止",
                    summary or "无历史记录",
                    "无日志",
                    "系统状态: 已停止",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=True),
                    gr.update(visible=False)  # 隐藏Gradio麦克风组件
                ]
            except Exception as e:
                print(f"[ERROR] 停止失败: {e}")
                return [
                    "停止失败",
                    "停止失败",
                    "停止失败",
                    f"异常: {str(e)}",
                    "系统状态: 错误",
                    gr.Timer(value=0.3, active=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                ]

        # 事件绑定
        update_timer.tick(
            fn=update_components,
            outputs=[
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator
            ]
        )

        file_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=audio_file
        )

        audio_file.change(
            fn=start_translation_from_file,
            inputs=audio_file,
            outputs=[
                running,
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator,
                update_timer,
                audio_file
            ]
        )
        
        mic_btn.click(
            fn=start_translation_from_mic,
            outputs=[
                running,
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator,
                update_timer,
                audio_file,
                gradio_mic
            ]
        )
        
        # 新增：Gradio麦克风流式处理事件绑定
        gradio_mic.stream(
            fn=process_gradio_mic_stream,
            inputs=[gr.State(None), gradio_mic],
            outputs=[gr.State(None), gr.Textbox()],
            show_progress=False,
            batch=False,
            max_batch_size=1
        )
        
        stop_btn.click(
            fn=stop_translation,
            outputs=[
                latest_translation,
                translation_output,
                history_output,
                debug_output,
                status_indicator,
                update_timer,
                audio_file,
                gradio_mic
            ]
        )

    return demo


if __name__ == "__main__":
    try:
        demo = create_translation_interface()
        demo.queue(
            default_concurrency_limit=3,
            api_open=False,
            max_size=100  # 增加队列大小以处理更多更新请求
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            debug=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在关闭应用...")
    except Exception as e:
        print(f"[ERROR] 应用启动异常: {e}")
    finally:
        print("[INFO] 应用已退出")
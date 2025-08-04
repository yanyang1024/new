"""
音频配置模块
包含所有音频处理相关的配置参数
"""

# 音频配置
RATE = 16000
CHANNELS = 1
CHUNK = 320
FORMAT = "paFloat32"  # 字符串形式，使用时需要转换
VAD_MODE = 2

# 语音检测配置
PARTIAL_UPDATE_FRAMES = 40  # 增加到800ms进行一次部分翻译
SILENCE_THRESHOLD_SHORT = 25  # 增加到500ms静音
SILENCE_THRESHOLD_LONG = 35  # 增加到700ms静音
MIN_SPEECH_FRAMES = 20  # 至少需要400ms的语音

# 模型路径配置
PATH_FOR_SEAMLESS_M4T = "/data/checkpoints/seamless-m4t-v2-large"

# 音频缓存配置
MAX_CACHE_SIZE = 200
CACHE_CLEANUP_INTERVAL = 1000

# 音频处理配置
ENERGY_THRESHOLD = 0.01  # 语音能量阈值，可以动态调整
MAX_AUDIO_LENGTH = 30  # 最大音频长度（秒）

# 音频数据范围配置
MAX_AMPLITUDE = 1000.0  # 音频数据最大振幅
NORMALIZATION_FACTOR = 32768.0  # 归一化因子 
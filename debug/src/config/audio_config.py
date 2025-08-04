# 音频配置参数

# 音频格式配置
RATE = 16000
CHANNELS = 1
CHUNK = 320
FORMAT = "paFloat32"  # 字符串格式，在代码中会转换为pyaudio常量

# VAD配置
VAD_MODE = 2

# 语音检测配置
PARTIAL_UPDATE_FRAMES = 40  # 增加到800ms进行一次部分翻译
SILENCE_THRESHOLD_SHORT = 25  # 增加到500ms静音
SILENCE_THRESHOLD_LONG = 35  # 增加到700ms静音
MIN_SPEECH_FRAMES = 20  # 至少需要400ms的语音

# 模型路径配置
PATH_FOR_SEAMLESS_M4T = "/data/checkpoints/seamless-m4t-v2-large"

# 音频处理配置
ENERGY_THRESHOLD = 0.01  # 语音能量阈值，可以动态调整
MAX_AMPLITUDE = 1000.0  # 音频数据最大振幅
NORMALIZATION_FACTOR = 32768.0  # 音频归一化因子

# 缓存配置
MAX_CACHE_SIZE = 200  # 最大缓存大小
CACHE_CLEANUP_INTERVAL = 1000  # 每1000帧清理一次缓存
MAX_AUDIO_LENGTH = 30  # 最多30秒音频

# 翻译配置
MAX_NEW_TOKENS = 200  # 最大新token数
NUM_BEAMS = 3  # beam search数量
EARLY_STOPPING = True  # 早停机制

# 句子处理配置
PUNCTUATION_THRESHOLD = 2.0  # 句子间隔阈值（秒）
MAX_HISTORY_LENGTH = 20  # 最多保留20条翻译记录
MAX_DEBUG_MESSAGES = 30  # 最多保留30条调试信息 
"""
翻译配置模块
包含翻译处理相关的配置参数
"""

# 翻译处理配置
PUNCTUATION_THRESHOLD = 2.0  # 句子间隔阈值（秒）
MAX_NEW_TOKENS = 200  # 最大新token数
NUM_BEAMS = 3  # 束搜索数量
EARLY_STOPPING = True  # 是否启用早停

# 翻译历史配置
MAX_TRANSLATION_HISTORY = 20  # 最大翻译历史记录数
MAX_PREVIOUS_TRANSLATIONS = 5  # 最大前序翻译记录数

# 翻译结果处理配置
MAX_SENTENCE_LENGTH = 100  # 最大句子长度
MIN_SENTENCE_LENGTH = 3  # 最小句子长度

# 翻译质量配置
VAD_RATIO_THRESHOLD = 0.4  # VAD比例阈值
ENERGY_MULTIPLIER = 1.1  # 能量阈值调整倍数
ENERGY_MIN_THRESHOLD = 0.005  # 最小能量阈值
ENERGY_MAX_THRESHOLD = 0.02  # 最大能量阈值

# 翻译模式配置
SUPPORTED_LANGUAGES = ["eng"]  # 支持的目标语言
DEFAULT_TARGET_LANG = "eng"  # 默认目标语言 
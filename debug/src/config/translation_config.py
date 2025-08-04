# 翻译配置参数

# 目标语言配置
TARGET_LANGUAGE = "eng"  # 目标语言为英语

# 翻译模型配置
MODEL_CONFIG = {
    "device_map": "auto",
    "torch_dtype": "float16",
    "max_new_tokens": 200,
    "num_beams": 3,
    "early_stopping": True
}

# 句子完整性检测配置
COMPLETE_SENTENCE_CONFIG = {
    "min_words": 3,  # 最少单词数
    "common_subjects": {
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'there', 'this', 'that'
    },
    "question_words": {
        'what', 'how', 'why', 'when', 'where', 'who', 'which'
    },
    "common_verbs": {
        'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does', 'can', 'will', 'would'
    }
}

# 翻译合并配置
MERGE_CONFIG = {
    "time_threshold": 2.0,  # 时间阈值（秒）
    "min_overlap": 1,  # 最小词重叠数
    "subject_continuity": True  # 是否检查主语连贯性
}

# 日志配置
LOG_CONFIG = {
    "log_dir": "translation_logs",
    "stats_file": "stats_{date}.json",
    "max_log_entries": 1000
}

# 服务器配置
SERVER_CONFIG = {
    "websocket": {
        "host": "0.0.0.0",
        "port": 8765
    },
    "udp": {
        "host": "0.0.0.0", 
        "port": 12345
    },
    "gradio": {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "show_error": True,
        "debug": True
    }
} 
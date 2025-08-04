# 实时语音翻译系统 v6 Gradio优化版

## 项目概述

这是一个基于SeamlessM4T模型的实时语音翻译系统，支持中文语音到英文文本的实时翻译。系统采用模块化设计，具有良好的可维护性和扩展性。

## 项目结构

```
todo/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── config/                   # 配置模块
│   │   ├── __init__.py
│   │   ├── audio_config.py       # 音频配置
│   │   └── translation_config.py # 翻译配置
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── audio_utils.py        # 音频工具函数
│   │   └── translation_utils.py  # 翻译工具函数
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   └── audio_translator.py   # 音频翻译器核心
│   ├── audio/                    # 音频处理模块
│   │   ├── __init__.py
│   │   └── audio_processor.py    # 音频处理器
│   ├── translation/              # 翻译模块
│   │   ├── __init__.py
│   │   └── translator.py         # 翻译处理器
│   └── ui/                       # 用户界面模块
│       ├── __init__.py
│       └── gradio_interface.py   # Gradio界面
├── main.py                       # 主应用入口
├── requirements.txt              # 依赖包列表
├── README.md                     # 项目说明
└── app.py                        # 原始单文件版本（保留）
```

## 模块说明

### 配置模块 (src/config/)
- **audio_config.py**: 包含音频处理相关的配置参数，如采样率、VAD模式、缓存设置等
- **translation_config.py**: 包含翻译相关的配置，如目标语言、模型参数、服务器设置等

### 工具模块 (src/utils/)
- **audio_utils.py**: 提供音频数据验证、归一化、重采样等工具函数
- **translation_utils.py**: 提供翻译文本格式化、合并、日志保存等工具函数

### 核心模块 (src/core/)
- **audio_translator.py**: 核心翻译器类，负责模型加载、VAD检测、状态管理等

### 音频处理模块 (src/audio/)
- **audio_processor.py**: 音频处理器，负责音频流处理、Gradio音频流处理等

### 翻译模块 (src/translation/)
- **translator.py**: 翻译处理器，负责语音到文本的翻译处理

### 用户界面模块 (src/ui/)
- **gradio_interface.py**: Gradio界面实现，提供用户交互界面

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

确保SeamlessM4T模型已下载到指定路径：
```
/data/checkpoints/seamless-m4t-v2-large
```

### 3. 运行应用

```bash
python main.py
```

应用将在 `http://localhost:7860` 启动。

## 功能特性

### 支持的输入模式
1. **文件模式**: 从音频文件进行翻译
2. **Gradio麦克风模式**: 使用Gradio的流式麦克风输入
3. **UDP麦克风模式**: 通过UDP接收音频数据（需要额外配置）

### 核心功能
- 实时语音检测 (VAD)
- 流式音频处理
- 部分翻译和完整翻译
- 翻译结果合并和格式化
- 调试日志和状态监控
- 翻译历史记录

### 优化特性
- GPU加速支持
- 音频缓存管理
- 内存使用优化
- 错误处理和恢复
- 模块化设计

## 配置说明

### 音频配置 (src/config/audio_config.py)
```python
RATE = 16000                    # 采样率
CHANNELS = 1                    # 声道数
CHUNK = 320                     # 音频块大小
VAD_MODE = 2                    # VAD模式
PARTIAL_UPDATE_FRAMES = 40      # 部分更新帧数
SILENCE_THRESHOLD_SHORT = 25    # 短静音阈值
SILENCE_THRESHOLD_LONG = 35     # 长静音阈值
```

### 翻译配置 (src/config/translation_config.py)
```python
TARGET_LANGUAGE = "eng"         # 目标语言
MODEL_CONFIG = {
    "max_new_tokens": 200,      # 最大新token数
    "num_beams": 3,             # beam search数量
    "early_stopping": True      # 早停机制
}
```

## 使用说明

1. **启动应用**: 运行 `python main.py`
2. **选择输入模式**: 
   - 点击"从音频文件翻译"选择音频文件
   - 点击"从麦克风翻译"使用实时麦克风
3. **查看翻译结果**: 界面会实时显示翻译结果和历史记录
4. **停止翻译**: 点击"停止翻译"按钮

## 调试和日志

系统提供详细的调试信息：
- VAD检测结果
- 音频处理状态
- 翻译进度
- 错误信息

调试日志会显示在界面的"调试日志"区域。

## 性能优化

- 使用GPU加速（如果可用）
- 音频缓存管理
- 流式处理减少延迟
- 内存使用优化

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的GPU内存

2. **音频设备问题**
   - 检查麦克风权限
   - 确认音频设备正常工作

3. **依赖包问题**
   - 运行 `pip install -r requirements.txt`
   - 检查Python版本兼容性

## 开发说明

### 添加新功能
1. 在相应模块中添加功能
2. 更新配置文件
3. 修改界面（如需要）
4. 更新文档

### 代码规范
- 使用中文注释
- 遵循PEP 8代码风格
- 添加适当的错误处理
- 编写单元测试（建议）

## 版本历史

- **v6**: 模块化重构，优化代码结构
- **v5**: Gradio界面优化
- **v4**: 音频处理改进
- **v3**: 翻译质量提升
- **v2**: 性能优化
- **v1**: 基础功能实现

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时语音翻译系统 - 主应用文件
版本: v6 Gradio优化版

这个文件是重构后的主入口，整合了所有模块：
- src/core/audio_translator.py: 核心翻译器
- src/audio/audio_processor.py: 音频处理器
- src/translation/translator.py: 翻译处理器
- src/ui/gradio_interface.py: Gradio界面
- src/config/: 配置模块
- src/utils/: 工具模块
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.gradio_interface import GradioInterface

def main():
    """主函数"""
    print("=" * 60)
    print("实时语音翻译系统 v6 Gradio优化版")
    print("=" * 60)
    print("正在启动系统...")
    
    try:
        # 创建并启动Gradio界面
        interface = GradioInterface()
        interface.launch()
        
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在关闭应用...")
    except Exception as e:
        print(f"[ERROR] 应用启动异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] 应用已退出")

if __name__ == "__main__":
    main() 
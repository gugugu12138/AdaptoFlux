# file: adaptoflux/model_trainer/method_pool/__init__.py
"""
方法池进化模块

Usage:
    from adaptoflux.model_trainer.method_pool import MethodPoolEvolver
"""
from .evolver import MethodPoolEvolver

# 可选：导出子模块供高级用户直接使用
# from .signature_analyzer import SignatureAnalyzer
# from .graph_processor import EvolutionGraphProcessor

__all__ = ['MethodPoolEvolver']
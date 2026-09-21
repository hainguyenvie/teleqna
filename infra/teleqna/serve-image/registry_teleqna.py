"""Chỉ đăng ký task teleqna: image này không mang 7 benchmark còn lại của Open Telco,
nên `evals._registry` gốc (import cả 8) sẽ lỗi. Inspect nạp entry point này lúc khởi động."""
from evals.teleqna.teleqna import teleqna

__all__ = ["teleqna"]

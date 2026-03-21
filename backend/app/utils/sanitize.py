"""输入安全 — 长度限制、控制字符过滤、注入模式检测"""

import re
import logging

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 2000

# 常见 prompt injection 模式
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
]


def sanitize_input(text: str) -> str:
    """清理用户输入：长度截断、控制字符过滤、注入检测。

    返回清理后的文本，或空字符串表示拒绝。
    """
    if not text or not text.strip():
        return ""

    # 长度限制
    text = text[:MAX_INPUT_LENGTH]

    # 移除控制字符（保留换行和制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 注入模式检测
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("[SANITIZE] Injection pattern detected: %s", text[:80])
            # 不拒绝，但清除可疑片段
            text = pattern.sub("", text)

    return text.strip()

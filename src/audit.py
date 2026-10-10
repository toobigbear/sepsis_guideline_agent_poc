# -*- coding: utf-8 -*-
"""审计留痕：所有工具调用、Harness 判定、拦截事件写入 jsonl（黑匣子）。"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import config


class AuditLogger:
    def __init__(self, path: str = config.AUDIT_PATH) -> None:
        self.path = path

    def log(self, event: str, **payload: Any) -> None:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "event": event, **payload}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

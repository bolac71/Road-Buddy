from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class KeyState:
    key: str
    index: int
    cooldown_until: float = 0.0
    disabled: bool = False
    last_reason: str = ""
    total_success: int = 0
    total_fail: int = 0
    total_switch_away: int = 0


class GroqKeyPool:
    def __init__(self, env_path: str, env_name: str = "GROQ_API_KEY"):
        env_file = Path(env_path)
        if not env_file.exists():
            raise FileNotFoundError(f"Khong tim thay env file: {env_path}")

        load_dotenv(env_file, override=True)
        raw = os.getenv(env_name, "").strip()
        if not raw:
            raise ValueError(f"Khong tim thay bien {env_name} trong file .env")

        keys = [x.strip() for x in raw.split(",") if x.strip()]
        if not keys:
            raise ValueError(f"{env_name} rong sau khi parse")

        self._keys: list[KeyState] = [KeyState(key=k, index=i) for i, k in enumerate(keys)]
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._keys)

    def _now(self) -> float:
        return time.time()

    def acquire(self) -> KeyState:
        with self._lock:
            now = self._now()
            alive = [k for k in self._keys if not k.disabled]
            if not alive:
                raise RuntimeError("Khong con Groq API key hop le nao trong pool")

            n = len(self._keys)
            for _ in range(n):
                ks = self._keys[self._cursor]
                self._cursor = (self._cursor + 1) % n
                if ks.disabled:
                    continue
                if ks.cooldown_until <= now:
                    return ks

            return min(alive, key=lambda x: x.cooldown_until)

    def report_success(self, key_index: int) -> None:
        with self._lock:
            self._keys[key_index].total_success += 1
            self._keys[key_index].last_reason = ""

    def report_failure(
        self,
        key_index: int,
        reason: str,
        cooldown_sec: int,
        disable_forever: bool = False,
    ) -> None:
        with self._lock:
            now = self._now()
            ks = self._keys[key_index]
            ks.total_fail += 1
            ks.total_switch_away += 1
            ks.last_reason = reason

            if disable_forever:
                ks.disabled = True
                ks.cooldown_until = float("inf")
                return

            ks.cooldown_until = max(ks.cooldown_until, now + max(0, cooldown_sec))

    def snapshot(self) -> list[dict]:
        with self._lock:
            now = self._now()
            out = []
            for ks in self._keys:
                out.append({
                    "key_index": ks.index,
                    "available": ks.cooldown_until <= now,
                    "cooldown_remaining_sec": max(0.0, ks.cooldown_until - now),
                    "last_reason": ks.last_reason,
                    "total_success": ks.total_success,
                    "total_fail": ks.total_fail,
                    "total_switch_away": ks.total_switch_away,
                    "disabled": ks.disabled,
                })
            return out

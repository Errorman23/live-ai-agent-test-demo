from __future__ import annotations

import re
import unicodedata


COMPANY_ALIAS_MAP: dict[str, str] = {
    # Tencent
    "tencent": "Tencent",
    "腾讯": "Tencent",
    "騰訊": "Tencent",
    # Volkswagen
    "volkswagen": "Volkswagen",
    "vw": "Volkswagen",
    "大众": "Volkswagen",
    "福斯": "Volkswagen",
    # TikTok
    "tiktok": "TikTok",
    "抖音": "TikTok",
    "字节跳动": "TikTok",
    # Tesla
    "tesla": "Tesla",
    "特斯拉": "Tesla",
    # Siemens
    "siemens": "Siemens",
    "西门子": "Siemens",
    # Pfizer
    "pfizer": "Pfizer",
    "辉瑞": "Pfizer",
    # Samsung
    "samsung": "Samsung",
    "三星": "Samsung",
    # Shell
    "shell": "Shell",
    "壳牌": "Shell",
    # Sony
    "sony": "Sony",
    "索尼": "Sony",
    # Grab
    "grab": "Grab",
    "grab控股": "Grab",
    # OpenAI
    "openai": "OpenAI",
    "open ai": "OpenAI",
}


LANGUAGE_ALIAS_MAP: dict[str, str] = {
    # English
    "english": "English",
    "英文": "English",
    "英语": "English",
    "in english": "English",
    "to english": "English",
    # German
    "german": "German",
    "deutsch": "German",
    "德语": "German",
    "德文": "German",
    "auf deutsch": "German",
    "in german": "German",
    # Chinese
    "chinese": "Chinese",
    "中文": "Chinese",
    "汉语": "Chinese",
    "華語": "Chinese",
    "華文": "Chinese",
    "普通话": "Chinese",
    "简体中文": "Chinese",
    "繁體中文": "Chinese",
    # Japanese
    "japanese": "Japanese",
    "日本語": "Japanese",
    "日语": "Japanese",
    "日文": "Japanese",
    # French
    "french": "French",
    "français": "French",
    "francais": "French",
    "法语": "French",
    # Spanish
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "西班牙语": "Spanish",
}


def _normalize_text(value: str) -> str:
    collapsed = unicodedata.normalize("NFKC", value).strip()
    return re.sub(r"\s+", " ", collapsed)


def canonicalize_language(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_text(value).lower()
    if not normalized:
        return None
    return LANGUAGE_ALIAS_MAP.get(normalized)


def canonicalize_company(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_text(value)
    lower = normalized.lower()
    if lower in COMPANY_ALIAS_MAP:
        return COMPANY_ALIAS_MAP[lower]
    return normalized if normalized else None


def extract_language_from_prompt(prompt: str) -> tuple[str | None, str]:
    value = _normalize_text(prompt)
    lower = value.lower()
    aliases = sorted(LANGUAGE_ALIAS_MAP.keys(), key=len, reverse=True)
    for alias in aliases:
        target = LANGUAGE_ALIAS_MAP[alias]
        if _is_non_latin(alias):
            if alias in value:
                return target, "alias"
            continue
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return target, "alias"
    return None, "heuristic"


def extract_company_from_prompt(prompt: str) -> tuple[str | None, str]:
    value = _normalize_text(prompt)
    lower = value.lower()
    aliases = sorted(COMPANY_ALIAS_MAP.keys(), key=len, reverse=True)
    for alias in aliases:
        canonical = COMPANY_ALIAS_MAP[alias]
        if _is_non_latin(alias):
            if alias in value:
                return canonical, "alias"
            continue
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return canonical, "alias"
    return None, "heuristic"


def _is_non_latin(value: str) -> bool:
    return any(ord(ch) > 127 for ch in value)


from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
ENV_VALUES = dotenv_values(ENV_FILE)
load_dotenv(ENV_FILE)


def _env(name: str, default: str = "", *, prefer_dotenv: bool = False) -> str:
    if prefer_dotenv:
        dotenv_value = ENV_VALUES.get(name)
        if dotenv_value not in (None, ""):
            return str(dotenv_value)
    if name in os.environ:
        return os.environ[name]
    dotenv_value = ENV_VALUES.get(name)
    if dotenv_value is not None:
        return str(dotenv_value)
    return default


class Settings(BaseModel):
    app_name: str = Field(default=_env("APP_NAME", "enterprise-document-qa-agent", prefer_dotenv=True))
    app_env: str = Field(default=_env("APP_ENV", "development"))
    debug: bool = Field(default=_env("DEBUG", "true").lower() == "true")
    host: str = Field(default=_env("HOST", "127.0.0.1"))
    port: int = Field(default=int(_env("PORT", "8000")))
    api_prefix: str = Field(default=_env("API_PREFIX", ""))
    log_level: str = Field(default=_env("LOG_LEVEL", "INFO"))
    DASHSCOPE_API_KEY: str = Field(default=_env("DASHSCOPE_API_KEY", ""))
    DASHSCOPE_BASE_URL: str = Field(
        default=_env(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    QWEN_MODEL: str = Field(default=_env("QWEN_MODEL", "qwen3.5-plus"))
    LLM_TIMEOUT: int = Field(default=int(_env("LLM_TIMEOUT", "30")))
    EMBEDDING_MODEL_NAME: str = Field(default=_env("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"))
    EMBEDDING_DEVICE: str = Field(default=_env("EMBEDDING_DEVICE", ""))
    EMBEDDING_PRELOAD: bool = Field(default=_env("EMBEDDING_PRELOAD", "true").lower() == "true")
    EMBEDDING_LOCAL_FILES_ONLY: bool = Field(default=_env("EMBEDDING_LOCAL_FILES_ONLY", "true").lower() == "true")
    DOCUMENT_DB_PATH: str = Field(default=_env("DOCUMENT_DB_PATH", str(BASE_DIR / "data" / "documents.db")))
    UPLOAD_DIR: str = Field(default=_env("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    CHROMA_DIR: str = Field(default=_env("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
    LOG_DIR: str = Field(default=_env("LOG_DIR", str(BASE_DIR / "logs")))


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def get_document_db_path() -> Path:
    return Path(_env("DOCUMENT_DB_PATH", str(BASE_DIR / "data" / "documents.db")))


def get_upload_dir() -> Path:
    return Path(_env("UPLOAD_DIR", str(BASE_DIR / "uploads")))


def get_chroma_dir() -> Path:
    return Path(_env("CHROMA_DIR", str(BASE_DIR / "chroma_db")))


def get_log_dir() -> Path:
    return Path(_env("LOG_DIR", str(BASE_DIR / "logs")))

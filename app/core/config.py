"""
Application configuration using Pydantic settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # Application
    app_name: str = Field(default="EEG-Edge-Model", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # Directories
    model_dir: str = Field(default="./models", env="MODEL_DIR")
    data_dir: str = Field(default="./data", env="DATA_DIR")
    log_dir: str = Field(default="./logs", env="LOG_DIR")

    # Akida Settings
    akida_device_id: int = Field(default=0, env="AKIDA_DEVICE_ID")
    akida_num_devices: int = Field(default=1, env="AKIDA_NUM_DEVICES")

    # Training Settings
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    epochs: int = Field(default=100, env="EPOCHS")
    learning_rate: float = Field(default=0.001, env="LEARNING_RATE")

    # TVM Settings
    tvm_target: str = Field(default="llvm", env="TVM_TARGET")
    tvm_opt_level: int = Field(default=3, env="TVM_OPT_LEVEL")

    # Device Settings
    device: str = Field(default="auto", env="DEVICE")  # auto, cpu, cuda, mps

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

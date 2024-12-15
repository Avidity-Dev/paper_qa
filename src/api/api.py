from . import create_app
from .config import DevelopmentConfig

app = create_app(DevelopmentConfig)

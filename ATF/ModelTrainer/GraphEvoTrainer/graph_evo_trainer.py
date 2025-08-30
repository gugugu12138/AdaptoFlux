from ..model_trainer import ModelTrainer
import numpy as np
import logging
import random
import copy
from typing import Optional
import os
import json
from ...PathGenerator.path_generator import PathGenerator
from ...GraphManager.graph_processor import GraphProcessor

# 设置日志
logger = logging.getLogger(__name__)

class GraphEvoTrainer(ModelTrainer):
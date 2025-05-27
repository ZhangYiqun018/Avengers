from enum import Enum
from typing import List

from loguru import logger

from core.inference.aggregation_generator import AggregationGenerator
from core.inference.base_generator import BaseGenerator
from core.inference.direct_generator import DirectGenerator
from core.inference.fastslow_generator import FastSlowGenerator
from core.inference.modelswitch_generator import ModelSwitchGenerator
from core.inference.selfconsistency_generator import SelfConsistencyGenerator
from core.inference.moa_generator import MoAGenerator
from core.inference.slowfast_generator import SlowFastGenerator
from core.routing import RouterOutput


class GeneratorType(Enum):
    SELF_CONSISTENCY = "self_consistency"
    DIRECT = "direct"
    MODEL_SWITCH = "model_switch"
    FAST_SLOW = "fast_slow"
    AGGREGATION = "aggregation"
    MoA = "moa"
    SLOW_FAST = "slow_fast"

class GeneratorFactory:
    @staticmethod
    def create_generator(experts: RouterOutput, generator_config: dict):
        generator_type = GeneratorType(generator_config["type"])
        # get normal experts and thinking experts
        normal_experts = experts.normal_experts
        thinking_experts = experts.thinking_experts
        
        if generator_type == GeneratorType.SELF_CONSISTENCY:
            return SelfConsistencyGenerator(expert=normal_experts[0], generator_config=generator_config["self_consistency"])
        elif generator_type == GeneratorType.DIRECT:
            return DirectGenerator(expert=normal_experts[0], generator_config=generator_config["direct"])
        elif generator_type == GeneratorType.MODEL_SWITCH:
            return ModelSwitchGenerator(experts=normal_experts, generator_config=generator_config["model_switch"])
        elif generator_type == GeneratorType.AGGREGATION:
            return AggregationGenerator(experts=normal_experts, generator_config=generator_config["aggregation"])
        elif generator_type == GeneratorType.MoA:
            return MoAGenerator(experts=normal_experts, generator_config=generator_config["moa"])
        elif generator_type == GeneratorType.FAST_SLOW:
            assert len(thinking_experts) >= 1, "FastSlowGenerator requires at least **1** thinking expert"
            return FastSlowGenerator(
                fast_expert=normal_experts[0],
                slow_expert=thinking_experts[0],
                generator_config=generator_config["fast_slow"]
            )
        elif generator_type == GeneratorType.SLOW_FAST:
            return SlowFastGenerator(
                fast_expert=normal_experts[0],
                slow_expert=thinking_experts[0],
                generator_config=generator_config["slow_fast"]
            )
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
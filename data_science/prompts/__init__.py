"""Modular prompt system for the MECDM Insight Agent."""

from .prompt_builder import (
    Persona,
    PromptConfig,
    DatasetConfig,
    RelationsConfig,
    load_dataset_config,
    load_relations_config,
    build_root_instruction,
    build_global_instruction,
)

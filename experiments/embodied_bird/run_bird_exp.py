# experiments/embodied_bird/run_bird_exp.py
import numpy as np
import logging
from ATF.core.adaptoflux import AdaptoFlux
from ATF.ModelTrainer.CombinedTrainer.combined_trainer import CombinedTrainer
from .bird_env import run_bird_episode

# Set up logging
logging.basicConfig(level=logging.INFO)

# Dummy input shape (FlappyBird observation dim = 8)
dummy_input = np.zeros((1, 8), dtype=np.float32)

# Initialize AdaptoFlux (no labels needed!)
af = AdaptoFlux(
    values=dummy_input,
    labels=None,
    methods_path="experiments/embodied_bird/methods_bird.py"
)

# Optional: set collapse function (not used, but required by framework)
def collapse_first(values):
    return values[0] if len(values) > 0 else 0.0
af.set_custom_collapse(collapse_first)

# Define custom evaluators
def bird_loss(model, input_data, target):
    survival = run_bird_episode(model, action_interval=5, max_steps=5000)
    return float(-survival)  # minimize → maximize survival

def bird_acc(model, input_data, target):
    survival = run_bird_episode(model, action_interval=5, max_steps=5000)
    return float(survival / 5000.0)  # normalized [0, 1]

# === 关键：将 custom evaluators 放入 config 字典中 ===
lg_config = {
    "max_attempts": 5,
    "decision_threshold": 0.0,
    "verbose": False,
    # ↓ 这些会被传给 LayerGrowTrainer.__init__
    "custom_loss_evaluator": bird_loss,
    "custom_accuracy_evaluator": bird_acc,
}

ge_config = {
    "verbose": False,
    "init_mode": "fixed",
    "max_init_layers": 5,
    # ↓ 这些会被传给 GraphEvoTrainer.__init__
    "custom_loss_evaluator": bird_loss,
    "custom_accuracy_evaluator": bird_acc,
}

# Use existing CombinedTrainer with custom evaluators!
trainer = CombinedTrainer(
    adaptoflux_instance=af,
    layer_grow_config=lg_config,      # ← 包含 custom evaluators
    graph_evo_config=ge_config,       # ← 包含 custom evaluators
    num_evolution_cycles=3,
    genetic_mode="disabled",
    save_dir="experiments/embodied_bird/results",
    verbose=True    
)



# Train! (input_data and target are dummy; not used)
results = trainer.train(
    input_data=dummy_input,
    target=np.array([0.0])  # or np.array([0]) to satisfy shape check
)

# Final evaluation
final_survival = run_bird_episode(trainer.adaptoflux, action_interval=5)
print(f"Final survival time: {final_survival} frames")
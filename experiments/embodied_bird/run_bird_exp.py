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

# Use existing CombinedTrainer with custom evaluators!
trainer = CombinedTrainer(
    adaptoflux_instance=af,
    custom_loss_evaluator=bird_loss,
    custom_accuracy_evaluator=bird_acc,
    use_pipeline=False,
    loss_fn='mse',          # ← still initialized, but not used
    task_type='regression', # ← dummy
    # Add your config here (genetic_config, etc.)
    num_evolution_cycles=5,
    genetic_config={
        "population_size": 10,
        "generations": 3,
        "subpool_size": 8,
        "data_fraction": 1.0,
        "elite_ratio": 0.3,
        "mutation_rate": 0.2,
    },
    save_dir="experiments/embodied_bird/results"
)

# Train! (input_data and target are dummy; not used)
results = trainer.train(
    input_data=dummy_input,
    target=None  # or np.array([0]) to satisfy shape check
)

# Final evaluation
final_survival = run_bird_episode(trainer.adaptoflux, action_interval=5)
print(f"Final survival time: {final_survival} frames")
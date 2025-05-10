import optuna 

from ezautoml.space.search_space import SearchSpace
from ezautoml.space.search_point import SearchPoint
from ezautoml.space.component import Component
from ezautoml.space.space import Integer, Categorical, Real

class OptunaParser:
    """Parse the search space and convert it into Optuna-compatible trials."""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def parse_hyperparameters(self, model_config: Component, trial: optuna.Trial):
        """Parse the hyperparameters for a given model configuration."""
        model_params = {}
        for hp in model_config.hyperparams:
            name = hp.name
            if isinstance(hp, Integer):
                model_params[name] = trial.suggest_int(name, hp.low, hp.high)
            elif isinstance(hp, Real):
                model_params[name] = trial.suggest_loguniform(name, hp.low, hp.high)
            elif isinstance(hp, Categorical):
                model_params[name] = trial.suggest_categorical(name, hp.categories)
        return model_params

    def convert_to_search_point(self, trial: optuna.Trial) -> SearchPoint:
        """Convert an Optuna trial to a SearchPoint."""
        
        # Select model configuration based on trial's suggested model index
        model_config = self.search_space.models[trial.suggest_int("model", 0, len(self.search_space.models) - 1)]
        
        # Parse the hyperparameters for the model
        model_params = self.parse_hyperparameters(model_config, trial)

        # Instead of instantiating the model here, just store the model configuration
        # Create a SearchPoint with the model configuration and the hyperparameters
        config = SearchPoint(
            model=model_config,  # Pass the model component, not the instantiated model
            model_params=model_params,  # Pass the hyperparameters for the model
            data_processors=[],  # Data processors (empty or populate as needed)
            feature_processors=[],  # Feature processors (empty or populate as needed)
            data_params_list=[],  # Data parameters
            feature_params_list=[],  # Feature parameters
        )

        return config
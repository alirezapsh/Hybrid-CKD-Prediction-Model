import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Import other optimizers for comparison
try:
    from models.metaheuristic_optimizers import (
        GreyWolfOptimizer, ParticleSwarmOptimizer,
        GeneticAlgorithm, WhaleOptimizationAlgorithm
    )
    OTHER_OPTIMIZERS_AVAILABLE = True
except ImportError:
    OTHER_OPTIMIZERS_AVAILABLE = False


class EquilibriumOptimizer:
    """
    Equilibrium Optimizer (EO) for hyperparameter optimization.
    
    Based on the physics-inspired mass balance equation optimization algorithm.
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        bounds: Dict[str, Tuple[float, float]] = None,
        a1: float = 2.0,
        a2: float = 1.0,
        GP: float = 0.5,
        random_state: Optional[int] = None
    ):
        """
        Initialize Equilibrium Optimizer.
        
        Args:
            n_particles: Number of particles (candidate solutions)
            n_iterations: Number of iterations
            bounds: Dictionary of parameter bounds {param_name: (min, max)}
            a1: Exploration constant
            a2: Exploitation constant
            GP: Generation probability
            random_state: Random seed for reproducibility
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds if bounds is not None else {}
        self.a1 = a1
        self.a2 = a2
        self.GP = GP
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.best_fitness = float('inf')
        self.best_solution = None
        self.fitness_history = []
        self.convergence_history = []
    
    def _initialize_particles(self, n_params: int) -> np.ndarray:
        """
        Initialize particle positions (candidate solutions).
        
        Args:
            n_params: Number of parameters to optimize
            
        Returns:
            Initial particle positions (n_particles, n_params)
        """
        particles = np.zeros((self.n_particles, n_params))
        
        for i in range(self.n_particles):
            for j, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
                particles[i, j] = np.random.uniform(min_val, max_val)
        
        return particles
    
    def _clip_to_bounds(self, particles: np.ndarray) -> np.ndarray:
        """
        Clip particle positions to bounds.
        
        Args:
            particles: Particle positions (n_particles, n_params)
            
        Returns:
            Clipped particle positions
        """
        clipped = particles.copy()
        
        for j, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
            clipped[:, j] = np.clip(clipped[:, j], min_val, max_val)
        
        return clipped
    
    def _calculate_equilibrium_pool(self, particles: np.ndarray, fitness: np.ndarray) -> List[np.ndarray]:
        """
        Calculate equilibrium pool (top 4 candidates + average).
        
        Args:
            particles: Particle positions (n_particles, n_params)
            fitness: Fitness values (n_particles,)
            
        Returns:
            List of equilibrium candidates
        """
        # Sort by fitness
        sorted_indices = np.argsort(fitness)
        
        # Top 4 candidates
        eq_pool = [
            particles[sorted_indices[0]],  # Best
            particles[sorted_indices[1]],  # 2nd best
            particles[sorted_indices[2]],  # 3rd best
            particles[sorted_indices[3]],  # 4th best
            np.mean(particles[sorted_indices[:4]], axis=0)  # Average of top 4
        ]
        
        return eq_pool
    
    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Run optimization.
        
        Args:
            fitness_function: Function that takes a parameter vector and returns fitness
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_solution, best_fitness, optimization_history)
        """
        n_params = len(self.bounds)
        
        # Initialize particles
        particles = self._initialize_particles(n_params)
        
        # Evaluate initial fitness
        fitness = np.array([fitness_function(particles[i]) for i in range(self.n_particles)])
        
        # Find initial best
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = particles[best_idx].copy()
        
        if verbose:
            print("=" * 80)
            print("EQUILIBRIUM OPTIMIZER")
            print("=" * 80)
            print(f"Particles: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Parameters: {list(self.bounds.keys())}")
            print("=" * 80)
            print(f"\nInitial Best Fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Calculate equilibrium pool
            eq_pool = self._calculate_equilibrium_pool(particles, fitness)
            
            # Update each particle
            for i in range(self.n_particles):
                # Select random equilibrium candidate
                eq_idx = np.random.randint(0, len(eq_pool))
                Ceq = eq_pool[eq_idx]
                
                # Calculate exponential term
                t = (1 - iteration / self.n_iterations) ** (self.a2 * iteration / self.n_iterations)
                
                # Generate random vector
                r = np.random.rand(n_params)
                
                # Calculate generation rate
                GP_vector = np.zeros(n_params)
                for j in range(n_params):
                    if r[j] >= self.GP:
                        GP_vector[j] = 0
                    else:
                        GP_vector[j] = np.random.rand()
                
                # Update particle position
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # Calculate F (exponential term)
                F = self.a1 * np.sign(r - 0.5) * (np.exp(-self.a2 * t) - 1)
                
                # Calculate generation rate term
                GCP = 0.5 * r1 * GP_vector * (Ceq - self.a2 * particles[i])
                
                # Update particle
                particles[i] = Ceq + (particles[i] - Ceq) * F + GCP / (self.a2 * t) * (1 - F)
            
            # Clip to bounds
            particles = self._clip_to_bounds(particles)
            
            # Evaluate fitness
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.n_iterations}: Evaluating {self.n_particles} particles...")
            fitness = []
            for i in range(self.n_particles):
                if verbose and (i + 1) % 5 == 0:
                    print(f"  Evaluating particle {i + 1}/{self.n_particles}...")
                try:
                    fit = fitness_function(particles[i])
                    fitness.append(fit)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Particle {i + 1} evaluation failed: {e}")
                    fitness.append(1e6)  # High penalty for failed evaluation
            
            fitness = np.array(fitness)
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = particles[best_idx].copy()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose:
                print(f"  Iteration {iteration + 1}/{self.n_iterations} complete: "
                      f"Best Fitness = {self.best_fitness:.6f}, "
                      f"Mean Fitness = {np.mean(fitness):.6f}")
        
        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE!")
            print("=" * 80)
            print(f"Best Fitness: {self.best_fitness:.6f}")
            print("\nBest Parameters:")
            for j, param_name in enumerate(self.bounds.keys()):
                print(f"  {param_name}: {self.best_solution[j]:.6f}")
            print("=" * 80)
        
        history = {
            'fitness_history': self.fitness_history,
            'convergence_history': self.convergence_history,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness
        }
        
        return self.best_solution, self.best_fitness, history


class HyperparameterOptimizer:
    """
    High-level interface for optimizing hyperparameters of SSCL and Transformer models.
    
    Supports multiple optimization algorithms:
    - 'eo': Equilibrium Optimizer (default)
    - 'gwo': Grey Wolf Optimizer
    - 'pso': Particle Swarm Optimization
    - 'ga': Genetic Algorithm
    - 'woa': Whale Optimization Algorithm
    """
    
    def __init__(
        self,
        model_type: str = 'transformer',  # 'sscl', 'transformer', or 'both'
        n_particles: int = 20,
        n_iterations: int = 30,
        algorithm: str = 'eo',  # 'eo', 'gwo', 'pso', 'ga', 'woa'
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            model_type: Type of model to optimize ('sscl', 'transformer', or 'both')
            n_particles: Number of particles for EO
            n_iterations: Number of iterations for EO
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.model_type = model_type
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate algorithm
        valid_algorithms = ['eo', 'gwo', 'pso', 'ga', 'woa']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}")
        
        if self.algorithm != 'eo' and not OTHER_OPTIMIZERS_AVAILABLE:
            raise ImportError(f"Algorithm '{algorithm}' requires metaheuristic_optimizers module. "
                            "Please ensure models/metaheuristic_optimizers.py exists.")
        
        # Define parameter bounds based on model type
        if model_type == 'sscl':
            self.bounds = {
                'learning_rate': (1e-5, 1e-2),
                'encoder_dropout': (0.0, 0.5),
                'projection_dropout': (0.0, 0.3)
            }
        elif model_type == 'transformer':
            self.bounds = {
                'learning_rate': (1e-5, 1e-2),
                'n_layers': (2, 8),  # Layer count
                'dropout': (0.05, 0.5)  # Dropout rate (minimum 5% to ensure regularization)
            }
        elif model_type == 'both':
            # Optimize both models
            self.bounds = {
                'sscl_learning_rate': (1e-5, 1e-2),
                'sscl_encoder_dropout': (0.0, 0.5),
                'transformer_learning_rate': (1e-5, 1e-2),
                'transformer_n_layers': (2, 8),
                'transformer_dropout': (0.0, 0.5)
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _create_fitness_function(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim: int,
        model_type: str
    ) -> Callable:
        """
        Create fitness function for optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_dim: Input dimension
            model_type: Model type
            
        Returns:
            Fitness function
        """
        def fitness_function(params: np.ndarray) -> float:
            """
            Fitness function: negative validation accuracy (to minimize).
            
            Args:
                params: Parameter vector
                
            Returns:
                Negative validation accuracy (to minimize)
            """
            try:
                if model_type == 'sscl':
                    # Extract parameters
                    learning_rate = params[0]
                    encoder_dropout = params[1]
                    projection_dropout = params[2]
                    
                    # Train SSCL model with these parameters
                    from models.sscl import train_sscl
                    
                    model, trainer, history = train_sscl(
                        X_train=X_train,
                        X_val=X_val,
                        input_dim=input_dim,
                        encoder_dropout=encoder_dropout,
                        projection_dropout=projection_dropout,
                        learning_rate=learning_rate,
                        epochs=20,  # Reduced for faster optimization
                        verbose=False
                    )
                    
                    # Use validation loss as fitness (lower is better)
                    # Note: For SSCL, loss is already in correct format (not percentage)
                    if len(history['val_loss']) > 0:
                        fitness = history['val_loss'][-1]
                    else:
                        fitness = history['train_loss'][-1]
                    
                    return fitness
                
                elif model_type == 'transformer':
                    # Extract parameters
                    learning_rate = params[0]
                    n_layers = int(np.round(params[1]))  # Convert to integer
                    dropout = params[2]
                    
                    # Train Transformer model with these parameters
                    from models.transformer import train_transformer
                    
                    model, trainer, history = train_transformer(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        input_dim=input_dim,
                        d_model=128,  # Fixed for compatibility
                        n_heads=8,  # Fixed for compatibility
                        learning_rate=learning_rate,
                        n_layers=n_layers,
                        dropout=dropout,
                        batch_size=32,  # Fixed batch size (not optimized)
                        epochs=30,  # Reduced for faster optimization
                        verbose=False
                    )
                    
                    # Use negative validation accuracy as fitness (to minimize)
                    # Note: accuracy is stored as percentage (0-100), so normalize to 0-1
                    if len(history['val_accuracy']) > 0:
                        acc = history['val_accuracy'][-1] / 100.0  # Normalize from percentage to fraction
                        fitness = -acc  # Negative because we minimize
                    else:
                        acc = history['train_accuracy'][-1] / 100.0  # Normalize from percentage to fraction
                        fitness = -acc
                    
                    return fitness
                
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
            
            except Exception as e:
                # Return high fitness for failed evaluations
                if self.verbose:
                    print(f"  [ERROR] Evaluation failed: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                return 1e6
        
        return fitness_function
    
    def optimize_sscl(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim: Optional[int] = None
    ) -> Dict:
        """
        Optimize SSCL hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels (not used for SSCL, but kept for interface consistency)
            X_val: Validation features
            y_val: Validation labels (not used for SSCL)
            input_dim: Input dimension
            
        Returns:
            Dictionary with optimized parameters and history
        """
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series):
            X_val = X_val.values
        
        if input_dim is None:
            input_dim = X_train.shape[1]
        
        # Create fitness function
        fitness_func = self._create_fitness_function(
            X_train, y_train, X_val, y_val, input_dim, 'sscl'
        )
        
        # Create EO optimizer
        bounds = {
            'learning_rate': self.bounds['learning_rate'],
            'encoder_dropout': self.bounds['encoder_dropout'],
            'projection_dropout': self.bounds['projection_dropout']
        }
        
        eo = EquilibriumOptimizer(
            n_particles=self.n_particles,
            n_iterations=self.n_iterations,
            bounds=bounds,
            random_state=self.random_state
        )
        
        # Run optimization
        best_params, best_fitness, history = eo.optimize(
            fitness_func,
            verbose=self.verbose
        )
        
        # Convert to parameter dictionary
        param_dict = {
            'learning_rate': best_params[0],
            'encoder_dropout': best_params[1],
            'projection_dropout': best_params[2]
        }
        
        return {
            'parameters': param_dict,
            'fitness': best_fitness,
            'history': history
        }
    
    def optimize_transformer(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim: Optional[int] = None
    ) -> Dict:
        """
        Optimize Transformer hyperparameters.
        
        Args:
            X_train: Training features (SSCL features)
            y_train: Training labels
            X_val: Validation features (SSCL features)
            y_val: Validation labels
            input_dim: Input dimension (SSCL output dimension)
            
        Returns:
            Dictionary with optimized parameters and history
        """
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
            X_train = X_train.values
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series):
            X_val = X_val.values
        if isinstance(y_val, pd.Series) or isinstance(y_val, pd.DataFrame):
            y_val = y_val.values
        
        if input_dim is None:
            input_dim = X_train.shape[1]
        
        # Create fitness function
        fitness_func = self._create_fitness_function(
            X_train, y_train, X_val, y_val, input_dim, 'transformer'
        )
        
        # Create optimizer based on selected algorithm
        bounds = {
            'learning_rate': self.bounds['learning_rate'],
            'n_layers': self.bounds['n_layers'],
            'dropout': self.bounds['dropout']
        }
        
        if self.algorithm == 'eo':
            optimizer = EquilibriumOptimizer(
                n_particles=self.n_particles,
                n_iterations=self.n_iterations,
                bounds=bounds,
                random_state=self.random_state
            )
        elif self.algorithm == 'gwo':
            optimizer = GreyWolfOptimizer(
                n_particles=self.n_particles,
                n_iterations=self.n_iterations,
                bounds=bounds,
                random_state=self.random_state
            )
        elif self.algorithm == 'pso':
            optimizer = ParticleSwarmOptimizer(
                n_particles=self.n_particles,
                n_iterations=self.n_iterations,
                bounds=bounds,
                random_state=self.random_state
            )
        elif self.algorithm == 'ga':
            optimizer = GeneticAlgorithm(
                n_particles=self.n_particles,
                n_iterations=self.n_iterations,
                bounds=bounds,
                random_state=self.random_state
            )
        elif self.algorithm == 'woa':
            optimizer = WhaleOptimizationAlgorithm(
                n_particles=self.n_particles,
                n_iterations=self.n_iterations,
                bounds=bounds,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Run optimization
        best_params, best_fitness, history = optimizer.optimize(
            fitness_func,
            verbose=self.verbose
        )
        
        # Convert to parameter dictionary
        param_dict = {
            'learning_rate': best_params[0],
            'n_layers': int(np.round(best_params[1])),  # Convert to integer
            'dropout': best_params[2]
        }
        
        return {
            'parameters': param_dict,
            'fitness': best_fitness,
            'history': history
        }


if __name__ == "__main__":
    print("Equilibrium Optimizer Module for Hyperparameter Tuning")
    print("This module implements EO algorithm for optimizing:")
    print("  - Learning Rate")
    print("  - Layer Count")
    print("  - Dropout Rate")

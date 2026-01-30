import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import random
warnings.filterwarnings('ignore')


class BaseOptimizer:
    """Base class for all metaheuristic optimizers."""
    
    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        bounds: Dict[str, Tuple[float, float]] = None,
        random_state: Optional[int] = None
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds if bounds is None else bounds
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        
        self.best_fitness = float('inf')
        self.best_solution = None
        self.fitness_history = []
        self.convergence_history = []
    
    def _initialize_particles(self, n_params: int) -> np.ndarray:
        """Initialize particle positions."""
        particles = np.zeros((self.n_particles, n_params))
        for i in range(self.n_particles):
            for j, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
                particles[i, j] = np.random.uniform(min_val, max_val)
        return particles
    
    def _clip_to_bounds(self, particles: np.ndarray) -> np.ndarray:
        """Clip particles to bounds."""
        clipped = particles.copy()
        for j, (param_name, (min_val, max_val)) in enumerate(self.bounds.items()):
            clipped[:, j] = np.clip(clipped[:, j], min_val, max_val)
        return clipped
    
    def optimize(self, fitness_function: Callable[[np.ndarray], float], verbose: bool = True) -> Tuple[np.ndarray, float, Dict]:
        """Optimize - to be implemented by subclasses."""
        raise NotImplementedError


class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) for hyperparameter optimization.
    
    Based on the social hierarchy and hunting behavior of grey wolves.
    """
    
    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run GWO optimization."""
        n_params = len(self.bounds)
        
        # Initialize wolves (particles)
        wolves = self._initialize_particles(n_params)
        
        # Evaluate initial fitness
        if verbose:
            print("Evaluating initial population...")
        fitness = []
        for i in range(self.n_particles):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Evaluating wolf {i + 1}/{self.n_particles}...")
            try:
                fit = fitness_function(wolves[i])
                fitness.append(fit)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Wolf {i + 1} evaluation failed: {e}")
                fitness.append(1e6)
        fitness = np.array(fitness)
        
        # Find initial best (alpha, beta, delta)
        sorted_indices = np.argsort(fitness)
        alpha_pos = wolves[sorted_indices[0]].copy()
        beta_pos = wolves[sorted_indices[1]].copy()
        delta_pos = wolves[sorted_indices[2]].copy()
        
        alpha_fitness = fitness[sorted_indices[0]]
        beta_fitness = fitness[sorted_indices[1]]
        delta_fitness = fitness[sorted_indices[2]]
        
        self.best_fitness = alpha_fitness
        self.best_solution = alpha_pos.copy()
        
        if verbose:
            print("=" * 80)
            print("GREY WOLF OPTIMIZER (GWO)")
            print("=" * 80)
            print(f"Wolves: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Parameters: {list(self.bounds.keys())}")
            print("=" * 80)
            print(f"\nInitial Best Fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Linearly decrease a from 2 to 0
            a = 2 - iteration * (2 / self.n_iterations)
            
            for i in range(self.n_particles):
                # Update position based on alpha, beta, delta
                for j in range(n_params):
                    # Calculate coefficients for alpha, beta, delta
                    r1, r2 = np.random.rand(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    # Update position (average of alpha, beta, delta)
                    wolves[i, j] = (X1 + X2 + X3) / 3.0
            
            # Clip to bounds
            wolves = self._clip_to_bounds(wolves)
            
            # Evaluate fitness
            if verbose:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: Evaluating {self.n_particles} wolves...")
            fitness = []
            for i in range(self.n_particles):
                if verbose and (i + 1) % 3 == 0:  # Print every 3 wolves
                    print(f"  Evaluating wolf {i + 1}/{self.n_particles}...")
                try:
                    fit = fitness_function(wolves[i])
                    fitness.append(fit)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Wolf {i + 1} evaluation failed: {e}")
                    fitness.append(1e6)  # High penalty for failed evaluation
            fitness = np.array(fitness)
            
            # Update alpha, beta, delta
            sorted_indices = np.argsort(fitness)
            if fitness[sorted_indices[0]] < alpha_fitness:
                alpha_pos = wolves[sorted_indices[0]].copy()
                alpha_fitness = fitness[sorted_indices[0]]
            if fitness[sorted_indices[1]] < beta_fitness:
                beta_pos = wolves[sorted_indices[1]].copy()
                beta_fitness = fitness[sorted_indices[1]]
            if fitness[sorted_indices[2]] < delta_fitness:
                delta_pos = wolves[sorted_indices[2]].copy()
                delta_fitness = fitness[sorted_indices[2]]
            
            # Update best
            if alpha_fitness < self.best_fitness:
                self.best_fitness = alpha_fitness
                self.best_solution = alpha_pos.copy()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
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


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) for hyperparameter optimization.
    
    Based on the social behavior of bird flocking or fish schooling.
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        bounds: Dict[str, Tuple[float, float]] = None,
        w: float = 0.7,  # Inertia weight
        c1: float = 1.5,  # Cognitive coefficient
        c2: float = 1.5,  # Social coefficient
        random_state: Optional[int] = None
    ):
        super().__init__(n_particles, n_iterations, bounds, random_state)
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run PSO optimization."""
        n_params = len(self.bounds)
        
        # Initialize particles
        particles = self._initialize_particles(n_params)
        velocities = np.zeros_like(particles)
        
        # Initialize personal best
        if verbose:
            print("Evaluating initial population...")
        pbest = particles.copy()
        pbest_fitness = []
        for i in range(self.n_particles):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Evaluating particle {i + 1}/{self.n_particles}...")
            try:
                fit = fitness_function(particles[i])
                pbest_fitness.append(fit)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Particle {i + 1} evaluation failed: {e}")
                pbest_fitness.append(1e6)
        pbest_fitness = np.array(pbest_fitness)
        
        # Initialize global best
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        self.best_fitness = gbest_fitness
        self.best_solution = gbest.copy()
        
        if verbose:
            print("=" * 80)
            print("PARTICLE SWARM OPTIMIZATION (PSO)")
            print("=" * 80)
            print(f"Particles: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Parameters: {list(self.bounds.keys())}")
            print("=" * 80)
            print(f"\nInitial Best Fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (pbest[i] - particles[i]) +
                                self.c2 * r2 * (gbest - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
            
            # Clip to bounds
            particles = self._clip_to_bounds(particles)
            
            # Evaluate fitness
            if verbose:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: Evaluating {self.n_particles} particles...")
            fitness = []
            for i in range(self.n_particles):
                if verbose and (i + 1) % 3 == 0:  # Print every 3 particles
                    print(f"  Evaluating particle {i + 1}/{self.n_particles}...")
                try:
                    fit = fitness_function(particles[i])
                    fitness.append(fit)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Particle {i + 1} evaluation failed: {e}")
                    fitness.append(1e6)  # High penalty for failed evaluation
            fitness = np.array(fitness)
            
            # Update personal best
            improved = fitness < pbest_fitness
            pbest[improved] = particles[improved].copy()
            pbest_fitness[improved] = fitness[improved]
            
            # Update global best
            best_idx = np.argmin(pbest_fitness)
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest = pbest[best_idx].copy()
                gbest_fitness = pbest_fitness[best_idx]
            
            # Update best
            if gbest_fitness < self.best_fitness:
                self.best_fitness = gbest_fitness
                self.best_solution = gbest.copy()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
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


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm (GA) for hyperparameter optimization.
    
    Based on natural selection and genetic operations.
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        bounds: Dict[str, Tuple[float, float]] = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        selection_rate: float = 0.5,
        random_state: Optional[int] = None
    ):
        super().__init__(n_particles, n_iterations, bounds, random_state)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = individual.copy()
        for j in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                param_name = list(self.bounds.keys())[j]
                min_val, max_val = self.bounds[param_name]
                std = (max_val - min_val) * 0.1
                mutated[j] += np.random.normal(0, std)
        return mutated
    
    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run GA optimization."""
        n_params = len(self.bounds)
        
        # Initialize population
        population = self._initialize_particles(n_params)
        
        # Evaluate initial fitness
        if verbose:
            print("Evaluating initial population...")
        fitness = []
        for i in range(self.n_particles):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Evaluating individual {i + 1}/{self.n_particles}...")
            try:
                fit = fitness_function(population[i])
                fitness.append(fit)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Individual {i + 1} evaluation failed: {e}")
                fitness.append(1e6)
        fitness = np.array(fitness)
        
        # Find initial best
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = population[best_idx].copy()
        
        if verbose:
            print("=" * 80)
            print("GENETIC ALGORITHM (GA)")
            print("=" * 80)
            print(f"Population: {self.n_particles}")
            print(f"Generations: {self.n_iterations}")
            print(f"Parameters: {list(self.bounds.keys())}")
            print("=" * 80)
            print(f"\nInitial Best Fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for generation in range(self.n_iterations):
            if verbose:
                print(f"Generation {generation + 1}/{self.n_iterations}: Creating new population...")
            # Selection (tournament selection)
            n_selected = int(self.n_particles * self.selection_rate)
            selected_indices = []
            for _ in range(n_selected):
                tournament_size = 3
                tournament = np.random.choice(self.n_particles, tournament_size, replace=False)
                winner = tournament[np.argmin(fitness[tournament])]
                selected_indices.append(winner)
            
            # Create new population
            new_population = []
            new_fitness = []
            
            # Elitism: keep best individual
            best_idx = np.argmin(fitness)
            new_population.append(population[best_idx].copy())
            new_fitness.append(fitness[best_idx])
            
            # Generate offspring
            offspring_count = 0
            while len(new_population) < self.n_particles:
                # Select parents
                parent1_idx = selected_indices[np.random.randint(len(selected_indices))]
                parent2_idx = selected_indices[np.random.randint(len(selected_indices))]
                
                # Crossover
                child1, child2 = self._crossover(
                    population[parent1_idx],
                    population[parent2_idx]
                )
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Clip to bounds
                child1 = self._clip_to_bounds(child1.reshape(1, -1))[0]
                child2 = self._clip_to_bounds(child2.reshape(1, -1))[0]
                
                # Evaluate
                if verbose and offspring_count % 5 == 0:
                    print(f"  Evaluating offspring {offspring_count + 1}...")
                try:
                    fitness1 = fitness_function(child1)
                    fitness2 = fitness_function(child2)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Offspring evaluation failed: {e}")
                    fitness1 = 1e6
                    fitness2 = 1e6
                offspring_count += 1
                
                new_population.append(child1)
                new_fitness.append(fitness1)
                
                if len(new_population) < self.n_particles:
                    new_population.append(child2)
                    new_fitness.append(fitness2)
            
            # Update population
            population = np.array(new_population[:self.n_particles])
            fitness = np.array(new_fitness[:self.n_particles])
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.convergence_history.append({
                'iteration': generation + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose and (generation + 1) % max(1, self.n_iterations // 10) == 0:
                print(f"Generation {generation + 1}/{self.n_iterations}: "
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


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    Whale Optimization Algorithm (WOA) for hyperparameter optimization.
    
    Based on the bubble-net hunting strategy of humpback whales.
    """
    
    def optimize(
        self,
        fitness_function: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float, Dict]:
        """Run WOA optimization."""
        n_params = len(self.bounds)
        
        # Initialize whales
        whales = self._initialize_particles(n_params)
        
        # Evaluate initial fitness
        if verbose:
            print("Evaluating initial population...")
        fitness = []
        for i in range(self.n_particles):
            if verbose and (i + 1) % 5 == 0:
                print(f"  Evaluating whale {i + 1}/{self.n_particles}...")
            try:
                fit = fitness_function(whales[i])
                fitness.append(fit)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Whale {i + 1} evaluation failed: {e}")
                fitness.append(1e6)
        fitness = np.array(fitness)
        
        # Find initial best (leader)
        leader_idx = np.argmin(fitness)
        leader_pos = whales[leader_idx].copy()
        leader_fitness = fitness[leader_idx]
        
        self.best_fitness = leader_fitness
        self.best_solution = leader_pos.copy()
        
        if verbose:
            print("=" * 80)
            print("WHALE OPTIMIZATION ALGORITHM (WOA)")
            print("=" * 80)
            print(f"Whales: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Parameters: {list(self.bounds.keys())}")
            print("=" * 80)
            print(f"\nInitial Best Fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Linearly decrease a from 2 to 0
            a = 2 - iteration * (2 / self.n_iterations)
            # Decrease a2 from -1 to -2
            a2 = -1 - iteration * (1 / self.n_iterations)
            
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * np.random.rand() + 1
                
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) >= 1:
                        # Exploration: search for prey
                        rand_leader_idx = np.random.randint(0, self.n_particles)
                        X_rand = whales[rand_leader_idx]
                        D_X_rand = abs(C * X_rand - whales[i])
                        whales[i] = X_rand - A * D_X_rand
                    else:
                        # Exploitation: encircling prey
                        D_Leader = abs(C * leader_pos - whales[i])
                        whales[i] = leader_pos - A * D_Leader
                else:
                    # Bubble-net attacking method
                    distance2Leader = abs(leader_pos - whales[i])
                    whales[i] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos
            
            # Clip to bounds
            whales = self._clip_to_bounds(whales)
            
            # Evaluate fitness
            if verbose:
                print(f"  Evaluating {self.n_particles} whales...")
            fitness = []
            for i in range(self.n_particles):
                if verbose and (i + 1) % 3 == 0:  # Print every 3 whales
                    print(f"    Evaluating whale {i + 1}/{self.n_particles}...")
                try:
                    fit = fitness_function(whales[i])
                    fitness.append(fit)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Whale {i + 1} evaluation failed: {e}")
                    fitness.append(1e6)  # High penalty for failed evaluation
            fitness = np.array(fitness)
            
            # Update leader
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < leader_fitness:
                leader_pos = whales[best_idx].copy()
                leader_fitness = fitness[best_idx]
            
            # Update best
            if leader_fitness < self.best_fitness:
                self.best_fitness = leader_fitness
                self.best_solution = leader_pos.copy()
            
            # Store history
            self.fitness_history.append(self.best_fitness)
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
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

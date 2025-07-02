import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import requests
import threading
import time
from flask import Flask, request, jsonify
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from utils.eval import mock_api

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/evaluate"
INTRINSICS_FILE = 'arm_intrinsics.json'
MAX_PROGRAM_LENGTH = 100 # Maximum number of instructions in the sorting function
REWARD_INVALID_PROGRAM = -100.0 # Penalty for a program that doesn't sort correctly
REWARD_TOO_LONG = -50.0 # Penalty for exceeding max length


def run_mock_api():
    # Running in debug mode is not recommended for production, but fine for this example.
    # The 'use_reloader=False' is important to prevent issues with threading.
    mock_api.run(port=5000, debug=False, use_reloader=False)

# --- Custom Gym Environment for ARM Assembly Generation ---
 class ARMSortingEnv(gym.Env):
    """
    A custom Gymnasium environment for generating an ARM sorting algorithm.

    The agent's goal is to select a sequence of ARM intrinsics that
    form a fast and correct sorting function.

    - **Observation**: A fixed-size array representing the instructions chosen so far.
    - **Action**: A discrete choice of which instruction to add next from a predefined list.
    - **Reward**: Based on the execution speed of the completed program, heavily
      penalized for incorrect or invalid programs.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, intrinsics_path, max_program_length=MAX_PROGRAM_LENGTH):
        super(ARMSortingEnv, self).__init__()

        self.max_program_length = max_program_length
        
        # Load the ARM intrinsics from the JSON file
        try:
            with open(intrinsics_path, 'r') as f:
                self.intrinsics = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{intrinsics_path}' was not found.")
            print("Please ensure 'arm_intrinsics.json' is in the same directory.")
            self.intrinsics = []

        # Add a special "RET" instruction to signify the end of the program
        self.instructions = [item['name'] for item in self.intrinsics]
        self.instructions.append("RET") # End-of-program action
        self.instruction_map = {i: name for i, name in enumerate(self.instructions)}

        # --- Define Action and Observation Spaces ---
        # The agent can choose any of the available instructions.
        self.action_space = spaces.Discrete(len(self.instructions))

        # The observation is the sequence of instructions chosen so far, represented
        # by their integer IDs. We use a fixed-size Box space and pad with -1.
        # The size is `max_program_length + 1` to hold all instructions.
        self.observation_space = spaces.Box(
            low=0, 
            high=len(self.instructions) - 1, 
            shape=(self.max_program_length,), 
            dtype=np.int32
        )

        # Environment state
        self.current_program_indices = []
        self.current_step = 0

    def _get_obs(self):
        """
        Pads the current program to the maximum length to create a fixed-size observation.
        """
        obs = np.zeros((self.max_program_length,), dtype=np.int32)
        if self.current_program_indices:
            obs[:len(self.current_program_indices)] = self.current_program_indices
        return obs

    def _get_info(self):
        """
        Returns diagnostic information about the current state.
        """
        return {
            "program_length": len(self.current_program_indices),
            "program": [self.instruction_map[i] for i in self.current_program_indices]
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        """
        super().reset(seed=seed)
        self.current_program_indices = []
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Executes one step in the environment.
        """
        self.current_program_indices.append(action)
        self.current_step += 1
        
        # Check for termination conditions
        is_ret_instruction = self.instruction_map[action] == "RET"
        is_too_long = self.current_step >= self.max_program_length

        terminated = is_ret_instruction or is_too_long
        truncated = is_too_long # Use truncated for time/length limits

        reward = 0.0
        if terminated:
            # Episode is over, now we evaluate the generated program
            program_str = self._assemble_program()
            
            try:
                # Send to the mock API for evaluation
                response = requests.post(API_URL, json={"assembly_code": program_str})
                response.raise_for_status()
                result = response.json()

                if not result.get("sorted"):
                    # Heavy penalty for incorrect sorting
                    reward = REWARD_INVALID_PROGRAM
                else:
                    # Reward is inversely proportional to execution time. Faster is better.
                    # Add a small epsilon to avoid division by zero.
                    execution_time = result.get("execution_time", 99999)
                    reward = 1000.0 / (execution_time + 1e-6)

            except requests.exceptions.RequestException as e:
                print(f"API call failed: {e}")
                reward = REWARD_INVALID_PROGRAM # Penalize if API call fails
            
            if truncated and not is_ret_instruction:
                reward = REWARD_TOO_LONG

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _assemble_program(self):
        """
        Converts the list of instruction indices into a formatted assembly string.
        """
        header = "sort_array:\n"
        footer = "\n"
        
        # Convert indices to instruction names
        instructions = [self.instruction_map[i] for i in self.current_program_indices]
        
        # Format with indentation
        body = "\n".join([f"    {inst}" for inst in instructions])
        
        return header + body + footer
        
    def render(self, mode='human'):
        """
        Prints the current state of the program.
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            program_str = self._assemble_program()
            print("--- Current Program ---")
            print(program_str)
            print("-----------------------")

if __name__ == '__main__':
    # --- 1. Start the Mock API in a separate thread ---
    print("Starting mock API server...")
    api_thread = threading.Thread(target=run_mock_api, daemon=True)
    api_thread.start()
    time.sleep(2) # Give the server a moment to start
    print("Mock API server started at " + API_URL)

    # --- 2. Create and check the custom environment ---
    print("\nInitializing ARM Sorting RL Environment...")
    env = ARMSortingEnv(intrinsics_path=INTRINSICS_FILE)
    
    # It's a good practice to check your custom environment
    # This will run a series of tests to ensure it follows the gymnasium API
    try:
        check_env(env)
        print("✅ Environment check passed!")
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        exit()

    # --- 3. Instantiate and train the RL Agent ---
    # We use Proximal Policy Optimization (PPO), a robust and popular algorithm.
    # "MlpPolicy" means it will use a Multi-Layer Perceptron (a standard neural network).
    print("\nTraining PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./arm_sorter_tensorboard/")
    
    # The agent will now learn for 20,000 steps.
    # In a real scenario, this would need millions of steps and significant tuning.
    model.learn(total_timesteps=20000)
    print("✅ Training complete.")

    # --- 4. Test the trained agent ---
    print("\n--- Testing trained agent ---")
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    print("\n--- Final Optimized Program ---")
    final_program = env._assemble_program()
    print(final_program)

    print("\n--- Simulated Performance of Final Program ---")
    try:
        response = requests.post(API_URL, json={"assembly_code": final_program})
        result = response.json()
        print(f"Correctly sorted: {result.get('sorted')}")
        print(f"Execution time (simulated): {result.get('execution_time'):.2f} ns")
    except requests.exceptions.RequestException as e:
        print(f"Could not evaluate final program: {e}")
    
    print("\nScript finished.")

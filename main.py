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
MAX_PROGRAM_LENGTH = 150  # Increased for realistic sorting implementations
REWARD_INVALID_PROGRAM = -100.0
REWARD_TOO_LONG = -50.0
REWARD_SYNTAX_ERROR = -80.0

# ARM Cortex-A72 specific configurations
NUM_REGISTERS = 16  # r0-r15 (including sp, lr, pc)
NUM_SIMD_REGISTERS = 32  # v0-v31 for NEON

def run_mock_api():
    mock_api.run(port=5000, debug=False, use_reloader=False)

class ARMInstruction:
    """Represents an ARM instruction with its operands"""
    def __init__(self, opcode, operands=None, operand_types=None):
        self.opcode = opcode
        self.operands = operands or []
        self.operand_types = operand_types or []
        
    def to_string(self):
        if not self.operands:
            return self.opcode
        return f"{self.opcode} {', '.join(map(str, self.operands))}"

class ARMSortingEnv(gym.Env):
    """
    Enhanced ARM assembly generation environment for sorting algorithms.
    
    Key improvements:
    - Proper instruction operands
    - Register allocation tracking
    - Memory access patterns
    - NEON SIMD instruction support for ARM Cortex-A72
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, intrinsics_path, max_program_length=MAX_PROGRAM_LENGTH):
        super(ARMSortingEnv, self).__init__()
        
        self.max_program_length = max_program_length
        
        # Define ARM instruction set with operand patterns
        self.instruction_templates = {
            # Memory operations
            "ldr": ["reg", "mem"],      # Load register from memory
            "str": ["reg", "mem"],      # Store register to memory
            "ldp": ["reg", "reg", "mem"],  # Load pair
            "stp": ["reg", "reg", "mem"],  # Store pair
            
            # Arithmetic
            "add": ["reg", "reg", "reg_or_imm"],
            "sub": ["reg", "reg", "reg_or_imm"],
            "mul": ["reg", "reg", "reg"],
            "cmp": ["reg", "reg_or_imm"],
            
            # Logic
            "and": ["reg", "reg", "reg_or_imm"],
            "orr": ["reg", "reg", "reg_or_imm"],
            "eor": ["reg", "reg", "reg_or_imm"],
            "mvn": ["reg", "reg"],
            
            # Shifts
            "lsl": ["reg", "reg", "imm"],
            "lsr": ["reg", "reg", "imm"],
            "asr": ["reg", "reg", "imm"],
            
            # Branches
            "b": ["label"],
            "beq": ["label"],
            "bne": ["label"],
            "blt": ["label"],
            "bgt": ["label"],
            "ble": ["label"],
            "bge": ["label"],
            
            # NEON SIMD operations (Cortex-A72 specific)
            "vld1.32": ["simd_reg", "mem"],  # Load vector
            "vst1.32": ["simd_reg", "mem"],  # Store vector
            "vadd.i32": ["simd_reg", "simd_reg", "simd_reg"],
            "vsub.i32": ["simd_reg", "simd_reg", "simd_reg"],
            "vcmp.s32": ["simd_reg", "simd_reg"],
            "vmax.s32": ["simd_reg", "simd_reg", "simd_reg"],
            "vmin.s32": ["simd_reg", "simd_reg", "simd_reg"],
            
            # Special
            "mov": ["reg", "reg_or_imm"],
            "ret": [],
            "nop": [],
        }
        
        # Define operand encodings
        self.operand_types = {
            "reg": list(range(16)),  # r0-r15
            "simd_reg": list(range(32)),  # v0-v31
            "imm": list(range(-128, 128)),  # Small immediates
            "label": list(range(10)),  # Up to 10 labels
            "mem": list(range(20)),  # Memory access patterns
        }
        
        # Calculate action space dimensions
        self.num_opcodes = len(self.instruction_templates)
        self.max_operands = max(len(ops) for ops in self.instruction_templates.values())
        
        # Action space: [opcode_idx, operand1, operand2, operand3]
        self.action_space = spaces.MultiDiscrete([
            self.num_opcodes,
            256,  # Operand 1 encoding space
            256,  # Operand 2 encoding space
            256,  # Operand 3 encoding space
        ])
        
        # Observation space includes:
        # - Current program (encoded)
        # - Register usage flags
        # - Memory access pattern
        # - Control flow state
        obs_dim = (
            self.max_program_length * 4 +  # Program encoding
            NUM_REGISTERS +  # Register usage
            NUM_SIMD_REGISTERS +  # SIMD register usage
            20  # Memory and control flow state
        )
        
        self.observation_space = spaces.Box(
            low=-1,
            high=255,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_program = []
        self.register_usage = np.zeros(NUM_REGISTERS)
        self.simd_register_usage = np.zeros(NUM_SIMD_REGISTERS)
        self.memory_accesses = []
        self.labels_defined = set()
        self.labels_used = set()
        self.current_step = 0
        
    def _encode_operand(self, operand_type, value):
        """Convert operand type and value to instruction encoding"""
        if operand_type == "reg":
            return f"r{value % 13}"  # r0-r12 (avoid sp, lr, pc for simplicity)
        elif operand_type == "simd_reg":
            return f"v{value % 32}"
        elif operand_type == "imm":
            return f"#{value}"
        elif operand_type == "label":
            return f".L{value}"
        elif operand_type == "mem":
            # Various addressing modes
            patterns = [
                f"[r{value % 13}]",
                f"[r{value % 13}, #{(value * 4) % 64}]",
                f"[r{value % 13}], #{(value * 4) % 64}",
                f"[r{value % 13}, r{(value + 1) % 13}]",
            ]
            return patterns[value % len(patterns)]
        elif operand_type == "reg_or_imm":
            if value < 128:
                return f"#{value}"
            else:
                return f"r{(value - 128) % 13}"
        return str(value)
    
    def _decode_action(self, action):
        """Convert action to instruction"""
        opcode_idx, op1, op2, op3 = action
        
        opcodes = list(self.instruction_templates.keys())
        if opcode_idx >= len(opcodes):
            return None
            
        opcode = opcodes[opcode_idx]
        operand_types = self.instruction_templates[opcode]
        
        operands = []
        operand_values = [op1, op2, op3]
        
        for i, op_type in enumerate(operand_types):
            if i < len(operand_values):
                operands.append(self._encode_operand(op_type, operand_values[i]))
                
        return ARMInstruction(opcode, operands, operand_types)
    
    def _get_obs(self):
        """Create observation vector"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Encode current program
        prog_offset = 0
        for i, inst in enumerate(self.current_program[:self.max_program_length]):
            if i >= self.max_program_length:
                break
            # Simple encoding: opcode + operands
            opcode_idx = list(self.instruction_templates.keys()).index(inst.opcode)
            obs[prog_offset] = opcode_idx
            prog_offset += 4
            
        # Register usage
        reg_offset = self.max_program_length * 4
        obs[reg_offset:reg_offset + NUM_REGISTERS] = self.register_usage
        
        # SIMD register usage
        simd_offset = reg_offset + NUM_REGISTERS
        obs[simd_offset:simd_offset + NUM_SIMD_REGISTERS] = self.simd_register_usage
        
        return obs
    
    def _update_register_usage(self, instruction):
        """Track register usage for better code generation"""
        if instruction.opcode in ["ldr", "mov", "add", "sub", "mul"]:
            # These write to first operand
            if instruction.operands and instruction.operands[0].startswith('r'):
                reg_num = int(instruction.operands[0][1:])
                if reg_num < NUM_REGISTERS:
                    self.register_usage[reg_num] = 1
                    
        # Track SIMD register usage
        if instruction.opcode.startswith('v'):
            for op in instruction.operands:
                if op.startswith('v'):
                    reg_num = int(op[1:])
                    if reg_num < NUM_SIMD_REGISTERS:
                        self.simd_register_usage[reg_num] = 1
    
    def _validate_instruction(self, instruction):
        """Check if instruction is valid in current context"""
        # Check branch targets
        if instruction.opcode in ["b", "beq", "bne", "blt", "bgt", "ble", "bge"]:
            if instruction.operands:
                label = instruction.operands[0]
                self.labels_used.add(label)
                
        # Check for function prologue/epilogue requirements
        if instruction.opcode == "ret" and len(self.current_program) < 3:
            return False  # Too early for return
            
        return True
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_program = []
        self.register_usage = np.zeros(NUM_REGISTERS)
        self.simd_register_usage = np.zeros(NUM_SIMD_REGISTERS)
        self.memory_accesses = []
        self.labels_defined = set()
        self.labels_used = set()
        self.current_step = 0
        
        # Add function prologue
        self.current_program.extend([
            ARMInstruction("stp", ["x29", "x30", "[sp, #-16]!"]),
            ARMInstruction("mov", ["x29", "sp"]),
        ])
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        instruction = self._decode_action(action)
        
        if instruction is None:
            # Invalid instruction
            return self._get_obs(), REWARD_SYNTAX_ERROR, True, False, self._get_info()
        
        # Validate and add instruction
        if self._validate_instruction(instruction):
            self.current_program.append(instruction)
            self._update_register_usage(instruction)
        else:
            return self._get_obs(), REWARD_SYNTAX_ERROR / 2, False, False, self._get_info()
        
        self.current_step += 1
        
        # Check termination
        is_ret = instruction.opcode == "ret"
        is_too_long = self.current_step >= self.max_program_length
        
        terminated = is_ret or is_too_long
        truncated = is_too_long
        
        reward = 0.0
        if terminated:
            # Add function epilogue if missing
            if not is_ret:
                self.current_program.extend([
                    ARMInstruction("ldp", ["x29", "x30", "[sp], #16"]),
                    ARMInstruction("ret"),
                ])
            
            # Evaluate the program
            program_str = self._assemble_program()
            
            try:
                response = requests.post(API_URL, json={
                    "assembly_code": program_str,
                    "architecture": "armv8-a",
                    "cpu": "cortex-a72",
                    "optimization_flags": ["-O2", "-march=armv8-a", "-mtune=cortex-a72"]
                })
                response.raise_for_status()
                result = response.json()
                
                if not result.get("sorted"):
                    reward = REWARD_INVALID_PROGRAM
                else:
                    # Consider multiple metrics
                    execution_time = result.get("execution_time", 99999)
                    cache_misses = result.get("cache_misses", 0)
                    branch_mispredicts = result.get("branch_mispredicts", 0)
                    
                    # Composite reward function
                    time_reward = 1000.0 / (execution_time + 1e-6)
                    cache_penalty = cache_misses * 0.1
                    branch_penalty = branch_mispredicts * 0.5
                    
                    # Bonus for using SIMD instructions efficiently
                    simd_bonus = np.sum(self.simd_register_usage) * 5.0
                    
                    reward = time_reward - cache_penalty - branch_penalty + simd_bonus
                    
            except Exception as e:
                print(f"Evaluation failed: {e}")
                reward = REWARD_INVALID_PROGRAM
                
            if truncated:
                reward = REWARD_TOO_LONG
                
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _assemble_program(self):
        """Generate properly formatted ARM assembly"""
        lines = [
            ".arch armv8-a",
            ".cpu cortex-a72",
            ".text",
            ".global sort_array",
            ".type sort_array, %function",
            "",
            "sort_array:",
            "    // x0: array pointer",
            "    // x1: array size",
        ]
        
        # Add the generated instructions
        for inst in self.current_program:
            lines.append(f"    {inst.to_string()}")
            
        # Add size directive
        lines.extend([
            "",
            ".size sort_array, .-sort_array",
        ])
        
        return "\n".join(lines)
    
    def _get_info(self):
        return {
            "program_length": len(self.current_program),
            "register_usage": np.sum(self.register_usage),
            "simd_usage": np.sum(self.simd_register_usage),
            "program": [inst.to_string() for inst in self.current_program],
        }
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print("Register usage:", np.where(self.register_usage)[0])
            print("SIMD usage:", np.where(self.simd_register_usage)[0])
            print("\n--- Current Program ---")
            print(self._assemble_program())
            print("-----------------------")

# Enhanced training script with pause/resume capability
import signal
import pickle
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

class PauseResumeCallback(BaseCallback):
    """Custom callback to handle pause/resume functionality"""
    def __init__(self, checkpoint_path="./training_state/", verbose=0):
        super(PauseResumeCallback, self).__init__(verbose)
        self.checkpoint_path = checkpoint_path
        self.pause_requested = False
        self.checkpoint_freq = 1000  # Save every 1000 steps
        
        # Setup signal handler for graceful pause (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_pause_signal)
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
    def _handle_pause_signal(self, signum, frame):
        print("\n⏸️  Pause requested. Saving checkpoint...")
        self.pause_requested = True
        
    def _on_step(self) -> bool:
        # Auto-save checkpoint periodically
        if self.n_calls % self.checkpoint_freq == 0:
            self._save_checkpoint()
            
        # Check if pause was requested
        if self.pause_requested:
            self._save_checkpoint()
            print("✅ Training paused. Resume with --resume flag")
            return False  # Stop training
            
        return True  # Continue training
    
    def _save_checkpoint(self):
        """Save complete training state"""
        checkpoint = {
            'timesteps': self.num_timesteps,
            'n_calls': self.n_calls,
            'best_model_path': getattr(self, 'best_model_path', None),
        }
        
        # Save model
        self.model.save(os.path.join(self.checkpoint_path, "model"))
        
        # Save training state
        with open(os.path.join(self.checkpoint_path, "training_state.pkl"), 'wb') as f:
            pickle.dump(checkpoint, f)
            
        if self.verbose > 0:
            print(f"💾 Checkpoint saved at timestep {self.num_timesteps}")

def load_training_state(checkpoint_path):
    """Load training state from checkpoint"""
    state_file = os.path.join(checkpoint_path, "training_state.pkl")
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            return pickle.load(f)
    return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ARM Assembly RL Optimizer')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-path', default='./training_state/', help='Checkpoint directory')
    parser.add_argument('--total-timesteps', type=int, default=100000, help='Total training timesteps')
    args = parser.parse_args()
    
    # Start API server
    print("Starting evaluation API...")
    api_thread = threading.Thread(target=run_mock_api, daemon=True)
    api_thread.start()
    time.sleep(2)
    
    # Create environment
    print("\nInitializing ARM assembly environment...")
    env = ARMSortingEnv(intrinsics_path=INTRINSICS_FILE)
    
    # Initialize or load model
    if args.resume:
        training_state = load_training_state(args.checkpoint_path)
        if training_state and os.path.exists(os.path.join(args.checkpoint_path, "model.zip")):
            print(f"📂 Resuming from checkpoint (timestep {training_state['timesteps']})")
            model = PPO.load(
                os.path.join(args.checkpoint_path, "model"),
                env=env,
                tensorboard_log="./arm_optimizer_logs/"
            )
            initial_timesteps = training_state['timesteps']
            remaining_timesteps = args.total_timesteps - initial_timesteps
            
            if remaining_timesteps <= 0:
                print("✅ Training already completed!")
                exit(0)
        else:
            print("❌ No checkpoint found. Starting fresh training.")
            args.resume = False
    
    if not args.resume:
        # Check environment
        try:
            check_env(env)
            print("✅ Environment check passed!")
        except Exception as e:
            print(f"❌ Environment check failed: {e}")
            
        # Create new model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./arm_optimizer_logs/"
        )
        initial_timesteps = 0
        remaining_timesteps = args.total_timesteps
    
    # Setup callbacks
    eval_env = ARMSortingEnv(intrinsics_path=INTRINSICS_FILE)
    
    callbacks = [
        PauseResumeCallback(checkpoint_path=args.checkpoint_path, verbose=1),
        EvalCallback(
            eval_env,
            best_model_save_path="./best_models/",
            log_path="./eval_logs/",
            eval_freq=5000,
            deterministic=True,
            render=False
        ),
        CheckpointCallback(
            save_freq=10000,
            save_path="./checkpoints/",
            name_prefix="arm_sort_model"
        )
    ]
    
    # Train
    print(f"\n🚀 Training agent ({remaining_timesteps} timesteps remaining)...")
    print("Press Ctrl+C to pause training\n")
    
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=False if args.resume else True,
            tb_log_name="PPO"
        )
        print("✅ Training complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Use --resume to continue.")
        exit(0)
    
    # Test the best model
    print("\n--- Testing best model ---")
    if os.path.exists("./best_models/best_model.zip"):
        best_model = PPO.load("./best_models/best_model")
    else:
        best_model = model
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    print("\n--- Final Optimized Program ---")
    final_program = env._assemble_program()
    print(final_program)
    
    # Detailed evaluation
    try:
        response = requests.post(API_URL, json={
            "assembly_code": final_program,
            "architecture": "armv8-a",
            "cpu": "cortex-a72",
            "test_cases": [
                {"size": 10, "pattern": "random"},
                {"size": 100, "pattern": "random"},
                {"size": 1000, "pattern": "random"},
                {"size": 10, "pattern": "sorted"},
                {"size": 100, "pattern": "reverse"},
            ]
        })
        result = response.json()
        
        print("\n--- Performance Metrics ---")
        print(f"Correctly sorted: {result.get('sorted')}")
        print(f"Average execution time: {result.get('avg_execution_time', 0):.2f} ns")
        print(f"Cache efficiency: {100 - result.get('cache_miss_rate', 0):.1f}%")
        print(f"Branch prediction accuracy: {100 - result.get('branch_miss_rate', 0):.1f}%")
        print(f"Instructions per cycle: {result.get('ipc', 0):.2f}")
        
    except Exception as e:
        print(f"Could not evaluate final program: {e}")
    
    print("\nOptimization complete!")
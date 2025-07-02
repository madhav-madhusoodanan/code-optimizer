import numpy as np
from flask import Flask, request, jsonify


# --- Mock API Server (Flask) ---
# This simulates the hardware testing environment. In a real scenario, this
# would be a service that compiles and runs the assembly on actual ARM hardware.
mock_api = Flask(__name__)

@mock_api.route('/evaluate', methods=['POST'])
def evaluate_assembly():
    """
    Mock endpoint to "evaluate" the speed of the generated assembly code.
    """
    data = request.get_json()
    if not data or 'assembly_code' not in data:
        return jsonify({"error": "Invalid request"}), 400

    code = data['assembly_code']
    
    # --- Mock Evaluation Logic ---
    # 1. Basic validation: Does the code seem plausible? (e.g., contains a return)
    # In a real system, this is where you would compile and run the code.
    is_valid = 'ret' in code.lower()

    if not is_valid:
        # Heavily penalize programs that are syntactically nonsense
        return jsonify({
            "sorted": False,
            "execution_time": 99999, 
            "error": "Program did not contain a return instruction."
        })

    # 2. Simulate execution time.
    # For this demo, we'll create a simple heuristic:
    # - Shorter programs are generally faster.
    # - Add some randomness to simulate real-world performance variance.
    num_lines = len(code.split('\n'))
    base_time = num_lines * 10 # 10ns per instruction (example)
    random_variance = np.random.uniform(0.9, 1.1)
    execution_time = base_time * random_variance

    # 3. Simulate correctness check.
    # A real evaluator would run the code on test arrays. Here, we'll just pretend.
    # Let's say there's a 95% chance the generated code is correct if valid.
    is_sorted_correctly = np.random.rand() > 0.05

    return jsonify({
        "sorted": is_sorted_correctly,
        "execution_time": execution_time,
        "error": None
    })

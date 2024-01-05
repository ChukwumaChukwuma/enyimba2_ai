import fire
from llama.llama import Llama
from typing import List

# Quantum-analogical prompt generator function
# Define the function to create a quantum-analogical prompt for the LLM
def generate_quantum_analogical_prompt(task_description):
    """
    Generates a quantum-analogical prompt for the LLM based on the given task.
    
    Parameters:
    task_description (str): A description of the task to be processed by the LLM.

    Returns:
    str: A prompt for the LLM that includes the task in the context of quantum states.
    """
    # Step 1: Define the initial state (task description)
    initial_state = task_description

    # Step 2: Craft an analogy of the quantum state
    quantum_state_analogy = """
    Imagine this task as a quantum state in a quantum computer. In a quantum system, 
    a state can be in multiple possibilities simultaneously, thanks to superposition. 
    Consider the various aspects and outcomes of this task as if they were quantum states in superposition. 
    Each aspect represents a different potential outcome or approach to the task.
    
    Now, generate all possible 'quantum states' of this task, considering the different dimensions, perspectives, 
    and outcomes that could exist in a superposed state. Think about the task not as a single path, but as a multitude 
    of possibilities, each representing a unique variation or approach.
    """

    # Step 3: Combine the task description with the quantum state analogy
    combined_prompt = f"Task: {initial_state}\n\n{quantum_state_analogy}"

    return combined_prompt

# The quantum_analogical_prompt is now ready to be used as input for the LLM


# Modified main function to return the output
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
) -> List[str]:
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    task = "Design a sustainable energy solution for urban areas."
    quantum_analogical_prompt = generate_quantum_analogical_prompt(task)

    prompts: List[str] = [quantum_analogical_prompt]

    results = generator.generate(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for prompt, result in zip(prompts, results):
        output.append((prompt, result['generation']))

    return output

if __name__ == "__main__":
    fire.Fire(main)
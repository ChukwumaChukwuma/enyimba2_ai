import fire
from llama.llama import Llama
from typing import List

# Assuming first_quibit.py has a function 'get_first_qubit_outputs' that returns the outputs
from quibit import generate_quantum_analogical_prompt 

def generate_superposition_prompt(outputs):
    """
    Generates a prompt asking to superpose all quantum states from the outputs.
    
    Parameters:
    outputs (List[str]): List of outputs from the first qubit.

    Returns:
    str: A prompt for the LLM to superpose all the quantum states.
    """
    combined_outputs = ' '.join(outputs)
    superposition_prompt = f"""
    Considering these various states: {combined_outputs}
    Now, create a superposition of all these states, analogous to quantum computing, 
    where each state can exist simultaneously, leading to a multitude of possibilities 
    and outcomes when observed.
    """
    return superposition_prompt

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

    first_qubit_outputs = generate_quantum_analogical_prompt()
    superposition_prompt = generate_superposition_prompt(first_qubit_outputs)

    prompts: List[str] = [superposition_prompt]

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

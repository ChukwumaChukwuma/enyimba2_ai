import fire
from llama.llama import Llama
from typing import List
from superposition import generate_superposition_prompt

def generate_quantum_entanglement_analogy(superposition_data, task_description):
    """
    Generates a quantum entanglement analogy prompt for any given task.

    Parameters:
    superposition_data (str): A description of the superposition of different states or ideas for the task.
    task_description (str): A description of the task to be processed.

    Returns:
    str: A prompt for the LLM to create an entangled state of ideas for the task.
    """
    entanglement_analogy = f"""
    Considering the superposition: {superposition_data}
    
    Task: {task_description}

    Create an entangled state where the exploration or resolution of one aspect of this task 
    immediately influences another aspect, irrespective of their apparent separation. 
    This analogy is akin to 'spooky action at a distance' in quantum entanglement. In this entangled state, 
    if the state or solution of one part is known or decided, the state or solution of its entangled partner 
    can be instantly determined or influenced. 

    This entanglement means that the different dimensions of the task are not independent but deeply interconnected, 
    reflecting a complex network of cause-and-effect relationships that operate beyond the confines of classical thinking. 
    Describe how these entangled aspects work together to form a comprehensive and efficient solution to the task.
    """
    return entanglement_analogy

# The entanglement_prompt is now ready to be used as input for the LLM

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

    # Example task and superposition data
    task = "Develop a new marketing strategy for a technology product."
    superposition_data = "Various marketing approaches including digital campaigns, influencer partnerships, and traditional media advertising."

    # Generate the entanglement prompt
    entanglement_prompt = generate_quantum_entanglement_analogy(superposition_data, task)

    prompts: List[str] = [entanglement_prompt]

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

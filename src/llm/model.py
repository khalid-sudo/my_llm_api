import time
import threading
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto"  # Automatically handle device placement
        )

    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        # Tokenize the input with attention_mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return PyTorch tensors
            padding=True,  # Pad the input to the max length
            truncation=True,  # Truncate if the input is too long
            max_length=max_length,  # Set the max length
            return_attention_mask=True  # Include the attention mask
        )
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # Function to simulate a progress bar
        def simulate_progress():
            with tqdm(total=max_length, desc="Generating response") as pbar:
                while not generation_done:
                    pbar.update(1)
                    time.sleep(1)  # Simulate progress every second
        
        # Start the progress bar thread
        generation_done = False
        progress_thread = threading.Thread(target=simulate_progress)
        progress_thread.start()
        
        # Generate the output
        start_time = time.time()
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Pass the attention mask
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id  # Use the pad_token_id
        )
        end_time = time.time()
        
        # Stop the progress bar thread
        generation_done = True
        progress_thread.join()
        
        # Calculate the time taken
        time_taken = end_time - start_time
        print(f"Time taken to generate response: {time_taken:.2f} seconds")
        
        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Initialize the model globally for reuse
llm = LocalLLM()

def generate_response(prompt: str) -> str:
    """Generate a response using the local LLM."""
    return llm.generate_response(prompt)
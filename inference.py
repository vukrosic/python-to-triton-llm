import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
from llm import MinimalLLM, ModelConfig, set_seed

class TextGenerator:
    def __init__(self, model_path: str = "trained_model.pth"):
        """Initialize the text generator with a trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔍 Using device: {self.device}")
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load trained model
        if os.path.exists(model_path):
            print(f"🔄 Loading trained model from {model_path}")
            self.model = self.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
        
        self.model.eval()
        print("✅ Model loaded successfully!")
        
    def load_model(self, model_path: str):
        """Load a trained model from checkpoint"""
        try:
            # Try loading with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"⚠️ Warning: {e}")
            print("🔄 Trying alternative loading method...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = ModelConfig()
            config.vocab_size = self.tokenizer.vocab_size
        
        model = MinimalLLM(config)
        model = model.to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.8, 
                top_p: float = 0.9, top_k: int = 50, do_sample: bool = True) -> str:
        """Generate text continuation from a prompt"""
        print(f"🎯 Generating with prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    
    def generate_code(self, prompt: str, max_length: int = 300, temperature: float = 0.7) -> str:
        """Generate Python code with optimized parameters for code generation"""
        return self.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            do_sample=True
        )
    
    def interactive_generation(self):
        """Interactive text generation loop"""
        print("\n🚀 Interactive Generation Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🎯 Enter your prompt: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("""
Available commands:
- quit: Exit the program
- help: Show this help message
- code <prompt>: Generate Python code
- text <prompt>: Generate general text
                    """)
                    continue
                elif user_input.lower().startswith('code '):
                    prompt = user_input[5:]
                    print(f"\n🐍 Generating Python code...")
                    result = self.generate_code(prompt)
                elif user_input.lower().startswith('text '):
                    prompt = user_input[5:]
                    print(f"\n📝 Generating text...")
                    result = self.generate(prompt)
                else:
                    # Default to code generation for Python-related prompts
                    print(f"\n🤖 Generating...")
                    result = self.generate(user_input)
                
                print(f"\n✨ Generated Output:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Text Generation with Trained LLM")
    parser.add_argument("--model_path", type=str, default="trained_model.pth", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    try:
        # Initialize generator
        generator = TextGenerator(model_path=args.model_path)
        
        if args.interactive:
            generator.interactive_generation()
        elif args.prompt:
            result = generator.generate(
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print(f"\n✨ Generated Output:")
            print("-" * 40)
            print(result)
            print("-" * 40)
        else:
            print("No prompt provided. Use --prompt or --interactive")
            print("Example: python inference.py --prompt 'def matrix_multiply'")
            print("Example: python inference.py --interactive")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have trained the model first using 'python llm.py'")

if __name__ == "__main__":
    main()

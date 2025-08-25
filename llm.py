import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
import pandas as pd
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    num_epochs: int = 10

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data from training_data.csv"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_training_data.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} training examples, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing training_data.csv (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data from CSV
    try:
        df = pd.read_csv("training_data.csv")
        print(f"📊 Loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Format as Python function -> Triton kernel pairs
        texts = []
        for _, row in df.iterrows():
            python_func = row['python_function_body'].strip()
            triton_kernel = row['triton_kernel_body'].strip()
            
            # Create training example: Python function followed by Triton kernel
            training_example = f"Python function:\n{python_func}\n\nTriton kernel:\n{triton_kernel}"
            texts.append(training_example)
            
            print(f"📝 Example {len(texts)}:")
            print(f"   Python: {python_func[:50]}...")
            print(f"   Triton: {triton_kernel[:50]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        # Fallback to some example data
        texts = [
            "Python function:\ny = torch.sigmoid(x)\n\nTriton kernel:\npid = tl.program_id(axis=0)\noffsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\nmask = offsets < n_elements\nx = tl.load(input_ptr + offsets, mask=mask)\nresult = tl.sigmoid(x)\ntl.store(output_ptr + offsets, result, mask=mask)",
            "Python function:\nresult = x + y\n\nTriton kernel:\npid = tl.program_id(axis=0)\noffsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\nmask = offsets < n_elements\nx = tl.load(x_ptr + offsets, mask=mask)\ny = tl.load(y_ptr + offsets, mask=mask)\noutput = x + y\ntl.store(output_ptr + offsets, output, mask=mask)"
        ]

    print(f"Loaded {len(texts)} training examples")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_p=0.9, 
                 top_k=50, do_sample=True, pad_token_id=None, eos_token_id=None, 
                 repetition_penalty=1.0):
        """Generate text continuation from input_ids"""
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # Set default tokens if not provided
        if pad_token_id is None:
            pad_token_id = self.config.vocab_size - 1
        if eos_token_id is None:
            eos_token_id = self.config.vocab_size - 1
        
        # Initialize output with input_ids
        output = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get model predictions
                logits = self.forward(output)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in set(output[i].tolist()):
                            if previous_token < next_token_logits.shape[-1]:
                                next_token_logits[i, previous_token] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token to output
                output = torch.cat([output, next_token], dim=-1)
                
                # Check if EOS token is generated
                if (next_token == eos_token_id).any():
                    break
        
        return output

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer for specified number of epochs"""
    print(f"\n🚀 Training Small model with Muon optimizer for {config.num_epochs} epochs")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Calculate total steps for learning rate scheduling
    total_steps = len(train_loader) * config.num_epochs
    
    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = total_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{config.num_epochs}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Update epoch statistics
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            predictions = logits.argmax(dim=-1)
            epoch_correct += (predictions == y).sum().item()
            epoch_total += y.numel()
            step += 1

            # Update progress bar
            if batch_idx % 10 == 0:
                current_loss = loss.item() * config.gradient_accumulation_steps
                current_accuracy = (predictions == y).float().mean().item()
                current_perplexity = math.exp(min(current_loss, 20))
                current_lr = optimizers[0].param_groups[0]["lr"]
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_accuracy:.3f}',
                    'ppl': f'{current_perplexity:.1f}',
                    'lr': f'{current_lr:.2e}'
                })

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_correct / epoch_total
        epoch_perplexity = math.exp(min(avg_epoch_loss, 20))
        
        print(f"  📊 Epoch {epoch + 1} Summary:")
        print(f"     Loss: {avg_epoch_loss:.4f}")
        print(f"     Accuracy: {avg_epoch_accuracy:.4f}")
        print(f"     Perplexity: {epoch_perplexity:.2f}")

        # Evaluation every epoch
        eval_metrics = evaluate_model(model, val_loader, config)
        print(f"  🔍 Validation - Loss: {eval_metrics['val_loss']:.4f}, "
              f"Acc: {eval_metrics['val_accuracy']:.4f}, "
              f"PPL: {eval_metrics['val_perplexity']:.2f}")

        if eval_metrics['val_loss'] < best_val_loss:
            best_val_loss = eval_metrics['val_loss']
            print(f"  🏆 New best validation loss: {best_val_loss:.4f}")

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")

    return model, final_eval

def load_trained_model(model_path: str = "trained_model.pth"):
    """Load a pre-trained model for inference"""
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Please train the model first or provide the correct path.")
        return None, None
    
    print(f"📥 Loading pre-trained model from {model_path}")
    
    try:
        # Try loading with weights_only=False for backward compatibility
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("The model file may be corrupted or incompatible.")
        return None, None
    
    # Initialize model
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Final metrics: {checkpoint['final_metrics']}")
    
    return model, config

def interactive_demo():
    """Interactive demo to test the model with custom Python functions"""
    print("\n🎮 INTERACTIVE DEMO MODE")
    print("=" * 40)
    print("Enter Python functions and see the generated Triton kernels!")
    print("Type 'quit' to exit.")
    
    # Load model
    model, config = load_trained_model()
    if model is None:
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = next(model.parameters()).device
    
    while True:
        try:
            python_func = input("\n🐍 Enter Python function (or 'quit'): ").strip()
            
            if python_func.lower() == 'quit':
                break
            
            if not python_func:
                continue
            
            print(f"\n🔄 Generating Triton kernel for: {python_func}")
            
            # Tokenize input
            input_text = f"Python function:\n{python_func}\n\nTriton kernel:\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 300,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode and extract Triton part
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            if "Triton kernel:" in generated_text:
                triton_part = generated_text.split("Triton kernel:")[1].strip()
            else:
                triton_part = generated_text[len(input_text):].strip()
            
            print(f"\n🎯 Generated Triton kernel:")
            print(triton_part)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def demonstrate_model(model: nn.Module, tokenizer, device: torch.device):
    """Demonstrate the model's ability to generate Triton code from Python functions"""
    print("\n🎯 DEMONSTRATING MODEL CAPABILITIES")
    print("=" * 50)
    
    # Example Python functions to test
    test_cases = [
        "y = torch.tanh(x)",
        "result = torch.maximum(x, y)",
        "output = torch.pow(x, 2.0)",
        "z = torch.clamp(x, min_val, max_val)"
    ]
    
    model.eval()
    with torch.no_grad():
        for i, python_func in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: Python function")
            print(f"Input: {python_func}")
            
            # Tokenize the Python function
            input_text = f"Python function:\n{python_func}\n\nTriton kernel:\n"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            
            # Generate Triton kernel
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 200,  # Allow space for Triton code
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Extract just the generated Triton part
            if "Triton kernel:" in generated_text:
                triton_part = generated_text.split("Triton kernel:")[1].strip()
            else:
                triton_part = generated_text[len(input_text):].strip()
            
            print(f"\n🎯 Generated Triton kernel:")
            print(triton_part)
            print("-" * 40)

if __name__ == "__main__":
    import sys
    
    # Check if user wants interactive demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        interactive_demo()
        sys.exit(0)
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    # Save the trained model
    model_save_path = "trained_model.pth"
    print(f"\n💾 Saving trained model to {model_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics
    }, model_save_path)
    print(f"✅ Model saved successfully!")
    
    # Demonstrate the model's capabilities
    device = next(model.parameters()).device
    demonstrate_model(model, tokenizer, device)
    
    print(f"\n🚀 Model is ready to generate Triton kernels from Python functions!")
    print(f"💡 You can now use the trained model to convert Python operations to Triton kernels.")
    print(f"🎮 Run with 'python llm.py --demo' for interactive testing!")
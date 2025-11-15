# watermark.py, contains functions relating to the watermark framework

import os
import hmac
import hashlib
import random
import torch
import numpy as np
from .models import LanguageModel
from transformers import LogitsProcessor

from tqdm import tqdm
from .fingerprinting import FingerprintingCode

def _calculate_entropy(logits):
    """Calculates the Shannon entropy of a logits distribution."""
    probabilities = torch.softmax(logits, dim=-1)
    # Add a small epsilon to prevent log(0)
    p_log_p = probabilities * torch.log2(probabilities + 1e-9)
    # Sum over the vocabulary dimension
    return -torch.sum(p_log_p, dim=-1)

def derive_key(master_secret_key: bytes, index: int, bit: int) -> bytes:
    """
    derives a 32-byte key using binary encoding
    """
    message = f"{index}_{bit}".encode("utf-8")
    new_message = hmac.new(key=master_secret_key, msg=message, digestmod=hashlib.sha256)
    return new_message.digest()[:32]

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor to apply the CGZ23 soft-scoring watermark bias.
    """
    def __init__(self, watermarker_instance, params: dict, sk: bytes):
        """
        Initializes the processor.
        
        Args:
            watermarker_instance: The main ZeroBitWatermarker object to access its methods.
            params (dict): A dictionary of dynamic parameters for this specific generation pass.
            sk (bytes): The secret key.
        """
        self.watermarker = watermarker_instance
        self.params = params
        self.sk = sk

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the watermark bias to the logits 'scores' only if entropy is high.
        """
        # Calculate entropy of the original, unbiased logits
        entropy = _calculate_entropy(scores)

        # Use the dynamic entropy_threshold from the params dictionary
        if entropy >= self.params['entropy_threshold']:
            # Call the method from the main watermarker instance
            score_vector = self.watermarker._get_score_vector(self.sk, input_ids[0])
            
            # Use the dynamic delta from the params dictionary
            scores += self.params['delta'] * score_vector
        
        return scores


class ZeroBitWatermarker:
    """
    Implements the block-by-block zero-bit watermarking scheme.
    """

    def __init__(self, model: LanguageModel, delta: float = 2.5, z_threshold: float = 4.0, entropy_threshold: float = 4.0, hashing_context: int = 5):
        """
        Initializes the watermarker.

        Args:
            model (LanguageModel): The model wrapper.
            delta (float): The bias strength (scaling factor for the score vector).
            z_threshold (float): The threshold for detecting the watermark.
            entropy_threshold (float): The entropy value above which a watermark is embedded.
            hashing_context (int): The number of previous tokens to use for the PRF context.
        """
        self.model = model
        self.delta = delta
        self.z_threshold = z_threshold
        self.entropy_threshold = entropy_threshold
        self.hashing_context = hashing_context
        
    def keygen(self, key_length: int = 32) -> bytes:
        """KeyGen: Generates a random secret key."""
        return os.urandom(key_length)

    def _get_score_vector(self, sk: bytes, prefix_tokens: torch.Tensor) -> torch.Tensor:
        """
        Generates a pseudorandom score vector for a given prefix.
        The vector is drawn from a standard normal distribution N(0, 1).

        Args:
            sk (bytes): The secret key.
            prefix_tokens (torch.Tensor): The tokenized prefix.

        Returns:
            torch.Tensor: A tensor of scores, one for each token in the vocabulary.
        """
        if len(prefix_tokens) > self.hashing_context:
            context_tokens = prefix_tokens[-self.hashing_context:]
        else:
            context_tokens = prefix_tokens

        prefix_str = str(context_tokens.tolist())
        h = hmac.new(sk, prefix_str.encode('utf-8'), hashlib.sha256)

        # The full 32-byte (256-bit) hash digest is too large for torch's 64-bit seed.
        # We truncate the hash to the first 8 bytes (64 bits) to create a valid seed.
        seed_bytes = h.digest()[:8]
        seed = int.from_bytes(seed_bytes, 'big', signed=True)
        
        # Use the seed to create a deterministic torch random number generator
        rng_generator = torch.Generator(device=self.model.device)
        rng_generator.manual_seed(seed)
        
        # Generate scores from N(0, 1)
        score_vector = torch.randn(
            self.model.vocab_size, 
            generator=rng_generator, 
            device=self.model.device
        )
        return score_vector
    
    def _generate_with_params(self, sk, prompt, max_new_tokens, params, **kwargs):
        """A helper to generate text with a specific set of watermarking parameters."""
        tokenizer = self.model.tokenizer
        
        # The LogitsProcessor needs the specific params for this run
        logits_processor = WatermarkLogitsProcessor(self, params, sk)

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            ).to(self.model.device)
        else:
            token_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        
        # Create explicit attention mask to avoid warning when pad_token_id == eos_token_id
        attention_mask = torch.ones_like(token_ids)
        
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            logits_processor=[logits_processor],
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            **kwargs
        )

        with torch.no_grad():
            output_ids = self.model._model.generate(token_ids, **gen_kwargs)
            
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def _count_blocks(self, text, params):
        """A helper to count the number of blocks in a text for a given entropy threshold."""
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        
        if len(token_ids) < 2: return 0
        
        with torch.no_grad():
            outputs = self.model._model(token_ids.unsqueeze(0))
        all_logits = outputs.logits.squeeze(0)

        block_count = 0
        for i in range(len(token_ids) - 1):
            entropy = _calculate_entropy(all_logits[i, :])
            if entropy >= params['entropy_threshold']:
                block_count += 1
        return block_count

    def embed(self, sk: bytes, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        """
        Embeds a watermark using a two-pass technique to ensure sufficient block count.
        Returns the raw watermarked text and the final parameters used.
        """
        # Pass 1: Generate with the initial parameters
        pass_1_params = {
            'delta': self.delta, 
            'entropy_threshold': self.entropy_threshold,
            'hashing_context': self.hashing_context
        }
        raw_text_pass1 = self._generate_with_params(sk, prompt, max_new_tokens, pass_1_params, **kwargs)
        block_count_pass1 = self._count_blocks(raw_text_pass1, pass_1_params)

        # If the first pass has enough blocks, we're done.
        if block_count_pass1 >= 75:
            # Return the text and the parameters that were used to generate it
            return raw_text_pass1, pass_1_params

        # Pass 2: If not enough blocks, try one more aggressive pass
        pass_2_params = {
            'delta': self.delta + 1.0,
            'entropy_threshold': self.entropy_threshold - 2.0,
            'hashing_context': self.hashing_context
        }
        # Ensure the new entropy threshold is valid
        if pass_2_params['entropy_threshold'] < 1.0:
            return raw_text_pass1, pass_1_params

        raw_text_pass2 = self._generate_with_params(sk, prompt, max_new_tokens, pass_2_params, **kwargs)
        
        # Return the result of the second pass, regardless of its block count
        return raw_text_pass2, pass_2_params

    def detect(self, sk: bytes, text: str, cached_logits: torch.Tensor = None) -> tuple[float, bool, int]:
        """
        Detects a watermark by checking for the signature only at high-entropy positions (blocks).
        """
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        
        num_tokens = len(token_ids)
        if num_tokens < 2:
            return 0.0, False, 0

        # Get the logits for every token position in one go.
        if cached_logits is None:
            with torch.no_grad():
            # The model returns logits for predicting the *next* token at each position.
            # So, outputs.logits[0, i, :] are the logits for predicting token i+1.
                outputs = self.model._model(token_ids.unsqueeze(0))
            all_logits = outputs.logits.squeeze(0) # Remove batch dimension
        else:
            all_logits = cached_logits

            
        total_score = 0.0
        block_count = 0 # Count of high-entropy positions found
        
        # We check each token from the first potential hash context up to the end.
        for i in range(self.hashing_context, num_tokens - 1):
            # The logits for predicting the token at position i
            current_logits = all_logits[i-1, :]
            entropy = _calculate_entropy(current_logits)

            if entropy >= self.entropy_threshold:
                block_count += 1
                
                # The token that was actually chosen at this position
                current_token_id = token_ids[i].item()
                
                # The prefix used for hashing ends at the previous token
                prefix_for_hash = token_ids[:i]
                score_vector = self._get_score_vector(sk, prefix_for_hash)
                
                token_score = score_vector[current_token_id].item()
                total_score += token_score
        
        if block_count == 0:
            return 0.0, False, 0

        z_score = total_score / np.sqrt(block_count)
        
        return z_score, z_score > self.z_threshold, block_count
    
    def parse_first_block(self, text: str, prefix: str, params: dict) -> str:
        """Extracts the first high-entropy block from text, or None if None"""
        combined_text = prefix + text
        tokenizer = self.model.tokenizer
        device = self.model.device
        tokens = tokenizer.encode(combined_text, return_tensors='pt').to(device)[0]

        if tokens.size(0) < 2:
            return None
        
        with torch.no_grad():
            model_ouput = self.model._model(tokens.unsqueeze(0))
        logits = model_ouput.logits.squeeze(0)
        prefix_len = len(tokenizer.encode(prefix, return_tensor='pt')[0])
        block_end_index = None

        # Scan from prefix end to find first high-entropy position (start of block)
        for pos in range(prefix_len, len(tokens) - 1):
            entropy = _calculate_entropy(logits[pos])
            if entropy >= params["entropy_threshold"]:
                block_end_index = pos + 1
                break
            
        if block_end_index is None:
            return None
        
        # Extract block text (from text start to first_block_end - prefix_len)
        block_tokens = tokens[prefix_len:block_end_index]

        return tokenizer.decode(block_tokens, skip_special_tokens=True)
    
class LBitWatermarker:
    def __init__(self, zero_bit_watermarker: ZeroBitWatermarker = None, model: LanguageModel = None, L: int = 32, delta: float = 2.5, z_threshold: float = 4.0, entropy_threshold: float = 4.0, hashing_context: int = 5):
        if zero_bit_watermarker:
            self.zero_bit = zero_bit_watermarker
            self.model = zero_bit_watermarker.model
        else:
            if model is None:
                raise ValueError("Must provide either zero_bit_watermarker or model")
            self.zero_bit = ZeroBitWatermarker(model, delta, z_threshold, entropy_threshold, hashing_context)
            self.model = model
        self.L = L

    def keygen(self, key_length: int = 32) -> bytes:
        return os.urandom(key_length)

    def embed(self, master_secret_key: bytes, message: str, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        if len(message) != self.L or any(c not in {'0', '1'} for c in message):
            raise ValueError(f"Message must be a {self.L}-bit binary string")
        tokenizer = self.model.tokenizer
        device = self.model.device

        pass_1_params = {
            'delta': self.zero_bit.delta,
            'entropy_threshold': self.zero_bit.entropy_threshold,
            'hashing_context': self.zero_bit.hashing_context
        }
        logit_processor = LBitLogitProcessor(self.zero_bit, master_secret_key, message)

        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            ).to(device)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Create explicit attention mask to avoid warning when pad_token_id == eos_token_id
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            logits_processor=[logit_processor],
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            **kwargs
        )

        with torch.no_grad():
            output_ids = self.model._model.generate(input_ids, **gen_kwargs)
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        block_count = self.zero_bit._count_blocks(raw_text, pass_1_params)

        if block_count >= 75:
            return raw_text

        pass_2_params = {
            'delta': self.zero_bit.delta + 1.0,
            'entropy_threshold': max(1.0, self.zero_bit.entropy_threshold - 2.0),
            'hashing_context': self.zero_bit.hashing_context
        }
        self.zero_bit.delta = pass_2_params['delta']
        self.zero_bit.entropy_threshold = pass_2_params['entropy_threshold']
        with torch.no_grad():
            output_ids = self.model._model.generate(input_ids, **gen_kwargs)
        self.zero_bit.delta = pass_1_params['delta']
        self.zero_bit.entropy_threshold = pass_1_params['entropy_threshold']
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def detect(self, master_secret_key: bytes, text: str) -> str:
        """Recovers L-bit message or None per bit if neither detects"""
        tokenizer = self.model.tokenizer
        token_ids = tokenizer.encode(text, return_tensors='pt').to(self.model.device)[0]
        if len(token_ids) < 2:
            return str(None) * self.L

        with torch.no_grad():
            outputs = self.model._model(token_ids.unsqueeze(0))
        all_logits = outputs.logits.squeeze(0)

        recovered_message = ""
        for i in range(1, self.L + 1):
            z_i0, detected0, _ = self.zero_bit.detect(derive_key(master_secret_key, i, 0), text, cached_logits=all_logits)
            z_i1, detected1, _ = self.zero_bit.detect(derive_key(master_secret_key, i, 1), text, cached_logits=all_logits)
            if z_i0 <= self.zero_bit.z_threshold and z_i1 <= self.zero_bit.z_threshold:
                recovered_message += "⊥"
            elif z_i0 > self.zero_bit.z_threshold and z_i1 <= self.zero_bit.z_threshold:
                recovered_message += "0"
            elif z_i1 > self.zero_bit.z_threshold and z_i0 <= self.zero_bit.z_threshold:
                recovered_message += "1"
            else:
                recovered_message += "*"
        return recovered_message
    
class LBitLogitProcessor(LogitsProcessor):
    def __init__(self, watermarker, master_secret_key: bytes, message: str):
        self.watermarker = watermarker
        self.master_secret_key = master_secret_key
        self.message = message
        self.L = len(message)
        self.index_bit = 0
        # Initialize with a random permutation of [1, 2, ..., L]
        self.current_permutation = list(range(1, self.L + 1))
        random.shuffle(self.current_permutation)
        self.permutation_index = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Modifies logits to encode the L-Bit message during generation"""
        if len(input_ids[0]) < 2:
            return scores
        
        entropy = _calculate_entropy(scores)
        if entropy < self.watermarker.entropy_threshold:
            return scores
        
        # Check if we've used all positions in current permutation
        if self.permutation_index >= self.L:
            # Generate a new random permutation for the next cycle
            self.current_permutation = list(range(1, self.L + 1))
            random.shuffle(self.current_permutation)
            self.permutation_index = 0
        
        # Get the bit position from the current permutation
        bit_position = self.current_permutation[self.permutation_index]
        bit_value = int(self.message[bit_position - 1])  # Either 0 or 1
        self.permutation_index += 1
        self.index_bit += 1

        key = derive_key(self.master_secret_key, bit_position, bit_value)
        context_ids = input_ids[0][-self.watermarker.hashing_context:]
        score_vector = self.watermarker._get_score_vector(key, context_ids)
        scores = scores + self.watermarker.delta * score_vector

        return scores

class MultiUserWatermarker:
    """
    Implements the multi-user scheme by embedding fingerprinting codes via the L-bit scheme.
    """
    def __init__(self, lbit_watermarker: LBitWatermarker):
        self.lbw = lbit_watermarker
        # Initialize the fingerprinting system with the number of users and codeword length (L)
        self.fingerprinter = FingerprintingCode(L=self.lbw.L)

    def keygen(self, key_length: int = 32) -> bytes:
        """Generates a single master secret key for the entire system."""
        return self.lbw.keygen(key_length)

    def embed(self, master_key: bytes, user_id: int, prompt: str, **kwargs) -> str:
        """Embeds the codeword for a specific user ID."""
        if self.fingerprinter.codewords is None or self.fingerprinter.codewords.size == 0:
            raise ValueError("Fingerprinter codes not generated. Call .fingerprinter.gen() first.")
        if user_id >= self.fingerprinter.N or user_id < 0:
            raise ValueError(f"User ID {user_id} is out of range for {self.fingerprinter.N} users.")
        
        # Get the user's assigned codeword (e.g. '00000001...')
        codeword_array = self.fingerprinter.codewords[user_id]
        codeword = "".join(map(str, codeword_array))
        
        print(f"Embedding codeword '{codeword}' for User ID {user_id}...")
        
        # Use the L-bit watermarker to embed this codeword
        raw_output = self.lbw.embed(master_key, codeword, prompt, **kwargs)
        return raw_output

    def trace(self, master_key: bytes, text: str, **kwargs) -> list[int]:
        """Extracts the L-bit codeword and traces it back to a list of accused user IDs."""
        # Use the L-bit watermarker to extract the noisy codeword
        print("Extracting L-bit codeword from text...")
        if self.fingerprinter.codewords is None or self.fingerprinter.user_metadata is None:
            raise ValueError("Fingerprinter codes or metadata not loaded. Call .fingerprinter.gen() and .fingerprinter.load_metadata() first.")

        noisy_codeword = self.lbw.detect(master_key, text, **kwargs)
        print(f"  - Recovered Codeword: {noisy_codeword}")
        
        # Use the fingerprinter to find the best match(es)
        print("Tracing codeword to find user(s)...")
        accused_users = self.fingerprinter.trace(noisy_codeword)
        
        return accused_users
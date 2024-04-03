import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from typing import Optional
import abc
import math
import numbers
from torch.nn import functional as F
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import os
import difflib
import json






class GroundascoreCrossAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states, *args)

        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        #if attention_probs.requires_grad == True:
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # save text-conditioned attention map only
        # get attention map of ref
        if hidden_states.shape[0] == 4:
            attn.hs = hidden_states[2:3]
        # get attention map of trg
        else:
            attn.hs = hidden_states[:1]

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states



def register_attention_control(model):
    attn_procs = {}
    for name in model.unet.attn_processors.keys():
        attn_procs[name] = GroundascoreCrossAttnProcessor()
    model.unet.set_attn_processor(attn_procs)


def create_directory_if_not_exists(path):
    # Base case: if the directory already exists, do nothing
    if os.path.exists(path):
        return
    # Recursive case: create the parent directory first
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        create_directory_if_not_exists(parent_dir)
    # Create the current directory
    os.mkdir(path)



def find_all_continuous_token_positions_in_b(a_tokens, b_tokens):
    """
    Find all positions of continuous tokens from A (excluding start and end tokens) in B,
    ensuring the tokens are continuous in B as well. Returns a list of positions for each token in A
    that is found in a continuous sequence in B.
    
    Parameters:
    - a_tokens: List[List[int]], token list for A
    - b_tokens: List[List[int]], token list for B
    
    Returns:
    - List[int]: All positions of continuous tokens from A in B, considering continuity, for each token in the sequence.
    """
    
    # Start and end tokens to exclude
    start_token = 49406
    end_token = 49407
    zero_token = 0
    a_tokens = a_tokens.tolist()
    b_tokens = b_tokens.tolist()


    
    # Extract tokens from A excluding start and end tokens
    a_core_tokens = [token for token in a_tokens[0] if token not in [start_token, end_token, zero_token]]
    
    # Initialize a list to store positions of continuous tokens in B
    positions = []
    
    # Search for continuous tokens of A in B
    b_len = len(b_tokens[0])
    a_len = len(a_core_tokens)
    for i in range(b_len):
        # Check if the slice of B from the current position matches the continuous tokens from A
        if i + a_len <= b_len and b_tokens[0][i:i+a_len] == a_core_tokens:
            # Add positions for each token in the continuous sequence from A found in B
            positions.extend(list(range(i + 1, i + 1 + a_len)))
            break  # Assuming we only need the first occurrence
    
    return positions


def find_difference(str1, str2):     
    str1 = re.sub(r'[^\w\s]', '', str1)
    str2 = re.sub(r'[^\w\s]', '', str2)
    lines1 = str1.strip().split()
    lines2 = str2.strip().split()
    gather = []
    output1, output2 = [], []
    for line in difflib.unified_diff(lines1, lines2, fromfile='file1', tofile='file2', lineterm='', n=0):
        for prefix in ('---', '+++', '@@'):
            if line.startswith(prefix):
                break
        else:
            gather.append(line)
    tmp = []
    for line in gather:
        if tmp and tmp[0][0] == line[0]:
            tmp.append(line)
        elif tmp:
            if tmp[0][0] == '-':
                output1.append(' '.join([t[1:] for t in tmp]))
            else:
                output2.append(' '.join([t[1:] for t in tmp]))
            tmp = [line]
        else:
            tmp = [line]
    if tmp:
        if tmp[0][0] == '-': 
            output1.append(' '.join([t[1:] for t in tmp]))
        else:
            output2.append(' '.join([t[1:] for t in tmp]))

    # Function to prepend articles "a", "an", "the" if they are present in the original strings
    def prepend_articles(original, differences):
        modified = []
        for diff in differences:
            diff_words = diff.split()
            first_diff_word = diff_words[0]
            index_in_original = original.index(first_diff_word) if first_diff_word in original else -1
            if index_in_original > 0 and original[index_in_original - 1] in ["a", "an", "the"]:
                modified.append(original[index_in_original - 1] + " " + diff)
            else:
                modified.append(diff)
        return modified

    output1 = prepend_articles(lines1, output1)
    output2 = prepend_articles(lines2, output2)

    return output1, output2


def find_phrase_word_indices(sentence, phrases):
    # Tokenize the sentence and keep track of each word's start and end indices
    sentence = re.sub(r'[^\w\s]', '', sentence)
    phrases = [re.sub(r'[^\w\s]', '', phrase) for phrase in phrases]
    print("s",sentence)
    print("p",phrases)
    words_in_sentence = []
    start = 0
    for word in sentence.split():
        end = start + len(word)
        words_in_sentence.append((word, start, end))
        start = end + 1  # +1 for the space

    # Function to find all occurrences of a phrase and individual word indices within the phrase
    def find_occurrences(phrase):
        phrase_words = phrase.split()
        occurrences = []
        for i in range(len(words_in_sentence)):
            if words_in_sentence[i][0] == phrase_words[0]:
                match = True
                occurrence_indices = []
                for j in range(len(phrase_words)):
                    if i+j >= len(words_in_sentence) or words_in_sentence[i+j][0] != phrase_words[j]:
                        match = False
                        break
                    else:
                        occurrence_indices.append([words_in_sentence[i+j][1], words_in_sentence[i+j][2]])
                if match:
                    occurrences = occurrence_indices
        print(occurrences)
        
        return occurrences

    # Finding indices for each phrase
    results = [find_occurrences(phrase) for phrase in phrases]

    return results

def log_image_optimization_params(output_dir, text_source, text_target, num_iters,
 bbox, beta, cutloss_flag,image_path, reweight_flags):
    """
    Logs the parameters for the image optimization process into a text file.

    Parameters:
    - output_dir: The directory where the log file will be saved.
    - text_source: The source text for the image optimization.
    - text_target: The target text for the image optimization.
    - num_iters: The number of iterations for the optimization.
    - bbox: The bounding box for the optimization.
    - beta: The beta parameters for the optimization.
    - cutloss_flag: Flags for cutloss in the optimization.

    """
    params = {
        "text_source": text_source,
        "text_target": text_target,
        "num_iters": num_iters,
        "bbox": bbox,
        "beta": beta,
        "intersection_mask_weight": 0.3,
        "cutloss_flag": cutloss_flag,
        "image_path" : image_path,
        "output_dir" :output_dir,
        "reweight_flags" : reweight_flags
    }
    log_file_path = os.path.join(output_dir, "optimization_params.json")
    with open(log_file_path, "w") as log_file:
        # Writing the parameters in a readable format
        log_file.write(json.dumps(params, indent=4))

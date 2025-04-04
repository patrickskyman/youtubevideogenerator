import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
import os
import argparse
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchaudio
from typing import Dict, List, Optional, Tuple
import time
import yaml
import json
import random
import sounddevice as sd
from tqdm import tqdm
import sys
from torch.nn import L1Loss, MSELoss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the models directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Update HiFi-GAN imports
try:
    from voice_cloner.hifigan_config import HiFiGANConfig
    from models.hifigan.models import Generator as HiFiGANGenerator
    from models.hifigan.env import AttrDict
    import json

    from models.hifigan.env import AttrDict
    HIFIGAN_AVAILABLE = True
    logger.info("HiFi-GAN loaded successfully")
except ImportError as e:
    logger.warning(f"HiFi-GAN not found: {e}. Falling back to basic vocoder if needed.")
    HIFIGAN_AVAILABLE = False
###test 
## From your project root
# python3 -c "from models.hifigan.models import Generator; print('HiFi-GAN import successful')"


# Add Data Preparation Utilities first
def prepare_dataset_from_dir(audio_dir, text_path, output_dir, config, test_split=0.1, val_split=0.1):
    """
    Prepare dataset from directory of audio files and text transcripts
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load text transcripts
        text_data = {}
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    filename, text = parts
                    text_data[filename] = text
        
        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(glob.glob(os.path.join(audio_dir, f"*{ext}")))
        
        # Match audio files with transcripts
        valid_files = []
        for audio_path in audio_files:
            filename = os.path.basename(audio_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            if filename in text_data:
                valid_files.append((audio_path, text_data[filename], filename))
            elif filename_no_ext in text_data:
                valid_files.append((audio_path, text_data[filename_no_ext], filename))
        
        logger.info(f"Found {len(valid_files)} valid audio files with transcripts")
        
        # Shuffle data
        random.shuffle(valid_files)
        
        # Split data
        test_size = int(len(valid_files) * test_split)
        val_size = int(len(valid_files) * val_split)
        train_size = len(valid_files) - test_size - val_size
        
        train_files = valid_files[:train_size]
        val_files = valid_files[train_size:train_size+val_size]
        test_files = valid_files[train_size+val_size:]
        
        # Create metadata files
        create_metadata_file(train_files, os.path.join(output_dir, "train_metadata.csv"), output_dir)
        create_metadata_file(val_files, os.path.join(output_dir, "val_metadata.csv"), output_dir)
        create_metadata_file(test_files, os.path.join(output_dir, "test_metadata.csv"), output_dir)
        
        logger.info(f"Dataset prepared: {train_size} training, {val_size} validation, {test_size} test samples")
        
        return True
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return False

def create_metadata_file(files, output_path, output_dir):
    """Create metadata CSV file and copy audio files"""
    try:
        # Create audio directory
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Create metadata
        metadata = []
        
        for src_path, text, filename in files:
            # Copy audio file
            dst_path = os.path.join(audio_dir, filename)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            # Add to metadata
            rel_path = os.path.join("audio", filename)
            metadata.append({
                "text": text,
                "audio_path": rel_path
            })
        
        # Write metadata to CSV
        import pandas as pd
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created metadata file with {len(metadata)} entries: {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")
        return False
    
class Config:
    """Enhanced configuration for the voice cloning system."""
    def __init__(self):
        self.sample_rate = 22050
        self.n_mels = 80
        self.n_fft = 2048
        self.hop_length = 256
        self.win_length = 1024
        self.f_min = 0
        self.f_max = 8000
        
        self.embedding_dim = 512
        self.encoder_hidden = 768
        self.decoder_hidden = 768
        self.attention_dim = 256
        self.hidden_size = 768
        self.prenet_dims = [256, 256]
        self.postnet_filters = 512
        
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.weight_decay = 1e-6
        self.max_epochs = 1000
        self.early_stopping_patience = 10
        self.checkpoint_dir = "checkpoints"
        self.data_dir = "data"
        self.audio_length = 10  # seconds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4  # for DataLoader
        
        # HiFi-GAN settings
        self.vocoder_model = "universal_v1"
        self.use_pretrained_vocoder = True
        self.hifigan_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'hifigan')
        self.hifigan_config_path = os.path.join(self.hifigan_dir, 'UNIVERSAL_V1', 'config.json')
        self.hifigan_checkpoint_path = os.path.join(self.hifigan_dir, 'UNIVERSAL_V1', 'g_02500000')
        self.hifigan_model = None
        self.hifigan_config = None
        
        self.mel_clipnorm = 1.0
        self.gradient_accumulation_steps = 1
        self.use_mixed_precision = torch.cuda.is_available()
        self.use_data_augmentation = True
        self.augmentation_prob = 0.5
        
        # Losses weights
        self.mel_loss_weight = 1.0
        self.postnet_loss_weight = 1.0

    def load_vocoder(self):
        """Load the specified HiFi-GAN vocoder"""
        if self.use_pretrained_vocoder and HIFIGAN_AVAILABLE:
            self.hifigan_model, self.hifigan_config = load_hifigan_model(self.vocoder_model)
            return True
        return False
    
class ImprovedAudioProcessor:
    """Improved audio processing with better mel extraction and normalization."""
    def __init__(self, config):
        self.config = config
        self.mel_basis = librosa.filters.mel(
            sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels,
            fmin=config.f_min, fmax=config.f_max
        )
        self.mel_mean = None
        self.mel_std = None
        self.is_fitted = False
        self.use_dynamic_range_compression = True
        self.ref_level_db = 20
        self.min_level_db = -100
        
        # For preprocessing
        self.top_db = 60
        self.preemphasis = 0.97
        self.silence_threshold = 0.05
    
    def load_normalization_stats(self, path):
        """Load normalization statistics from file."""
        if os.path.exists(path):
            norm_data = np.load(path)
            self.mel_mean, self.mel_std = norm_data["mean"], norm_data["std"]
            self.is_fitted = True
            return True
        return False
    
    def normalize_mel(self, mel):
        """Apply dynamic range compression and normalization to mel spectrograms."""
        if self.use_dynamic_range_compression:
            mel = 20 * np.log10(np.maximum(1e-5, mel))
            mel = np.clip((mel - self.ref_level_db + self.min_level_db) / self.min_level_db, 0, 1)
        if self.is_fitted:
            mel = (mel - self.mel_mean[:, np.newaxis]) / (self.mel_std[:, np.newaxis] + 1e-8)
        return mel
    
    def denormalize_mel(self, mel):
        """Revert normalization and dynamic range compression."""
        if self.is_fitted:
            mel = mel * self.mel_std[:, np.newaxis] + self.mel_mean[:, np.newaxis]
        if self.use_dynamic_range_compression:
            mel = (np.clip(mel, 0, 1) * -self.min_level_db) + self.min_level_db + self.ref_level_db
            mel = np.power(10.0, mel * 0.05)
        return mel
    
    def preprocess_audio(self, audio):
        """Apply preprocessing to audio before feature extraction."""
        # Trim silent parts
        if self.silence_threshold > 0:
            audio, _ = librosa.effects.trim(audio, top_db=self.top_db)
        
        # Apply preemphasis filter
        if self.preemphasis > 0:
            audio = np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])
        
        # Normalize audio to range [-1, 1]
        max_norm = np.abs(audio).max()
        if max_norm > 0:
            audio = audio / max_norm
        
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram from audio with preprocessing."""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        
        # Preprocess the audio
        audio = self.preprocess_audio(audio)
        
        # Compute STFT
        stft = librosa.stft(
            audio, 
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length, 
            win_length=self.config.win_length, 
            window='hann'
        )
        
        # Convert to power spectrogram
        magnitude = np.abs(stft) ** 2
        
        # Convert to mel scale
        mel_spectrogram = np.dot(self.mel_basis, magnitude)
        
        # Apply normalization
        mel_spectrogram = self.normalize_mel(mel_spectrogram)
        
        return mel_spectrogram
    
    def audio_to_mel_tensor(self, audio_path, return_audio=False):
        """Convert audio file to mel spectrogram tensor."""
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        mel_spectrogram = self.extract_mel_spectrogram(audio)
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
        
        if return_audio:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            return mel_tensor, audio_tensor
        
        return mel_tensor

class ImprovedSpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm_hidden = 768
        self.lstm_layers = 3
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(256, self.lstm_hidden, self.lstm_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(self.lstm_hidden * 2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.projection = nn.Linear(self.lstm_hidden * 2, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, audio):
        """
        Forward pass for speaker encoder
        Args:
            audio: Audio tensor of shape [batch_size, sequence_length]
                  or [batch_size, n_mels, sequence_length] for mel spectrograms
        """
        if audio.dim() == 2:  # Raw audio
            x = audio.unsqueeze(1)  # [B, 1, T]
        elif audio.dim() == 3:  # Mel spectrogram [B, n_mels, T]
            # Take mean across mel dimension to get [B, 1, T]
            x = torch.mean(audio, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected input shape: {audio.shape}")
        
        x = self.conv_layers(x)  # [B, 256, T//16]
        x = x.transpose(1, 2)    # [B, T//16, 256]
        
        # Handle variable length sequences
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = self.lstm(x)
        else:
            x, _ = self.lstm(x)  # [B, T//16, 2*lstm_hidden]
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)  # [B, T//16]
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # [B, 2*lstm_hidden]
        
        # Project to embedding dimension
        x = self.projection(x)  # [B, embedding_dim]
        speaker_embedding = self.layer_norm(self.dropout(x))
        
        # L2 normalize the embedding
        return F.normalize(speaker_embedding, p=2, dim=1)

class ImprovedTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Try to load preferred model, fall back to alternative if needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.text_encoder = AutoModel.from_pretrained("roberta-base")
            logger.info("Using RoBERTa-base for text encoding")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa-base: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
                logger.info("Using BERT-base-uncased for text encoding")
            except Exception as e:
                logger.error(f"Could not load BERT-base-uncased: {e}")
                raise RuntimeError("Could not load any text encoder model")
        
        # Add projection layers to match embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, config.embedding_dim)
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, text):
        """
        Forward pass for text encoder
        Args:
            text: String or list of strings to encode
        """
        if isinstance(text, str):
            text = [text]
            
        # Tokenize the input text
        tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        
        # Move to correct device
        tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
        
        # Encode text using the transformer model
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
        
        # Get the last hidden state
        last_hidden_states = outputs.last_hidden_state
        
        # Weighted pooling using attention mask
        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        weighted_sum = torch.sum(last_hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        
        # Project to embedding dimension
        projected = self.projection(weighted_sum)
        normalized = self.layer_norm(projected)
        
        return normalized

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_dim, encoder_dim, decoder_dim, location_features=32):
        super().__init__()
        self.location_conv = nn.Conv1d(1, location_features, kernel_size=31, padding=15, bias=False)
        self.location_layer = nn.Linear(location_features, attention_dim)
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")
    
    def forward(self, query, keys, values, processed_memory, attention_weights_cat):
        """Forward pass for location-sensitive attention"""
        processed_query = self.query_layer(query.unsqueeze(1))  # [B, 1, attention_dim]
        processed_loc = self.location_layer(self.location_conv(attention_weights_cat).transpose(1, 2))  # [B, time, attention_dim]
        
        # Compute attention scores
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_loc)).squeeze(-1)  # [B, time]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(energies, dim=1)  # [B, time]
        
        # Apply attention weights to values
        attention_context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [B, encoder_dim]
        
        return attention_context, attention_weights

class ImprovedDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.n_mels = config.n_mels
        self.hidden_size = config.hidden_size
        self.prenet_dims = config.prenet_dims
        
        # Pre-net (bottleneck)
        modules = []
        in_dim = self.n_mels
        for dim in self.prenet_dims:
            modules.extend([
                nn.Linear(in_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.5)  # High dropout for regularization
            ])
            in_dim = dim
        self.prenet = nn.Sequential(*modules)
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(
            config.attention_dim, 
            self.embedding_dim * 2,  # Combined text and speaker embedding
            self.hidden_size
        )
        
        # LSTM decoder
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(self.prenet_dims[-1] + self.embedding_dim * 2, self.hidden_size),
            nn.LSTMCell(self.hidden_size, self.hidden_size)
        ])
        
        # Frame projection (mel prediction)
        self.frame_projection = nn.Linear(
            self.hidden_size + self.embedding_dim * 2,  # LSTM output + context
            self.n_mels
        )
        
        # Post-net for refining mel predictions
        self.postnet = nn.Sequential(
            nn.Conv1d(self.n_mels, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters), nn.Tanh(), nn.Dropout(0.5),
            nn.Conv1d(config.postnet_filters, self.n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.n_mels), nn.Dropout(0.5)
        )
        
        # Learnable initial states
        self.initial_hidden = nn.Parameter(torch.zeros(2, self.hidden_size))
        self.initial_cell = nn.Parameter(torch.zeros(2, self.hidden_size))
        
        # Go frame for kickstarting the decoding process
        self.go_frame = nn.Parameter(torch.zeros(1, 1, self.n_mels))
        
        # Stopping token prediction
        self.stop_projection = nn.Linear(self.hidden_size + self.embedding_dim * 2, 1)
    
    def initialize_decoder_states(self, batch_size):
        """Initialize hidden and cell states for the decoder"""
        h_states = [
            self.initial_hidden[0].repeat(batch_size, 1),
            self.initial_hidden[1].repeat(batch_size, 1)
        ]
        c_states = [
            self.initial_cell[0].repeat(batch_size, 1),
            self.initial_cell[1].repeat(batch_size, 1)
        ]
        return h_states, c_states
    
    def parse_decoder_inputs(self, mel_targets):
        """Prepare decoder inputs from target mels by prepending go frame"""
        if mel_targets is None:
            return self.go_frame.repeat(1, 1, 1)
        
        B = mel_targets.size(0)
        go_frames = self.go_frame.repeat(B, 1, 1)
        
        # Concatenate go frame with all frames except the last one
        decoder_inputs = torch.cat((go_frames, mel_targets[:, :-1, :]), dim=1)
        return decoder_inputs
    
    def forward(self, combined_embedding, target_mels=None, max_length=None):
        """
        Main decoder forward pass
        Args:
            combined_embedding: Combined text and speaker embeddings [B, embedding_dim*2]
            target_mels: Target mel spectrograms for teacher forcing [B, T, n_mels]
            max_length: Maximum decoding length
        """
        batch_size = combined_embedding.size(0)
        device = combined_embedding.device
        
        # Set up encoder outputs for attention
        encoder_outputs = combined_embedding.unsqueeze(1)  # [B, 1, embedding_dim*2]
        
        # Initialize decoder states
        h_states, c_states = self.initialize_decoder_states(batch_size)
        
        # Determine sequence length
        if max_length is None:
            max_length = 1000 if target_mels is None else target_mels.size(1)
        
        # Initialize attention weights and context
        attention_weights = torch.zeros(batch_size, 1, encoder_outputs.size(1)).to(device)
        attention_context = torch.zeros(batch_size, combined_embedding.size(-1)).to(device)
        
        # Initialize first decoder input
        if target_mels is None:
            decoder_input = self.go_frame.repeat(batch_size, 1, 1).squeeze(1).to(device)
        else:
            decoder_input = target_mels[:, 0, :]
        
        # Outputs collecting
        mel_outputs, alignments, stop_tokens = [], [], []
        processed_memory = encoder_outputs.squeeze(1)  # Pre-compute once
        
        # Main decoding loop
        for t in range(max_length):
            # Run through pre-net to get bottleneck features
            prenet_output = self.prenet(decoder_input)
            
            # Combine with attention context
            lstm_input = torch.cat([prenet_output, attention_context], dim=-1)
            
            # First LSTM layer
            h_states[0], c_states[0] = self.lstm_layers[0](lstm_input, (h_states[0], c_states[0]))
            
            # Second LSTM layer
            h_states[1], c_states[1] = self.lstm_layers[1](h_states[0], (h_states[1], c_states[1]))
            
            # Compute attention
            attention_context, attention_weights_t = self.attention(
                h_states[1], 
                encoder_outputs, 
                encoder_outputs, 
                processed_memory, 
                attention_weights
            )
            attention_weights = attention_weights_t.unsqueeze(1)
            alignments.append(attention_weights_t)
            
            # Concatenate LSTM output and attention context
            decoder_lstm_output = torch.cat([h_states[1], attention_context], dim=1)
            
            # Generate mel output
            mel_output = self.frame_projection(decoder_lstm_output)
            mel_outputs.append(mel_output.unsqueeze(1))
            
            # Generate stop token prediction
            stop_token = self.stop_projection(decoder_lstm_output)
            stop_tokens.append(stop_token.unsqueeze(1))
            
            # Get next input (teacher forcing or generated)
            if target_mels is not None and t < target_mels.size(1) - 1:
                decoder_input = target_mels[:, t+1, :]
            else:
                decoder_input = mel_output
        
        # Stack outputs
        mel_outputs = torch.cat(mel_outputs, dim=1)  # [B, T, n_mels]
        alignments = torch.stack(alignments, dim=1)  # [B, T, src_len]
        stop_tokens = torch.cat(stop_tokens, dim=1)  # [B, T, 1]
        
        # Apply post-net to refine mel predictions
        postnet_output = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)  # [B, T, n_mels]
        mel_outputs_postnet = mel_outputs + postnet_output
        
        return mel_outputs_postnet, alignments, stop_tokens, mel_outputs

class ImprovedVocoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_pretrained = config.use_pretrained_vocoder
        
        # Try to load pre-trained HiFi-GAN
        if self.use_pretrained and HIFIGAN_AVAILABLE:
            try:
                if not os.path.exists(config.hifigan_config_path):
                    logger.warning(f"HiFi-GAN config file not found at {config.hifigan_config_path}")
                    self.use_pretrained = False
                    self.initialize_basic_vocoder()
                    return
                
                logger.info(f"Loading HiFi-GAN config from {config.hifigan_config_path}")
                with open(config.hifigan_config_path, "r") as f:
                    h = json.load(f)
                    h = AttrDict(h)
                
                self.hifigan = HiFiGANGenerator(h)
                
                if not os.path.exists(config.hifigan_checkpoint_path):
                    logger.warning(f"HiFi-GAN checkpoint not found at {config.hifigan_checkpoint_path}")
                    self.use_pretrained = False
                    self.initialize_basic_vocoder()
                    return
                
                logger.info(f"Loading HiFi-GAN checkpoint from {config.hifigan_checkpoint_path}")
                checkpoint = torch.load(config.hifigan_checkpoint_path, map_location=config.device)
                self.hifigan.load_state_dict(checkpoint['generator'])
                self.hifigan.eval()
                self.hifigan.remove_weight_norm()
                logger.info("Successfully loaded pre-trained HiFi-GAN vocoder")
            
            except Exception as e:
                logger.error(f"Error loading pre-trained HiFi-GAN vocoder: {e}")
                logger.info("Falling back to basic vocoder implementation")
                self.use_pretrained = False
                self.initialize_basic_vocoder()
        else:
            logger.info("Using basic vocoder implementation")
            self.initialize_basic_vocoder()

    def initialize_basic_vocoder(self):
        """Initialize a simpler vocoder when HiFi-GAN is not available"""
        self.conv_in = nn.Conv1d(self.config.n_mels, 512, kernel_size=7, padding=3)
        self.upsample_rates = [4, 4, 4, 4]
        self.upsample_blocks = nn.ModuleList()
        current_channels = 512
        
        for i, rate in enumerate(self.upsample_rates):
            is_last = (i == len(self.upsample_rates) - 1)
            out_channels = 1 if is_last else current_channels // 2
            
            self.upsample_blocks.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.ConvTranspose1d(
                    current_channels, 
                    out_channels, 
                    kernel_size=rate*2, 
                    stride=rate, 
                    padding=rate//2
                ),
                nn.LeakyReLU(0.1) if not is_last else nn.Tanh()
            ))
            
            if not is_last:
                current_channels //= 2

    def forward(self, mel_spec):
        """
        Convert mel spectrogram to audio
        Args:
            mel_spec: Mel spectrogram [B, T, n_mels] or [T, n_mels]
        """
        # Ensure input is properly shaped for processing
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        
        # Transpose for 1D convolution if needed
        if mel_spec.shape[1] != self.config.n_mels:
            mel_spec = mel_spec.transpose(1, 2)  # [B, n_mels, T]
        
        if self.use_pretrained and HIFIGAN_AVAILABLE:
            with torch.no_grad():
                waveform = self.hifigan(mel_spec)
        else:
            x = self.conv_in(mel_spec)
            for upsample_block in self.upsample_blocks:
                x = upsample_block(x)
            waveform = x
        
        # Remove batch dimension if input had no batch
        if waveform.shape[0] == 1 and len(mel_spec.shape) == 2:
            waveform = waveform.squeeze(0)
        
        return waveform

    def generate_audio(self, mel_spectrogram):
        """
        Generate audio from mel spectrogram
        Args:
            mel_spectrogram: Mel spectrogram array or tensor
        """
        self.eval()
        with torch.no_grad():
            # Convert input to torch tensor if necessary
            if isinstance(mel_spectrogram, np.ndarray):
                mel_spectrogram = torch.tensor(mel_spectrogram).to(self.config.device)
            elif isinstance(mel_spectrogram, torch.Tensor):
                mel_spectrogram = mel_spectrogram.to(self.config.device)
            else:
                raise ValueError("mel_spectrogram must be numpy array or torch tensor")
            
            # Generate waveform
            waveform = self.forward(mel_spectrogram)
            
            # Convert to numpy array
            return waveform.cpu().numpy()

class ImprovedVoiceCloningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core components
        self.speaker_encoder = ImprovedSpeakerEncoder(config)
        self.text_encoder = ImprovedTextEncoder(config)
        self.decoder = ImprovedDecoder(config)
        self.vocoder = ImprovedVocoder(config)
        
        # Global Style Tokens mechanism
        self.use_gst = True
        if self.use_gst:
            self.gst_embedding_dim = 256
            self.gst_num_tokens = 10
            self.gst_token_embedding = nn.Parameter(torch.randn(self.gst_num_tokens, self.gst_embedding_dim))
            self.gst_attention = nn.Linear(config.embedding_dim, self.gst_num_tokens)
    
    def apply_global_style_tokens(self, speaker_embedding):
        """Apply Global Style Tokens to speaker embedding for style control
        
        Args:
            speaker_embedding: Speaker embedding tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Modified speaker embedding with GST influence [batch_size, embedding_dim]
        """
        if not self.use_gst:
            return speaker_embedding
        
        # Compute attention weights for GST tokens
        gst_attention_weights = F.softmax(self.gst_attention(speaker_embedding), dim=-1)
        
        # Weight and sum the GST tokens using the helper method
        weighted_tokens = self.weighted_gst_tokens(gst_attention_weights)
        
        # Project weighted tokens to match speaker embedding dimension if needed
        if weighted_tokens.size(-1) != speaker_embedding.size(-1):
            weighted_tokens = F.linear(
                weighted_tokens,
                torch.zeros(speaker_embedding.size(-1), self.gst_embedding_dim).to(speaker_embedding.device)
            )
        
        # Combine speaker embedding with weighted GST tokens
        modified_embedding = speaker_embedding + weighted_tokens
        
        # Normalize the modified embedding
        return F.normalize(modified_embedding, p=2, dim=-1)

    def weighted_gst_tokens(self, gst_attention_weights):
        """Compute weighted GST tokens based on attention weights"""
        batch_size = gst_attention_weights.size(0)
        expanded_tokens = self.gst_token_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        weighted_tokens = torch.bmm(gst_attention_weights.unsqueeze(1), expanded_tokens).squeeze(1)
        return weighted_tokens
    
    def forward(self, text, speaker_audio=None, reference_mel=None, target_mels=None, max_length=None):
        """
        Forward pass for the voice cloning model
        Args:
            text: Text to synthesize
            speaker_audio: Reference speaker audio (raw waveform)
            reference_mel: Reference speaker mel spectrogram (alternative to speaker_audio)
            target_mels: Target mel spectrograms for teacher forcing during training
            max_length: Maximum decoding length
        """
        # Encode text
        text_embedding = self.text_encoder(text)
        
        # Get speaker embedding from audio or mel spectrogram
        if speaker_audio is not None:
            speaker_embedding = self.speaker_encoder(speaker_audio)
        elif reference_mel is not None:
            speaker_embedding = self.speaker_encoder(reference_mel)
        else:
            raise ValueError("Either speaker_audio or reference_mel must be provided")
        
        # Apply GST if enabled
        if self.use_gst:
            gst_attention_weights = F.softmax(self.gst_attention(speaker_embedding), dim=-1)
            gst_embedding = self.weighted_gst_tokens(gst_attention_weights)
            speaker_embedding = speaker_embedding + gst_embedding
        
        # Combine text and speaker embeddings
        combined_embedding = torch.cat([text_embedding, speaker_embedding], dim=-1)
        
        # Decode mel spectrogram
        mel_outputs_postnet, alignments, stop_tokens, mel_outputs = self.decoder(
            combined_embedding, target_mels, max_length
        )
        
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'alignments': alignments,
            'stop_tokens': stop_tokens,
            'text_embedding': text_embedding,
            'speaker_embedding': speaker_embedding,
            'combined_embedding': combined_embedding
        }
    
    def synthesize(self, text, speaker_audio=None, reference_mel=None, max_length=1000):
        """
        Synthesize speech from text using the voice cloning model
        Args:
            text: Text to synthesize
            speaker_audio: Reference speaker audio
            reference_mel: Reference speaker mel spectrogram (alternative to speaker_audio)
            max_length: Maximum decoding length
        """
        self.eval()
        with torch.no_grad():
            # Forward pass without target mels (inference mode)
            outputs = self.forward(text, speaker_audio, reference_mel, None, max_length)
            
            # Generate audio from mel spectrogram
            mel_spec = outputs['mel_outputs_postnet']
            audio = self.vocoder.generate_audio(mel_spec)
            
            return {
                'audio': audio,
                'mel_spectrogram': mel_spec.cpu().numpy(),
                'alignments': outputs['alignments'].cpu().numpy()
            }

class VoiceCloningDataset(Dataset):
    """Dataset for voice cloning system"""
    def __init__(self, config, data_dir, metadata_path, processor=None, is_training=True):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.is_training = is_training
        
        # Load metadata (assume it's a CSV with text, audio_path columns)
        import pandas as pd
        try:
            self.metadata = pd.read_csv(metadata_path)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = pd.DataFrame(columns=["text", "audio_path"])
        
        # Initialize audio processor
        if processor is None:
            self.processor = ImprovedAudioProcessor(config)
        else:
            self.processor = processor
        
        # Data augmentation
        self.use_augmentation = config.use_data_augmentation and is_training
        self.aug_prob = config.augmentation_prob
    
    def __len__(self):
        return len(self.metadata)
    
    def apply_augmentation(self, audio):
        """Apply audio augmentations"""
        # Only apply augmentation with specified probability
        if not self.use_augmentation or random.random() > self.aug_prob:
            return audio
        
        # Time stretch (0.9-1.1x)
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, stretch_factor)
        
        # Pitch shift (-2 to +2 semitones)
        if random.random() < 0.3:
            semitones = random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.config.sample_rate, n_steps=semitones)
        
        # Add small amount of noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.005)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
            audio = np.clip(audio, -1.0, 1.0)  # Ensure values stay in valid range
        
        return audio
    
    def __getitem__(self, idx):
        """Get a single training/validation sample"""
        try:
            # Get metadata for this sample
            row = self.metadata.iloc[idx]
            text = row["text"]
            audio_path = os.path.join(self.data_dir, row["audio_path"])
            
            # Load audio and convert to mel spectrogram
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # Apply augmentation if enabled
            if self.is_training:
                audio = self.apply_augmentation(audio)
            
            # Process audio to mel spectrogram
            mel_spectrogram = self.processor.extract_mel_spectrogram(audio)
            mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            return {
                "text": text,
                "audio": audio_tensor,
                "mel_spectrogram": mel_tensor
            }
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return an empty sample as a fallback
            return {
                "text": "",
                "audio": torch.zeros(1000, dtype=torch.float32),
                "mel_spectrogram": torch.zeros(self.config.n_mels, 50, dtype=torch.float32)
            }

class VoiceCloningTrainer:
    """Training logic for voice cloning system"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = ImprovedVoiceCloningModel(config).to(self.device)
        
        # Optimization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.stop_token_loss = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Mix precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Audio processor
        self.audio_processor = ImprovedAudioProcessor(config)
        
        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_patience = config.early_stopping_patience
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        # Save best model if this is the best so far
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "best_model.pth"))
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return False
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 0)
            val_loss = checkpoint.get('val_loss', float('inf'))
            
            logger.info(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def compute_loss(self, outputs, batch):
        """Compute training loss"""
        # Get target mel spectrograms
        target_mels = batch["mel_spectrogram"].to(self.device)
        
        # Get model outputs
        mel_outputs = outputs["mel_outputs"]
        mel_outputs_postnet = outputs["mel_outputs_postnet"]
        stop_tokens = outputs["stop_tokens"]
        
        # Create target stops (ones at the end of sequence)
        seq_len = target_mels.size(1)
        target_stops = torch.zeros(stop_tokens.size()).to(self.device)
        for i in range(target_stops.size(0)):
            target_stops[i, seq_len-5:, 0] = 1.0  # Mark last 5 frames as stop
        
        # Calculate losses
        mel_loss = self.reconstruction_loss(mel_outputs, target_mels) * self.config.mel_loss_weight
        postnet_loss = self.reconstruction_loss(mel_outputs_postnet, target_mels) * self.config.postnet_loss_weight
        stop_loss = self.stop_token_loss(stop_tokens, target_stops)
        
        # Total loss
        total_loss = mel_loss + postnet_loss + stop_loss
        
        losses = {
            'total': total_loss.item(),
            'mel': mel_loss.item(),
            'postnet': postnet_loss.item(),
            'stop': stop_loss.item()
        }
        
        return total_loss, losses
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for i, batch in enumerate(pbar):
            # Prepare inputs
            text = batch["text"]
            audio = batch["audio"].to(self.device)
            mel_spectrograms = batch["mel_spectrogram"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text, speaker_audio=audio, target_mels=mel_spectrograms)
                    loss, losses_dict = self.compute_loss(outputs, batch)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.mel_clipnorm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.mel_clipnorm)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                outputs = self.model(text, speaker_audio=audio, target_mels=mel_spectrograms)
                loss, losses_dict = self.compute_loss(outputs, batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.mel_clipnorm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.mel_clipnorm)
                
                # Update weights
                self.optimizer.step()
            
            # Track losses
            epoch_losses.append(losses_dict)
            
            # Update progress bar
            pbar.set_postfix(loss=f"{losses_dict['total']:.4f}")
            
            # Gradient accumulation
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Compute average losses for the epoch
        avg_loss = np.mean([loss['total'] for loss in epoch_losses])
        avg_mel_loss = np.mean([loss['mel'] for loss in epoch_losses])
        avg_postnet_loss = np.mean([loss['postnet'] for loss in epoch_losses])
        avg_stop_loss = np.mean([loss['stop'] for loss in epoch_losses])
        
        logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Mel: {avg_mel_loss:.4f}, "
                   f"Postnet: {avg_postnet_loss:.4f}, Stop: {avg_stop_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
            for batch in pbar:
                # Prepare inputs
                text = batch["text"]
                audio = batch["audio"].to(self.device)
                mel_spectrograms = batch["mel_spectrogram"].to(self.device)
                
                # Forward pass
                outputs = self.model(text, speaker_audio=audio, target_mels=mel_spectrograms)
                _, losses_dict = self.compute_loss(outputs, batch)
                
                # Track losses
                val_losses.append(losses_dict)
                
                # Update progress bar
                pbar.set_postfix(loss=f"{losses_dict['total']:.4f}")
        
        # Compute average validation loss
        avg_val_loss = np.mean([loss['total'] for loss in val_losses])
        logger.info(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate scheduler
        self.scheduler.step(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, train_dataset, val_dataset, num_epochs=None):
        """Main training loop"""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
        # Set number of epochs
        if num_epochs is None:
            num_epochs = self.config.max_epochs
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # Check if this is the best model so far
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Log training progress
            logger.info(f"Epoch {epoch}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Best Val Loss: {self.best_val_loss:.4f}")
            
            # Generate sample during training
            if epoch % 5 == 0:
                self.generate_sample(val_dataset, epoch)
    
    def generate_sample(self, dataset, epoch):
        """Generate and save a sample during training"""
        try:
            # Pick a random sample from the dataset
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            
            # Synthesize speech
            synthesis_result = self.model.synthesize(
                sample["text"],
                reference_mel=sample["mel_spectrogram"].unsqueeze(0).to(self.device)
            )
            
            # Create output directory
            samples_dir = os.path.join(self.config.checkpoint_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save audio
            audio = synthesis_result["audio"]
            sf.write(
                os.path.join(samples_dir, f"sample_epoch_{epoch}.wav"),
                audio.squeeze(),
                self.config.sample_rate
            )
            
            # Save alignment plot
            alignment = synthesis_result["alignments"][0]
            plt.figure(figsize=(10, 4))
            plt.imshow(alignment.T, aspect='auto', origin='lower')
            plt.savefig(os.path.join(samples_dir, f"alignment_epoch_{epoch}.png"))
            plt.close()
            
            logger.info(f"Generated sample for epoch {epoch}")
        
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Filter out invalid samples
        valid_batch = [item for item in batch if item["text"] != ""]
        
        if len(valid_batch) == 0:
            return {
                "text": [""],
                "audio": torch.zeros(1, 1000, dtype=torch.float32),
                "mel_spectrogram": torch.zeros(1, self.config.n_mels, 50, dtype=torch.float32)
            }
        
        # Get texts
        texts = [item["text"] for item in valid_batch]
        
        # Pad audio tensors
        max_audio_len = max(item["audio"].size(0) for item in valid_batch)
        padded_audios = []
        for item in valid_batch:
            audio = item["audio"]
            padding = torch.zeros(max_audio_len - audio.size(0), dtype=torch.float32)
            padded_audio = torch.cat([audio, padding])
            padded_audios.append(padded_audio)
        audios = torch.stack(padded_audios)
        
        # Pad mel spectrograms
        max_mel_len = max(item["mel_spectrogram"].size(1) for item in valid_batch)
        padded_mels = []
        for item in valid_batch:
            mel = item["mel_spectrogram"]
            padding = torch.zeros(mel.size(0), max_mel_len - mel.size(1), dtype=torch.float32)
            padded_mel = torch.cat([mel, padding], dim=1)
            padded_mels.append(padded_mel)
        mels = torch.stack(padded_mels)
        
        return {
            "text": texts,
            "audio": audios,
            "mel_spectrogram": mels
        }

def prepare_datasets(config):
    """Prepare training and validation datasets"""
    # Create audio processor
    audio_processor = ImprovedAudioProcessor(config)
    
    # Try to load normalization stats or compute them
    stats_path = os.path.join(config.data_dir, "norm_stats.npz")
    if os.path.exists(stats_path):
        audio_processor.load_normalization_stats(stats_path)
    else:
        logger.info("Computing mel spectrogram normalization statistics...")
        compute_normalization_stats(config, audio_processor, stats_path)
    
    # Create datasets
    train_metadata = os.path.join(config.data_dir, "train_metadata.csv")
    val_metadata = os.path.join(config.data_dir, "val_metadata.csv")
    
    train_dataset = VoiceCloningDataset(
        config, 
        config.data_dir, 
        train_metadata, 
        processor=audio_processor, 
        is_training=True
    )
    
    val_dataset = VoiceCloningDataset(
        config, 
        config.data_dir, 
        val_metadata, 
        processor=audio_processor, 
        is_training=False
    )
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def compute_normalization_stats(config, audio_processor, save_path):
    """Compute normalization statistics for mel spectrograms"""
    # Load metadata
    import pandas as pd
    train_metadata = pd.read_csv(os.path.join(config.data_dir, "train_metadata.csv"))
    
    # Sample a subset for computing stats (to save time)
    if len(train_metadata) > 100:
        train_metadata = train_metadata.sample(100)
    
    # Collect mel spectrograms
    mel_specs = []
    
    for _, row in tqdm(train_metadata.iterrows(), total=len(train_metadata), desc="Computing mel stats"):
        try:
            audio_path = os.path.join(config.data_dir, row["audio_path"])
            audio, sr = librosa.load(audio_path, sr=config.sample_rate)
            mel_spec = audio_processor.extract_mel_spectrogram(audio)
            mel_specs.append(mel_spec)
        except Exception as e:
            logger.warning(f"Error processing {row['audio_path']}: {e}")
    
    # Compute mean and std
    mel_specs = np.concatenate(mel_specs, axis=1)
    mel_mean = np.mean(mel_specs, axis=1)
    mel_std = np.std(mel_specs, axis=1)
    
    # Save stats
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, mean=mel_mean, std=mel_std)
    
    # Update audio processor
    audio_processor.mel_mean = mel_mean
    audio_processor.mel_std = mel_std
    audio_processor.is_fitted = True
    
    logger.info(f"Saved normalization stats to {save_path}")

class InferenceInterface:
    """Interface for voice cloning inference"""
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = ImprovedVoiceCloningModel(config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_model(checkpoint_path)
        
        # Audio processor
        self.audio_processor = ImprovedAudioProcessor(config)
        
        # Try to load normalization stats
        stats_path = os.path.join(config.data_dir, "norm_stats.npz")
        if os.path.exists(stats_path):
            self.audio_processor.load_normalization_stats(stats_path)
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        try:
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def clone_voice(self, text, reference_audio_path, output_path=None):
        """Clone voice from reference audio"""
        try:
            # Load reference audio
            reference_audio, sr = librosa.load(reference_audio_path, sr=self.config.sample_rate)
            reference_audio = torch.tensor(reference_audio).unsqueeze(0).to(self.device)
            
            # Synthesize speech
            logger.info(f"Synthesizing text: '{text}'")
            synthesis_result = self.model.synthesize(text, speaker_audio=reference_audio)
            
            # Get synthesized audio
            audio = synthesis_result["audio"]
            
            # Save audio if output path is provided
            if output_path:
                sf.write(output_path, audio.squeeze(), self.config.sample_rate)
                logger.info(f"Saved synthesized audio to {output_path}")
            
            return {
                "audio": audio.squeeze(),
                "sample_rate": self.config.sample_rate,
                "mel_spectrogram": synthesis_result["mel_spectrogram"],
                "alignments": synthesis_result["alignments"]
            }
        
        except Exception as e:
            logger.error(f"Error in voice cloning: {e}")
            return None
    
    def visualize_synthesis(self, result, output_path=None):
        """Visualize synthesis results"""
        if result is None:
            logger.error("No synthesis result to visualize")
            return
        
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot mel spectrogram
            mel_spec = result["mel_spectrogram"]
            ax1.imshow(mel_spec, aspect='auto', origin='lower')
            ax1.set_title("Synthesized Mel Spectrogram")
            ax1.set_ylabel("Mel Bins")
            ax1.set_xlabel("Frames")
            
            # Plot alignment
            alignment = result["alignments"][0]
            ax2.imshow(alignment.T, aspect='auto', origin='lower')
            ax2.set_title("Attention Alignment")
            ax2.set_ylabel("Encoder Steps")
            ax2.set_xlabel("Decoder Steps")
            
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved visualization to {output_path}")
            else:
                plt.show()
            
            plt.close()
        
        except Exception as e:
            logger.error(f"Error in visualization: {e}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Voice Cloning System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prepare_parser.add_argument('--audio_dir', type=str, required=True,
                              help='Directory containing audio files')
    prepare_parser.add_argument('--text_path', type=str, required=True,
                              help='Path to transcript file')
    prepare_parser.add_argument('--test_split', type=float, default=0.1,
                              help='Test set split ratio')
    prepare_parser.add_argument('--val_split', type=float, default=0.1,
                              help='Validation set split ratio')
    prepare_parser.add_argument('--config', type=str, default='config.yaml',
                              help='Path to config file')
    prepare_parser.add_argument('--data_dir', type=str, default='data',
                              help='Output data directory')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, required=True,
                            help='Path to config file')
    train_parser.add_argument('--data_dir', type=str, default='data',
                            help='Data directory')
    train_parser.add_argument('--output_dir', type=str, default='checkpoints',
                            help='Output directory for checkpoints')
    train_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for resuming')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=16,
                            help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Learning rate')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    infer_parser.add_argument('--text', type=str, required=True,
                            help='Text to synthesize')
    infer_parser.add_argument('--reference_audio', type=str, required=True,
                            help='Reference audio file')
    infer_parser.add_argument('--output_audio', type=str,
                            help='Output audio file path')
    infer_parser.add_argument('--visualize', action='store_true',
                            help='Visualize spectrograms')

    args = parser.parse_args()

    # Initialize config
    config = Config()
    
    # Load configuration from file if it exists
    if hasattr(args, 'config') and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Execute command
    if args.command == 'prepare':
        logger.info("Starting dataset preparation")
        prepare_dataset_from_dir(
            args.audio_dir,
            args.text_path,
            args.data_dir,
            config,
            test_split=args.test_split,
            val_split=args.val_split
        )
    elif args.command == 'train':
        logger.info("Starting training")
        # Update config with command line arguments
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.output_dir:
            config.checkpoint_dir = args.output_dir
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate

        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(config)
        
        # Initialize trainer
        trainer = VoiceCloningTrainer(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        trainer.train(train_dataset, val_dataset, num_epochs=args.epochs)
    else:
        parser.print_help()
    
    args = parser.parse_args()
    
    # Configure logging level
    #  if args.debug:
     #   logging.getLogger().setLevel(logging.DEBUG)

    # Initialize config before using it
    config = Config()

    if args.command == 'prepare':
        logger.info("Starting dataset preparation")
        prepare_dataset_from_dir(
            args.audio_dir,
            args.text_path,
            config.data_dir,
            config,
            test_split=args.test_split,
            val_split=args.val_split
        )
    # Load configuration
    config = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override configuration with command-line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.checkpoint_dir = args.output_dir
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Create output directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Execute selected mode
    if args.mode == "train":
        logger.info("Starting training mode")
        
        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(config)
        
        # Initialize trainer
        trainer = VoiceCloningTrainer(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        trainer.train(train_dataset, val_dataset, num_epochs=args.epochs)
        
    elif args.mode == "infer":
        logger.info("Starting inference mode")
        
        # Verify required arguments
        if not args.text:
            parser.error("--text is required for inference mode")
        if not args.reference_audio:
            parser.error("--reference_audio is required for inference mode")
        
        # Check if reference audio exists
        if not os.path.exists(args.reference_audio):
            parser.error(f"Reference audio file not found: {args.reference_audio}")
        
        # Set default output path if not provided
        if not args.output_audio:
            output_dir = os.path.join(config.checkpoint_dir, "inference")
            os.makedirs(output_dir, exist_ok=True)
            args.output_audio = os.path.join(output_dir, f"synthesized_{int(time.time())}.wav")
        
        # Initialize inference interface
        interface = InferenceInterface(config, checkpoint_path=args.checkpoint)
        
        # Clone voice
        result = interface.clone_voice(args.text, args.reference_audio, args.output_audio)
        
        if result and args.visualize:
            visualization_path = os.path.splitext(args.output_audio)[0] + "_viz.png"
            interface.visualize_synthesis(result, visualization_path)
        
        if result:
            logger.info(f"Voice cloning completed successfully. Output saved to {args.output_audio}")
        else:
            logger.error("Voice cloning failed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

# Add Real-time Voice Conversion Pipeline
class RealTimeVoiceConverter:
    """Real-time voice conversion for live processing"""
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = ImprovedVoiceCloningModel(config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_model(checkpoint_path)
        
        # Audio processor
        self.audio_processor = ImprovedAudioProcessor(config)
        
        # Try to load normalization stats
        stats_path = os.path.join(config.data_dir, "norm_stats.npz")
        if os.path.exists(stats_path):
            self.audio_processor.load_normalization_stats(stats_path)
        
        # Speaker embedding cache
        self.speaker_embedding = None
        
        # Buffer for real-time processing
        self.buffer_size = 2 * config.sample_rate  # 2 seconds buffer
        self.audio_buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        
        # Processing parameters
        self.chunk_size = config.hop_length * 5  # Process in chunks
        self.overlap = 0.2  # 20% overlap between processed chunks
        
        # Output buffer for smooth transitions
        self.output_buffer = np.zeros(self.buffer_size)
        self.output_index = 0
    
    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        try:
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def set_reference_voice(self, reference_audio_path):
        """Extract and cache speaker embedding from reference audio"""
        try:
            # Load reference audio
            reference_audio, sr = librosa.load(reference_audio_path, sr=self.config.sample_rate)
            reference_audio = torch.tensor(reference_audio).unsqueeze(0).to(self.device)
            
            # Extract speaker embedding
            with torch.no_grad():
                self.speaker_embedding = self.model.speaker_encoder(reference_audio)
            
            logger.info("Reference voice set successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting reference voice: {e}")
            return False
    
    def process_audio_chunk(self, input_audio):
        """Process a chunk of input audio"""
        if self.speaker_embedding is None:
            logger.error("No reference voice set. Call set_reference_voice first.")
            return input_audio
        
        try:
            # Convert to tensor
            audio_tensor = torch.tensor(input_audio).unsqueeze(0).to(self.device)
            
            # Extract mel spectrogram 
            mel_spec = self.audio_processor.extract_mel_spectrogram(input_audio)
            mel_tensor = torch.tensor(mel_spec).unsqueeze(0).to(self.device)
            
            # Extract features and synthesize using speaker embedding
            with torch.no_grad():
                # Use the model's vocoder directly to convert input mel to target voice
                mel_outputs = self.model.decoder(
                    torch.cat([
                        torch.mean(mel_tensor, dim=1),  # Simple text representation
                        self.speaker_embedding
                    ], dim=-1)
                )[0]
                
                # Convert mel spectrogram to audio
                output_audio = self.model.vocoder.generate_audio(mel_outputs)
            
            return output_audio.squeeze().cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return input_audio
    
    def process_stream(self, input_stream, output_stream, duration=10):
        """Process audio stream in real-time"""
        if self.speaker_embedding is None:
            logger.error("No reference voice set. Call set_reference_voice first.")
            return False
        
        try:
            # Set up audio streams parameters
            sample_rate = self.config.sample_rate
            block_size = 1024
            
            # Define callback for stream processing
            def callback(indata, outdata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                # Add input data to buffer
                if self.buffer_index + frames > self.buffer_size:
                    # Buffer overflow, reset
                    self.buffer_index = 0
                
                self.audio_buffer[self.buffer_index:self.buffer_index+frames] = indata[:, 0]
                self.buffer_index += frames
                
                # Process when we have enough data
                if self.buffer_index >= self.chunk_size:
                    # Process chunk
                    chunk = self.audio_buffer[:self.chunk_size]
                    processed_chunk = self.process_audio_chunk(chunk)
                    
                    # Add to output buffer with overlap
                    overlap_samples = int(self.chunk_size * self.overlap)
                    
                    # Apply fade for smooth transitions
                    fade_in = np.linspace(0, 1, overlap_samples)
                    fade_out = np.linspace(1, 0, overlap_samples)
                    
                    # Add processed audio to output buffer
                    self.output_buffer[self.output_index:self.output_index+overlap_samples] *= fade_out
                    self.output_buffer[self.output_index:self.output_index+overlap_samples] += processed_chunk[:overlap_samples] * fade_in
                    self.output_buffer[self.output_index+overlap_samples:self.output_index+self.chunk_size] = processed_chunk[overlap_samples:]
                    
                    # Update buffer index
                    self.buffer_index -= (self.chunk_size - overlap_samples)
                    self.output_index = (self.output_index + self.chunk_size - overlap_samples) % self.buffer_size
                    
                    # Shift buffer
                    self.audio_buffer = np.roll(self.audio_buffer, -(self.chunk_size - overlap_samples))
                
                # Output audio
                outdata[:, 0] = self.output_buffer[self.output_index:self.output_index+frames]
                outdata[:, 1] = outdata[:, 0]  # Copy to right channel
                
                # Update output index
                self.output_index = (self.output_index + frames) % self.buffer_size
            
            # Start streams
            with sd.Stream(
                channels=2,
                samplerate=sample_rate,
                blocksize=block_size,
                callback=callback
            ):
                logger.info(f"Real-time voice conversion started. Duration: {duration}s")
                sd.sleep(int(duration * 1000))  # Convert to milliseconds
            
            return True
        
        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            return False

# Add Model Evaluation Metrics
class VoiceCloningEvaluator:
    """Evaluation metrics for voice cloning quality"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Initialize audio processor
        self.audio_processor = ImprovedAudioProcessor(config)
        
        # Try to initialize objective metrics like PESQ if available
        try:
            import pesq
            self.pesq_available = True
        except ImportError:
            logger.warning("PESQ metric not available. Install with 'pip install pesq' for enhanced evaluation.")
            self.pesq_available = False
        
        # Try to initialize other metrics
        try:
            import pystoi
            self.stoi_available = True
        except ImportError:
            logger.warning("STOI metric not available. Install with 'pip install pystoi' for enhanced evaluation.")
            self.stoi_available = False
    
    def mel_cepstral_distortion(self, reference_audio, synthesized_audio):
        """Calculate Mel Cepstral Distortion between reference and synthesized audio"""
        try:
            # Extract MFCCs for both audio samples
            mfcc_ref = librosa.feature.mfcc(
                y=reference_audio, 
                sr=self.config.sample_rate, 
                n_mfcc=13
            )
            
            mfcc_syn = librosa.feature.mfcc(
                y=synthesized_audio, 
                sr=self.config.sample_rate, 
                n_mfcc=13
            )
            
            # Handle different lengths by truncating or padding
            min_len = min(mfcc_ref.shape[1], mfcc_syn.shape[1])
            mfcc_ref = mfcc_ref[:, :min_len]
            mfcc_syn = mfcc_syn[:, :min_len]
            
            # Calculate MCD
            diff = mfcc_ref - mfcc_syn
            mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
            
            return mcd
        
        except Exception as e:
            logger.error(f"Error calculating MCD: {e}")
            return float('inf')
    
    def calculate_pesq(self, reference_audio, synthesized_audio):
        """Calculate PESQ score if available"""
        if not self.pesq_available:
            return None
        
        try:
            from pesq import pesq
            
            # Ensure the same sample rate (PESQ requires 16k or 8k)
            target_sr = 16000
            
            # Resample if needed
            if self.config.sample_rate != target_sr:
                reference_audio = librosa.resample(
                    reference_audio, 
                    orig_sr=self.config.sample_rate, 
                    target_sr=target_sr
                )
                synthesized_audio = librosa.resample(
                    synthesized_audio, 
                    orig_sr=self.config.sample_rate, 
                    target_sr=target_sr
                )
            
            # Normalize audio
            reference_audio = reference_audio / np.max(np.abs(reference_audio))
            synthesized_audio = synthesized_audio / np.max(np.abs(synthesized_audio))
            
            # Match lengths
            min_len = min(len(reference_audio), len(synthesized_audio))
            reference_audio = reference_audio[:min_len]
            synthesized_audio = synthesized_audio[:min_len]
            
            # Calculate PESQ
            pesq_score = pesq(target_sr, reference_audio, synthesized_audio, 'wb')
            
            return pesq_score
        
        except Exception as e:
            logger.error(f"Error calculating PESQ: {e}")
            return None
    
    def calculate_stoi(self, reference_audio, synthesized_audio):
        """Calculate STOI score if available"""
        if not self.stoi_available:
            return None
        
        try:
            from pystoi import stoi
            
            # Ensure the same sample rate
            target_sr = 10000  # STOI works well with this sample rate
            
            # Resample if needed
            if self.config.sample_rate != target_sr:
                reference_audio = librosa.resample(
                    reference_audio, 
                    orig_sr=self.config.sample_rate, 
                    target_sr=target_sr
                )
                synthesized_audio = librosa.resample(
                    synthesized_audio, 
                    orig_sr=self.config.sample_rate, 
                    target_sr=target_sr
                )
            
            # Match lengths
            min_len = min(len(reference_audio), len(synthesized_audio))
            reference_audio = reference_audio[:min_len]
            synthesized_audio = synthesized_audio[:min_len]
            
            # Calculate STOI
            stoi_score = stoi(reference_audio, synthesized_audio, target_sr)
            
            return stoi_score
        
        except Exception as e:
            logger.error(f"Error calculating STOI: {e}")
            return None
    
    def calculate_speaker_similarity(self, reference_audio, synthesized_audio, model):
        """Calculate speaker embedding similarity"""
        try:
            # Convert to tensors
            ref_tensor = torch.tensor(reference_audio).unsqueeze(0).to(self.device)
            syn_tensor = torch.tensor(synthesized_audio).unsqueeze(0).to(self.device)
            
            # Extract speaker embeddings
            with torch.no_grad():
                ref_embedding = model.speaker_encoder(ref_tensor)
                syn_embedding = model.speaker_encoder(syn_tensor)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(ref_embedding, syn_embedding).item()
            
            return similarity
        
        except Exception as e:
            logger.error(f"Error calculating speaker similarity: {e}")
            return 0.0
    
    def evaluate_model(self, model, test_dataset, num_samples=10):
        """Evaluate model on test dataset"""
        if len(test_dataset) == 0:
            logger.error("Empty test dataset")
            return {}
        
        # Set model to eval mode
        model.eval()
        
        # Select samples
        if num_samples >= len(test_dataset):
            indices = list(range(len(test_dataset)))
        else:
            indices = random.sample(range(len(test_dataset)), num_samples)
        
        # Metrics collection
        metrics = {
            'mcd': [],
            'pesq': [],
            'stoi': [],
            'speaker_similarity': []
        }
        
        for idx in tqdm(indices, desc="Evaluating model"):
            try:
                # Get test sample
                sample = test_dataset[idx]
                
                # Synthesize speech
                with torch.no_grad():
                    synthesis_result = model.synthesize(
                        sample["text"],
                        reference_mel=sample["mel_spectrogram"].unsqueeze(0).to(self.device)
                    )
                
                # Get synthesized audio
                synthesized_audio = synthesis_result["audio"]
                
                # Get reference audio
                reference_audio = sample["audio"].cpu().numpy()
                
                # Calculate metrics
                mcd = self.mel_cepstral_distortion(reference_audio, synthesized_audio)
                metrics['mcd'].append(mcd)
                
                # Calculate PESQ if available
                pesq_score = self.calculate_pesq(reference_audio, synthesized_audio)
                if pesq_score is not None:
                    metrics['pesq'].append(pesq_score)
                
                # Calculate STOI if available
                stoi_score = self.calculate_stoi(reference_audio, synthesized_audio)
                if stoi_score is not None:
                    metrics['stoi'].append(stoi_score)
                
                # Calculate speaker similarity
                sim_score = self.calculate_speaker_similarity(
                    reference_audio, synthesized_audio, model
                )
                metrics['speaker_similarity'].append(sim_score)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
        
        # Calculate average metrics
        results = {}
        for key, values in metrics.items():
            if values:
                results[f'avg_{key}'] = np.mean(values)
                results[f'std_{key}'] = np.std(values)
        
        return results

# Add Data Preparation Utilities
def prepare_dataset_from_dir(audio_dir, text_path, output_dir, config, test_split=0.1, val_split=0.1):
    """
    Prepare dataset from directory of audio files and text transcripts
    
    Args:
        audio_dir: Directory containing audio files
        text_path: Path to text transcripts file (format: filename|transcript)
        output_dir: Directory to save processed data
        config: Configuration object
        test_split: Fraction of data for testing
        val_split: Fraction of data for validation
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load text transcripts
        text_data = {}
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|', 1)
                if len(parts) == 2:
                    filename, text = parts
                    text_data[filename] = text
        
        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(glob.glob(os.path.join(audio_dir, f"*{ext}")))
        
        # Match audio files with transcripts
        valid_files = []
        for audio_path in audio_files:
            filename = os.path.basename(audio_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            if filename in text_data:
                valid_files.append((audio_path, text_data[filename], filename))
            elif filename_no_ext in text_data:
                valid_files.append((audio_path, text_data[filename_no_ext], filename))
        
        logger.info(f"Found {len(valid_files)} valid audio files with transcripts")
        
        # Shuffle data
        random.shuffle(valid_files)
        
        # Split data
        test_size = int(len(valid_files) * test_split)
        val_size = int(len(valid_files) * val_split)
        train_size = len(valid_files) - test_size - val_size
        
        train_files = valid_files[:train_size]
        val_files = valid_files[train_size:train_size+val_size]
        test_files = valid_files[train_size+val_size:]
        
        # Create metadata files
        create_metadata_file(train_files, os.path.join(output_dir, "train_metadata.csv"), output_dir)
        create_metadata_file(val_files, os.path.join(output_dir, "val_metadata.csv"), output_dir)
        create_metadata_file(test_files, os.path.join(output_dir, "test_metadata.csv"), output_dir)
        
        logger.info(f"Dataset prepared: {train_size} training, {val_size} validation, {test_size} test samples")
        
        return True
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return False

def create_metadata_file(files, output_path, output_dir):
    """Create metadata CSV file and copy audio files"""
    try:
        # Create audio directory
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Create metadata
        metadata = []
        
        for src_path, text, filename in files:
            # Copy audio file
            dst_path = os.path.join(audio_dir, filename)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            # Add to metadata
            rel_path = os.path.join("audio", filename)
            metadata.append({
                "text": text,
                "audio_path": rel_path
            })
        
        # Write metadata to CSV
        import pandas as pd
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created metadata file with {len(metadata)} entries: {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")
        return False

# Add enhanced command-line interface with data preparation options
def extended_main():
    """Extended main function with additional command-line options"""
    parser = argparse.ArgumentParser(description="Voice Cloning System")
    
    # Top-level subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    common_parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory")
    common_parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output directory")
    common_parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu)")
    common_parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    
    # Train command
    train_parser = subparsers.add_parser("train", parents=[common_parser],
                                         help="Train the voice cloning model")
    train_parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to resume training")
    train_parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training")
    train_parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate for training")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", parents=[common_parser],
                                         help="Run inference with the voice cloning model")
    infer_parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    infer_parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    infer_parser.add_argument("--reference_audio", type=str, required=True,
                        help="Path to reference audio file")
    infer_parser.add_argument("--output_audio", type=str, default=None,
                        help="Path to save synthesized audio")
    infer_parser.add_argument("--visualize", action="store_true",
                        help="Visualize synthesis results")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", parents=[common_parser],
                                        help="Evaluate the voice cloning model")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    eval_parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to evaluate")
    
    # Real-time conversion command
    realtime_parser = subparsers.add_parser("realtime", parents=[common_parser],
                                            help="Run real-time voice conversion")
    realtime_parser.add_argument("--checkpoint", type=str, required=True,
                           help="Path to model checkpoint")
    realtime_parser.add_argument("--reference_audio", type=str, required=True,
                           help="Path to reference audio file")
    realtime_parser.add_argument("--duration", type=int, default=60,
                           help="Duration in seconds for real-time processing")
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser("prepare", parents=[common_parser],
                                           help="Prepare dataset for training")
    prepare_parser.add_argument("--audio_dir", type=str, required=True,
                          help="Directory containing audio files")
    prepare_parser.add_argument("--text_path", type=str, required=True,
                          help="Path to text transcripts file")
    prepare_parser.add_argument("--test_split", type=float, default=0.1,
                          help="Fraction of data for testing")
    prepare_parser.add_argument("--val_split", type=float, default=0.1,
                          help="Fraction of data for validation")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override configuration with command-line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.checkpoint_dir = args.output_dir
    if args.device:
        config.device = args.device
    
    # Create output directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Execute selected command
    if args.command == "train":
        logger.info("Starting training")
        
        # Override additional configuration
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        # Prepare datasets
        train_dataset, val_dataset = prepare_datasets(config)
        
        # Initialize trainer
        trainer = VoiceCloningTrainer(config)
        
        # Load checkpoint if provided
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # Start training
        trainer.train(train_dataset, val_dataset, num_epochs=args.epochs)
    
    elif args.command == "infer":
        logger.info("Starting inference")
        
        # Check if reference audio exists
        if not os.path.exists(args.reference_audio):
            logger.error(f"Reference audio file not found: {args.reference_audio}")
            return
        
        # Set default output path if not provided
        if not args.output_audio:
            output_dir = os.path.join(config.checkpoint_dir, "inference")
            os.makedirs(output_dir, exist_ok=True)
            args.output_audio = os.path.join(output_dir, f"synthesized_{int(time.time())}.wav")
        
        # Initialize inference interface
        interface = InferenceInterface(config, checkpoint_path=args.checkpoint)
        
        # Clone voice
        result = interface.clone_voice(args.text, args.reference_audio, args.output_audio)
        
        if result and args.visualize:
            visualization_path = os.path.splitext(args.output_audio)[0] + "_viz.png"
            interface.visualize_synthesis(result, visualization_path)
        
        if result:
            logger.info(f"Voice cloning completed successfully. Output saved to {args.output_audio}")
            # Play audio if possible
            try:
                sd.play(result["audio"], result["sample_rate"])
                sd.wait()
            except Exception as e:
                logger.warning(f"Could not play audio: {e}")
        else:
            logger.error("Voice cloning failed")
    
    elif args.command == "evaluate":
        logger.info("Starting model evaluation")
        
        # Load test dataset
        test_metadata = os.path.join(config.data_dir, "test_metadata.csv")
        if not os.path.exists(test_metadata):
            logger.error(f"Test metadata not found: {test_metadata}")
            return
        
        audio_processor = ImprovedAudioProcessor(config)
        stats_path = os.path.join(config.data_dir, "norm_stats.npz")
        if os.path.exists(stats_path):
            audio_processor.load_normalization_stats(stats_path)
        
        test_dataset = VoiceCloningDataset(
            config, 
            config.data_dir, 
            test_metadata, 
            processor=audio_processor, 
            is_training=False
        )
        
        # Initialize model
        model = ImprovedVoiceCloningModel(config).to(config.device)
        
        # Load checkpoint
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            return
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize evaluator
        evaluator = VoiceCloningEvaluator(config)
        
        # Run evaluation
        results = evaluator.evaluate_model(model, test_dataset, num_samples=args.num_samples)
        
        # Print results
        logger.info("\nEvaluation Results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save results
        results_path = os.path.join(config.checkpoint_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
    
    elif args.command == "realtime":
        logger.info("Starting real-time voice conversion")
        
        # Initialize real-time converter
        converter = RealTimeVoiceConverter(config, checkpoint_path=args.checkpoint)
        
        # Set reference voice
        if not converter.set_reference_voice(args.reference_audio):
            logger.error("Failed to set reference voice")
            return
        
        # Start real-time processing
        logger.info(f"Starting real-time conversion for {args.duration} seconds...")
        if not converter.process_stream(None, None, duration=args.duration):
            logger.error("Real-time processing failed")
            return
        
        logger.info("Real-time conversion completed")
    
    elif args.command == "prepare":
        logger.info("Starting dataset preparation")
        
        if not os.path.exists(args.audio_dir):
            logger.error(f"Audio directory not found: {args.audio_dir}")
            return
        
        if not os.path.exists(args.text_path):
            logger.error(f"Text transcripts file not found: {args.text_path}")
            return
        
        # Prepare dataset
        success = prepare_dataset_from_dir(
            args.audio_dir,
            args.text_path,
            config.data_dir,
            config,
            test_split=args.test_split,
            val_split=args.val_split
        )
        
        if success:
            logger.info(f"Dataset prepared successfully in {config.data_dir}")
        else:
            logger.error("Dataset preparation failed")
    
    else:
        parser.print_help()

def setup_environment():
    """Setup the environment for training/inference"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Enable cuDNN benchmarking for better performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Set up logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('voice_cloning.log')
        ]
    )

if __name__ == "__main__":
    try:
        setup_environment()
        extended_main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
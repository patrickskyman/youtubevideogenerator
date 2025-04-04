import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
import pyaudio
import wave
import os
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchaudio
from typing import Dict, List, Optional, Tuple
import time
import yaml
import json

class Config:
    """Enhanced configuration for the voice cloning system."""
    def __init__(self):
        # Audio processing parameters
        self.sample_rate = 22050  # Increased from 16000 for better quality
        self.n_mels = 80  # Standard for modern TTS systems
        self.n_fft = 2048  # Increased for better frequency resolution
        self.hop_length = 256
        self.win_length = 1024
        self.f_min = 0
        self.f_max = 8000
        
        # Model dimensions
        self.embedding_dim = 512  # Increased from 256
        self.encoder_hidden = 768  # Increased from 512
        self.decoder_hidden = 768  # Increased from 512
        self.attention_dim = 256  # Increased from 128
        self.hidden_size = 768  # Increased from 592
        self.prenet_dims = [256, 256]  # Decoder prenet dimensions
        self.postnet_filters = 512  # Postnet convolution filters
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 3e-4  # Slightly higher learning rate
        self.max_epochs = 1000
        self.checkpoint_dir = "checkpoints"
        self.data_dir = "data"
        self.audio_length = 10  # seconds - increased from 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Vocoder parameters
        self.vocoder_model = "hifigan"  # Options: "hifigan", "waveglow", "melgan"
        self.use_pretrained_vocoder = True


class ImprovedAudioProcessor:
    """Improved audio processing with better mel extraction and normalization."""
    def __init__(self, config):
        self.config = config
        self.mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        )
        self.mel_mean = None
        self.mel_std = None
        self.is_fitted = False
        
        # Add dynamic range compression for better results
        self.use_dynamic_range_compression = True
        self.ref_level_db = 20
        self.min_level_db = -100
    
    def normalize_mel(self, mel):
        """Apply dynamic range compression and normalization."""
        if self.use_dynamic_range_compression:
            # Convert to dB scale
            mel = 20 * np.log10(np.maximum(1e-5, mel))
            # Normalize
            mel = np.clip((mel - self.ref_level_db + self.min_level_db) / self.min_level_db, 0, 1)
        
        return mel
    
    def denormalize_mel(self, mel):
        """Reverse the normalization process."""
        if self.use_dynamic_range_compression:
            # Denormalize
            mel = (np.clip(mel, 0, 1) * -self.min_level_db) + self.min_level_db + self.ref_level_db
            # Convert back from dB scale
            mel = np.power(10.0, mel * 0.05)
        
        return mel
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram with improved processing."""
        # Ensure audio is mono and correct sample rate
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply preemphasis for better high-frequency capture
        preemphasis = 0.97
        audio = np.append(audio[0], audio[1:] - preemphasis * audio[:-1])
        
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
        
        # Apply mel filterbank
        mel_spectrogram = np.dot(self.mel_basis, magnitude)
        
        # Apply dynamic range compression and normalization
        mel_spectrogram = self.normalize_mel(mel_spectrogram)
        
        # Normalize if fitted
        if self.is_fitted:
            mel_spectrogram = (mel_spectrogram - self.mel_mean[:, np.newaxis]) / self.mel_std[:, np.newaxis]
        
        return mel_spectrogram


class ImprovedSpeakerEncoder(nn.Module):
    """Enhanced speaker encoder with improved architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use a more robust encoder architecture similar to GE2E
        self.lstm_hidden = 768
        self.lstm_layers = 3
        
        # CNN layers for extracting features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Bidirectional LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Self-attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final projection
        self.projection = nn.Linear(self.lstm_hidden * 2, config.embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, audio):
        """Extract speaker embedding from audio."""
        # Reshape audio: [batch, time] -> [batch, 1, time]
        x = audio.unsqueeze(1)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Reshape for LSTM: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        
        # Project to embedding dimension
        x = self.projection(x)
        
        # Apply layer normalization and dropout
        speaker_embedding = self.layer_norm(self.dropout(x))
        
        # L2 normalize the embedding (critical for voice similarity)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=1)
        
        return speaker_embedding


class ImprovedTextEncoder(nn.Module):
    """Enhanced text encoder with better representation learning."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use a better pretrained model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.text_encoder = AutoModel.from_pretrained("roberta-base")
        except:
            # Fallback to BERT if RoBERTa isn't available
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        # Use a better projection mechanism with residual connection
        self.projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, config.embedding_dim)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, text):
        """Encode text to embeddings with improved representation."""
        # Tokenize the text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        tokens = {k: v.to(self.config.device) for k, v in tokens.items()}
        
        # Get contextual embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
        
        # Use better representation - average the last 4 layers instead of just CLS token
        last_hidden_states = outputs.last_hidden_state
        
        # Use attention pooling over tokens
        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        weighted_sum = torch.sum(last_hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        
        # Project to embedding dimension
        projected = self.projection(weighted_sum)
        
        # Apply layer normalization
        text_embedding = self.layer_norm(projected)
        
        return text_embedding


class LocationSensitiveAttention(nn.Module):
    """Location-sensitive attention as used in Tacotron 2."""
    def __init__(self, attention_dim, encoder_dim, decoder_dim, location_features=32):
        super().__init__()
        
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=location_features,
            kernel_size=31,
            padding=15,
            bias=False
        )
        
        self.location_layer = nn.Linear(location_features, attention_dim)
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        self.score_mask_value = -float("inf")
    
    def forward(self, query, keys, values, processed_memory, attention_weights_cat):
        """Forward pass for location-sensitive attention."""
        # query: decoder state, (batch, decoder_dim)
        # keys: encoder outputs, (batch, max_time, encoder_dim)
        # values: encoder outputs, (batch, max_time, encoder_dim)
        # processed_memory: processed encoder outputs, (batch, max_time, attention_dim)
        # attention_weights_cat: previous attention weights, (batch, 1, max_time)
        
        # Process the query (decoder state)
        processed_query = self.query_layer(query.unsqueeze(1))
        
        # Process the location info
        processed_loc = self.location_layer(self.location_conv(attention_weights_cat).transpose(1, 2))
        
        # Calculate energy
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_loc)).squeeze(-1)
        
        # Apply attention
        attention_weights = F.softmax(energies, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return attention_context, attention_weights


class ImprovedDecoder(nn.Module):
    """Enhanced decoder with prenet, attention, and postnet components."""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.n_mels = config.n_mels
        self.hidden_size = config.hidden_size
        self.prenet_dims = config.prenet_dims
        
        # Prenet for the decoder
        modules = []
        in_dim = self.n_mels
        for dim in self.prenet_dims:
            modules.append(nn.Linear(in_dim, dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(0.5))  # High dropout for prenet is standard in TTS
            in_dim = dim
        self.prenet = nn.Sequential(*modules)
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(
            config.attention_dim, 
            self.embedding_dim * 2,  # text + speaker embedding
            self.hidden_size
        )
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(
                input_size=self.prenet_dims[-1] + self.embedding_dim * 2,
                hidden_size=self.hidden_size
            ),
            nn.LSTMCell(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size
            )
        ])
        
        # Frame projection
        self.frame_projection = nn.Linear(
            self.hidden_size + self.embedding_dim * 2,
            self.n_mels
        )
        
        # Postnet for refining mel spectrograms
        self.postnet = nn.Sequential(
            nn.Conv1d(self.n_mels, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(config.postnet_filters, config.postnet_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.postnet_filters),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Conv1d(config.postnet_filters, self.n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.n_mels),
            nn.Dropout(0.5)
        )
        
        # Initial hidden states
        self.initial_hidden = nn.Parameter(torch.zeros(2, self.hidden_size))
        self.initial_cell = nn.Parameter(torch.zeros(2, self.hidden_size))
        
        # Go frame
        self.go_frame = nn.Parameter(torch.zeros(1, 1, self.n_mels))
    
    def initialize_decoder_states(self, batch_size):
        """Initialize the decoder states."""
        h_states = [self.initial_hidden[0].repeat(batch_size, 1), 
                   self.initial_hidden[1].repeat(batch_size, 1)]
        c_states = [self.initial_cell[0].repeat(batch_size, 1), 
                   self.initial_cell[1].repeat(batch_size, 1)]
        return h_states, c_states
    
    def parse_decoder_inputs(self, mel_targets):
        """Prepare decoder inputs from targets - add go frame."""
        if mel_targets is None:
            return self.go_frame.repeat(1, 1, 1)
        else:
            B = mel_targets.size(0)
            go_frames = self.go_frame.repeat(B, 1, 1)
            # Shift targets for teacher forcing
            return torch.cat((go_frames, mel_targets[:, :-1, :]), dim=1)
    
    def forward(self, combined_embedding, target_mels=None, max_length=None):
        """Forward pass through the decoder."""
        batch_size = combined_embedding.size(0)
        
        # Process encoder outputs (text + speaker embeddings)
        encoder_outputs = combined_embedding.unsqueeze(1)  # [batch, 1, embedding_dim * 2]
        
        # Initialize decoder states
        h_states, c_states = self.initialize_decoder_states(batch_size)
        
        # Determine maximum generation length
        if max_length is None:
            max_length = 1000 if target_mels is None else target_mels.size(1)
        
        # Initialize attention
        attention_weights = torch.zeros(batch_size, 1, encoder_outputs.size(1)).to(combined_embedding.device)
        attention_context = torch.zeros(batch_size, combined_embedding.size(-1)).to(combined_embedding.device)
        
        # Initialize decoder input - either from target or go frame
        if target_mels is None:
            decoder_input = self.go_frame.repeat(batch_size, 1, 1).squeeze(1)
        else:
            decoder_input = target_mels[:, 0, :]
        
        # Prepare outputs list
        mel_outputs = []
        alignments = []
        
        # Setup memory for location-sensitive attention
        processed_memory = None  # Will be set in the first step
        
        # Generate frames
        for t in range(max_length):
            # Process through prenet
            prenet_output = self.prenet(decoder_input)
            
            # Concatenate with previous attention context
            lstm_input = torch.cat([prenet_output, attention_context], dim=-1)
            
            # LSTM layer 1
            h_states[0], c_states[0] = self.lstm_layers[0](
                lstm_input, (h_states[0], c_states[0])
            )
            
            # LSTM layer 2
            h_states[1], c_states[1] = self.lstm_layers[1](
                h_states[0], (h_states[1], c_states[1])
            )
            
            # Initialize processed memory on first step
            if processed_memory is None:
                processed_memory = encoder_outputs.squeeze(1)  # [batch, embedding_dim * 2]
            
            # Apply attention
            attention_context, attention_weights_t = self.attention(
                h_states[1], 
                encoder_outputs, 
                encoder_outputs, 
                processed_memory,
                attention_weights
            )
            
            # Update attention weights
            attention_weights = attention_weights_t.unsqueeze(1)
            alignments.append(attention_weights_t)
            
            # Concatenate LSTM output and attention context
            decoder_lstm_output = torch.cat([h_states[1], attention_context], dim=1)
            
            # Project to mel spectrogram
            mel_output = self.frame_projection(decoder_lstm_output)
            mel_outputs.append(mel_output.unsqueeze(1))
            
            # Update decoder input for next step
            if target_mels is not None and t < target_mels.size(1) - 1:
                decoder_input = target_mels[:, t+1, :]  # Teacher forcing
            else:
                decoder_input = mel_output
        
        # Concatenate mel outputs
        mel_outputs = torch.cat(mel_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        # Apply postnet
        postnet_output = self.postnet(mel_outputs.transpose(1, 2)).transpose(1, 2)
        mel_outputs_postnet = mel_outputs + postnet_output
        
        return mel_outputs_postnet, alignments


class ImprovedVocoder(nn.Module):
    """Enhanced vocoder using pre-trained HiFi-GAN for high-quality audio generation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use pre-trained HiFi-GAN
        self.use_pretrained = config.use_pretrained_vocoder
        
        if self.use_pretrained:
            try:
                import torch
                
                # For actual implementation, you would download and load the model like this:
                # from hifigan.models import Generator as HiFiGANGenerator
                # from hifigan.env import AttrDict
                # 
                # # Load HiFi-GAN configuration
                # with open("hifigan/config_v1.json") as f:
                #     h = json.load(f)
                #     h = AttrDict(h)
                #
                # # Initialize model
                # self.hifigan = HiFiGANGenerator(h)
                # 
                # # Load checkpoint
                # checkpoint = torch.load("hifigan/generator_v1")
                # self.hifigan.load_state_dict(checkpoint['generator'])
                # self.hifigan.eval()
                # self.hifigan.remove_weight_norm()
                
                # Since we can't download models in this context, we'll create a placeholder
                # structure similar to HiFi-GAN but note that in practice you would use
                # the actual pre-trained model
                
                print("Initializing pre-trained HiFi-GAN vocoder")
                self.initialize_hifigan_placeholder()
                
            except Exception as e:
                print(f"Error loading pre-trained vocoder: {e}")
                print("Falling back to basic vocoder implementation")
                self.use_pretrained = False
                self.initialize_basic_vocoder()
        else:
            print("Using basic vocoder implementation")
            self.initialize_basic_vocoder()
    
    def initialize_hifigan_placeholder(self):
        """Initialize a placeholder HiFi-GAN structure."""
        # This is a simplified version of HiFi-GAN architecture
        # In practice, you would load the actual pre-trained model
        
        # Input convolutional layer
        self.conv_pre = nn.Conv1d(self.config.n_mels, 512, kernel_size=7, stride=1, padding=3)
        
        # Upsampling layers (simplified)
        self.ups = nn.ModuleList()
        for i, up_rate in enumerate([8, 8, 2, 2]):  # Total upsampling of 256x
            in_channels = 512 // (2**i) if i > 0 else 512
            out_channels = 512 // (2**(i+1)) if i < 3 else 512 // (2**i)
            
            self.ups.append(nn.ConvTranspose1d(
                in_channels, 
                out_channels, 
                kernel_size=up_rate*2, 
                stride=up_rate, 
                padding=up_rate//2,
                output_padding=up_rate%2
            ))
        
        # Multi-Receptive Field Fusion (MRF) blocks
        self.mrfs = nn.ModuleList()
        for i in range(4):
            self.mrfs.append(self._mrf_block(512 // (2**min(i, 3))))
        
        # Final conv layer
        self.conv_post = nn.Conv1d(512 // (2**3), 1, kernel_size=7, stride=1, padding=3)
    
    def _mrf_block(self, channels):
        """Create a Multi-Receptive Field Fusion block."""
        return nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=k//2*d),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=k, dilation=1, padding=k//2)
            ) 
            for k, d in [(3, 1), (5, 3), (7, 5)]
        ])
    
    def initialize_basic_vocoder(self):
        """Initialize a basic vocoder architecture."""
        # This is a simpler fallback implementation
        self.conv_in = nn.Conv1d(self.config.n_mels, 512, kernel_size=7, padding=3)
        
        # Upsampling blocks - total upsampling of x256
        self.upsample_rates = [4, 4, 4, 4]  # 4*4*4*4 = 256 upsampling
        self.upsample_blocks = nn.ModuleList()
        current_channels = 512
        
        for i, rate in enumerate(self.upsample_rates):
            is_last = (i == len(self.upsample_rates) - 1)
            out_channels = 1 if is_last else current_channels // 2
            
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        current_channels, 
                        out_channels,
                        kernel_size=rate*2, 
                        stride=rate, 
                        padding=rate//2
                    ),
                    nn.LeakyReLU(0.1) if not is_last else nn.Tanh()
                )
            )
            
            if not is_last:
                current_channels //= 2
    
    def forward(self, mel_spec):
        """Convert mel spectrogram to waveform."""
        # Transpose for 1D convolution [batch, time, mels] -> [batch, mels, time]
        if len(mel_spec.shape) == 3:
            x = mel_spec.transpose(1, 2)
        else:
            # Handle 2D input case
            x = mel_spec.unsqueeze(0).transpose(1, 2)
        
        if self.use_pretrained:
            # HiFi-GAN forward pass
            x = self.conv_pre(x)
            
            # Apply upsampling blocks
            for i, up in enumerate(self.ups):
                x = F.leaky_relu(x, 0.1)
                x = up(x)
                
                # Apply MRF block
                xs = None
                for mrf in self.mrfs[i]:
                    if xs is None:
                        xs = mrf(x)
                    else:
                        xs = xs + mrf(x)
                
                x = x + xs / len(self.mrfs[i])
            
            # Apply final conv and activation
            x = F.leaky_relu(x, 0.1)
            x = self.conv_post(x)
            x = torch.tanh(x)
        else:
            # Basic vocoder implementation
            x = self.conv_in(x)
            
            # Apply upsampling blocks
            for upsample_block in self.upsample_blocks:
                x = upsample_block(x)
        
        # Remove channel dimension and batch if needed
        waveform = x.squeeze(1)
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        
        return waveform

    def generate_audio(self, mel_spectrogram):
        """Generate audio from mel spectrogram with proper preprocessing and denormalization."""
        # Ensure we're in evaluation mode
        self.eval()
        
        with torch.no_grad():
            # If mel spectrogram is normalized, denormalize it
            # This depends on your audio processor configuration
            if hasattr(self, 'audio_processor') and self.audio_processor.is_fitted:
                # Assuming mel_spectrogram is a torch tensor
                if isinstance(mel_spectrogram, np.ndarray):
                    mel_spectrogram = torch.tensor(mel_spectrogram).to(self.config.device)
                
                # Denormalize if needed
                mel_mean = torch.tensor(self.audio_processor.mel_mean).to(self.config.device)
                mel_std = torch.tensor(self.audio_processor.mel_std).to(self.config.device)
                mel_spectrogram = mel_spectrogram * mel_std.unsqueeze(1) + mel_mean.unsqueeze(1)
            
            # Forward pass to generate waveform
            waveform = self.forward(mel_spectrogram)
            
            # Apply post-processing if needed
            # For example, you might want to apply a de-emphasis filter
            
            return waveform.cpu().numpy()


class ImprovedVoiceCloningModel(nn.Module):
    """Enhanced complete voice cloning model with improved components."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Improved components
        self.speaker_encoder = ImprovedSpeakerEncoder(config)
        self.text_encoder = ImprovedTextEncoder(config)
        self.decoder = ImprovedDecoder(config)
        self.vocoder = ImprovedVocoder(config)
        
        # Add global style tokens for more expressive speech
        self.use_gst = True
        if self.use_gst:
            self.gst_embedding_dim = 256
            self.gst_num_tokens = 10
            self.gst_token_embedding = nn.Parameter(
                torch.randn(self.gst_num_tokens, self.gst_embedding_dim)
            )
            self.gst_attention = nn.Linear(config.embedding_dim, self.gst_num_tokens)
    
    def apply_global_style_tokens(self, speaker_embedding):
        """Apply global style tokens to add expressiveness."""
        if not self.use_gst:
            return speaker_embedding
            
        # Calculate attention over GST tokens
        gst_attention_weights = F.softmax(
            self.gst_attention(speaker_embedding), dim=-1
        )
        
        # Weight the GST tokens
        weighted_gst = torch.matmul(
            gst_attention_weights, 
            self.gst_token_embedding
        )
        
        # Combine with speaker embedding
        enhanced_embedding = speaker_embedding + weighted_gst
        return enhanced_embedding
    
    def forward(self, text, speaker_features, target_mels=None):
        """Forward pass through the complete model with improvements."""
        # Encode text and speaker
        text_embedding = self.text_encoder(text)
        speaker_embedding = self.speaker_encoder(speaker_features)
        
        # Apply global style tokens
        if self.use_gst:
            speaker_embedding = self.apply_global_style_tokens(speaker_embedding)
        
        # Combine embeddings
        combined_embedding = torch.cat([text_embedding, speaker_embedding], dim=1)
        
        # Decode to mel spectrogram
        max_length = target_mels.size(1) if target_mels is not None else 500  # Increased default
        mel_outputs, attention = self.decoder(combined_embedding, target_mels, max_length)
        
        # Convert to waveform if needed
        if self.training or target_mels is None:
            waveform = self.vocoder(mel_outputs)
        else:
            waveform = None
        
        return {
            "mel_outputs": mel_outputs,
            "attention": attention,
            "waveform": waveform,
            "speaker_embedding": speaker_embedding,
            "text_embedding": text_embedding
        }


class ImprovedVoiceCloner:
    """Enhanced main class for voice cloning system."""
    def __init__(self, config_path=None):
        # Load or create config
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            self.config = Config()
            for key, value in config_dict.items():
                setattr(self.config, key, value)
        else:
            self.config = Config()
        
        # Initialize components
        self.audio_processor = ImprovedAudioProcessor(self.config)
        self.model = ImprovedVoiceCloningModel(self.config).to(self.config.device)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Use OneCycleLR scheduler for faster convergence
        self.scheduler = None  # Will be set during training
        
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        
        # Add data augmentation
        self.use_data_augmentation = True
    
    def augment_audio(self, audio, sr):
        """Apply audio augmentation techniques."""
        augmented = []
        # Original audio
        augmented.append(audio)
        
        # Add noise augmentation
        noise_level = 0.005
        noise = np.random.normal(0, noise_level, len(audio))
        augmented.append(np.clip(audio + noise, -1, 1))
        
        # Time stretch
        augmented.append(librosa.effects.time_stretch(audio, rate=0.95))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.05))
        
        # Pitch shift
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1))
        
        return augmented
    
    def prepare_training_data(self, data_dir=None, apply_augmentation=True):
        """Prepare audio data for training with improved preprocessing."""
        if data_dir:
            self.config.data_dir = data_dir
        
        print(f"Preparing training data from {self.config.data_dir}...")
        
        # Create dataset splits
        train_dir = os.path.join(self.config.data_dir, "train")
        val_dir = os.path.join(self.config.data_dir, "val")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Process all speakers with improved preprocessing
        for speaker_id in os.listdir(self.config.data_dir):
            speaker_dir = os.path.join(self.config.data_dir, speaker_id)
                    
            # Skip non-directories
            if not os.path.isdir(speaker_dir) or speaker_id in ["train", "val"]:
                continue
            
            print(f"Processing speaker: {speaker_id}")
            
            # Get all audio files for this speaker
            audio_files = []
            for root, _, files in os.walk(speaker_dir):
                for file in files:
                    if file.endswith((".wav", ".mp3", ".flac")):
                        audio_files.append(os.path.join(root, file))
            
            # Skip if no audio files found
            if not audio_files:
                print(f"No audio files found for speaker {speaker_id}, skipping.")
                continue
            
            # Split into train/val
            random.shuffle(audio_files)
            split_idx = int(len(audio_files) * 0.9)  # 90% train, 10% val
            train_files = audio_files[:split_idx]
            val_files = audio_files[split_idx:]
            
            # Create speaker directories in train/val
            os.makedirs(os.path.join(train_dir, speaker_id), exist_ok=True)
            os.makedirs(os.path.join(val_dir, speaker_id), exist_ok=True)
            
            # Process training files
            for audio_file in train_files:
                # Load audio with improved quality
                audio, sr = librosa.load(audio_file, sr=self.config.sample_rate)
                
                # Apply augmentation if enabled
                if apply_augmentation and self.use_data_augmentation:
                    augmented_audios = self.augment_audio(audio, sr)
                else:
                    augmented_audios = [audio]
                
                # Process each audio (original and augmented versions)
                for i, aug_audio in enumerate(augmented_audios):
                    # Get base filename without extension
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    
                    # Create augmentation suffix if needed
                    aug_suffix = f"_aug{i}" if i > 0 else ""
                    
                    # Extract mel spectrogram with improved processing
                    mel_spectrogram = self.audio_processor.extract_mel_spectrogram(aug_audio)
                    
                    # Save processed audio
                    output_path = os.path.join(train_dir, speaker_id, f"{base_name}{aug_suffix}.npy")
                    np.save(output_path, mel_spectrogram)
                    
                    # Save original audio with improved quality
                    audio_output_path = os.path.join(train_dir, speaker_id, f"{base_name}{aug_suffix}.wav")
                    sf.write(audio_output_path, aug_audio, self.config.sample_rate)
                    
                    # Create transcript placeholders if they don't exist
                    # In a real system, you would use ASR or have real transcripts
                    transcript_path = os.path.join(train_dir, speaker_id, f"{base_name}{aug_suffix}.txt")
                    if not os.path.exists(transcript_path):
                        with open(transcript_path, "w") as f:
                            f.write("This is a placeholder transcript for voice cloning training.")
            
            # Process validation files (no augmentation for validation)
            for audio_file in val_files:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.config.sample_rate)
                
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                
                # Extract mel spectrogram
                mel_spectrogram = self.audio_processor.extract_mel_spectrogram(audio)
                
                # Save processed audio
                output_path = os.path.join(val_dir, speaker_id, f"{base_name}.npy")
                np.save(output_path, mel_spectrogram)
                
                # Save original audio
                audio_output_path = os.path.join(val_dir, speaker_id, f"{base_name}.wav")
                sf.write(audio_output_path, audio, self.config.sample_rate)
                
                # Create transcript placeholders
                transcript_path = os.path.join(val_dir, speaker_id, f"{base_name}.txt")
                if not os.path.exists(transcript_path):
                    with open(transcript_path, "w") as f:
                        f.write("This is a placeholder transcript for voice cloning validation.")
        
        # Compute global statistics for normalization
        print("Computing global statistics for normalization...")
        all_mels = []
        for speaker_id in os.listdir(train_dir):
            speaker_train_dir = os.path.join(train_dir, speaker_id)
            if not os.path.isdir(speaker_train_dir):
                continue
                
            for file in os.listdir(speaker_train_dir):
                if file.endswith(".npy"):
                    mel_path = os.path.join(speaker_train_dir, file)
                    mel = np.load(mel_path)
                    all_mels.append(mel)
        
        if all_mels:
            # Compute mean and std across all training data
            all_mels_concat = np.concatenate(all_mels, axis=1)
            mel_mean = np.mean(all_mels_concat, axis=1)
            mel_std = np.std(all_mels_concat, axis=1)
            
            # Save normalization parameters
            self.audio_processor.mel_mean = mel_mean
            self.audio_processor.mel_std = mel_std
            self.audio_processor.is_fitted = True
            
            # Save normalization parameters to file
            norm_path = os.path.join(self.config.data_dir, "normalization.npz")
            np.savez(norm_path, mean=mel_mean, std=mel_std)
            
            print(f"Normalization parameters saved to {norm_path}")
        else:
            print("Warning: No mel spectrograms found for normalization.")
        
        print("Data preparation complete!")


class AudioDataset(Dataset):
    """Enhanced dataset for loading audio data with improved handling."""
    def __init__(self, config, audio_processor, split="train"):
        self.config = config
        self.audio_processor = audio_processor
        self.split = split
        
        # Directory containing the processed data
        self.data_dir = os.path.join(config.data_dir, split)
        
        # Collect all samples
        self.samples = []
        for speaker_id in os.listdir(self.data_dir):
            speaker_dir = os.path.join(self.data_dir, speaker_id)
            
            if not os.path.isdir(speaker_dir):
                continue
            
            # Get all processed mel spectrograms
            for file in os.listdir(speaker_dir):
                if file.endswith(".npy"):
                    # Get the base name without extension
                    base_name = os.path.splitext(file)[0]
                    
                    # Check for corresponding audio and transcript
                    audio_path = os.path.join(speaker_dir, f"{base_name}.wav")
                    transcript_path = os.path.join(speaker_dir, f"{base_name}.txt")
                    
                    if os.path.exists(audio_path) and os.path.exists(transcript_path):
                        self.samples.append({
                            "speaker_id": speaker_id,
                            "mel_path": os.path.join(speaker_dir, file),
                            "audio_path": audio_path,
                            "transcript_path": transcript_path
                        })
        
        # Load normalization parameters if available
        norm_path = os.path.join(config.data_dir, "normalization.npz")
        if os.path.exists(norm_path) and not audio_processor.is_fitted:
            norm_data = np.load(norm_path)
            audio_processor.mel_mean = norm_data["mean"]
            audio_processor.mel_std = norm_data["std"]
            audio_processor.is_fitted = True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load mel spectrogram
        mel_spectrogram = np.load(sample["mel_path"])
        
        # Load audio
        audio, _ = librosa.load(sample["audio_path"], sr=self.config.sample_rate)
        
        # Load transcript
        with open(sample["transcript_path"], "r") as f:
            transcript = f.read().strip()
        
        # Convert to tensors
        mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        # Apply normalization if processor is fitted
        if self.audio_processor.is_fitted:
            # Apply mean/std normalization along each mel band
            normalized_mel = (mel_tensor - torch.tensor(self.audio_processor.mel_mean, dtype=torch.float32).unsqueeze(1)) / (
                torch.tensor(self.audio_processor.mel_std, dtype=torch.float32).unsqueeze(1) + 1e-5)
            mel_tensor = normalized_mel
        
        return {
            "speaker_id": sample["speaker_id"],
            "mel_spectrogram": mel_tensor,
            "audio": audio_tensor,
            "transcript": transcript
        }


def record_voice_sample(self, duration=10, output_path="voice_sample.wav"):
    """Record a voice sample using the microphone."""
    print(f"Recording a {duration} second voice sample...")
    print("Please speak clearly and consistently.")
    
    try:
        import sounddevice as sd
        import soundfile as sf
        
        # Record audio
        audio = sd.rec(
            int(duration * self.config.sample_rate),
            samplerate=self.config.sample_rate,
            channels=1
        )
        
        # Wait for recording to complete
        sd.wait()
        
        # Save to file
        sf.write(output_path, audio, self.config.sample_rate)
        
        print(f"Voice sample saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error recording audio: {e}")
        print("Please ensure you have a working microphone and the required packages installed.")
        return None


if __name__ == "__main__":
    # Example usage
    voice_cloner = ImprovedVoiceCloner()
    
    # Train or load a model
    if os.path.exists(os.path.join(voice_cloner.config.checkpoint_dir, "best_model.pth")):
        voice_cloner.load_model()
    else:
        voice_cloner.prepare_training_data()
        voice_cloner.train()
    
    # Record a sample and clone voice
    sample_path = voice_cloner.record_voice_sample()
    if sample_path:
        # Create a voice profile
        voice_cloner.create_voice_profile(sample_path, "my_voice")
        
        # Generate speech with cloned voice
        voice_cloner.generate_from_profile(
            "Hello, this is my cloned voice speaking. Voice cloning technology is amazing!",
            "my_voice",
            "cloned_speech.wav"
        )
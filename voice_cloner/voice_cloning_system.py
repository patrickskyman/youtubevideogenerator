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
    """Configuration for the voice cloning system."""
    def __init__(self):
        self.sample_rate = 16000
        self.n_mels = 251  # Update this to 251 based on your output
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.f_min = 0
        self.f_max = 8000
        self.embedding_dim = 256
        self.encoder_hidden = 512
        self.decoder_hidden = 512
        self.attention_dim = 128
        self.hidden_size = 592 
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.max_epochs = 1000
        self.checkpoint_dir = "checkpoints"
        self.data_dir = "data"
        self.audio_length = 4  # seconds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class AudioProcessor:
    """Processes audio data for training and inference."""
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
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram from audio data."""
        # Ensure audio is mono and correct sample rate
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        #print(f"Audio shape: {audio.shape}")
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
       # print(f"STFT shape: {stft.shape}")
        magnitude = np.abs(stft) ** 2
        #print(f"Magnitude shape: {magnitude.shape}")
        mel_spectrogram = np.dot(self.mel_basis, magnitude)
       # print(f"Mel spectrogram shape before log: {mel_spectrogram.shape}")
        log_mel_spectrogram = np.log10(np.maximum(mel_spectrogram, 1e-5))
       # print(f"Log mel spectrogram shape: {log_mel_spectrogram.shape}")
       # return log_mel_spectrogram
        
        # Convert to power spectrogram
        magnitude = np.abs(stft) ** 2
        
        # Apply mel filterbank
        mel_spectrogram = np.dot(self.mel_basis, magnitude)
        
        # Convert to log scale
        log_mel_spectrogram = np.log10(np.maximum(mel_spectrogram, 1e-5))
        
        # Normalize if fitted
        if self.is_fitted:
            # Broadcasting: apply mean and std across frequency dimension only
            # Reshape for proper broadcasting
            log_mel_spectrogram = log_mel_spectrogram - self.mel_mean[:, np.newaxis]
            log_mel_spectrogram = log_mel_spectrogram / self.mel_std[:, np.newaxis]
        
        return log_mel_spectrogram
    
    def fit_normalizer(self, mel_spectrograms):
        """Fit normalizer to mel spectrograms."""
        if not mel_spectrograms:
            print("No mel spectrograms provided for normalization")
            return
            
        # Ensure consistent shape (frequency dimension should be the same)
        freq_dim = mel_spectrograms[0].shape[0]  # First dimension is frequency bins
        
        # Flatten all time steps across all spectrograms
        all_values = []
        for spec in mel_spectrograms:
            if spec.shape[0] != freq_dim:
                print(f"Warning: Inconsistent mel spectrogram shape: {spec.shape}, expected first dim {freq_dim}")
                continue
            all_values.append(spec.reshape(-1, freq_dim))
        
        if not all_values:
            print("No valid mel spectrograms after shape filtering")
            return
            
        # Concatenate along time dimension
        concat_mels = np.vstack(all_values)
        
        # Calculate mean and std across time steps
        self.mel_mean = np.mean(concat_mels, axis=0)
        self.mel_std = np.std(concat_mels, axis=0)
        
        # Prevent division by zero
        self.mel_std = np.maximum(self.mel_std, 1e-5)
        
        self.is_fitted = True
        print(f"Normalizer fitted with {len(mel_spectrograms)} spectrograms, mean shape: {self.mel_mean.shape}")
    
    def audio_to_mel(self, audio_path):
        """Convert audio file to mel spectrogram."""
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        return self.extract_mel_spectrogram(audio)
    
    def save_audio(self, audio, filename):
        """Save audio data to file."""
        sf.write(filename, audio, self.config.sample_rate)

class SpeakerEncoder(nn.Module):
    """Encoder for extracting speaker embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create a CNN-based speaker encoder instead of using pretrained model
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Bidirectional LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final projection
        self.projection = nn.Linear(512, config.embedding_dim)
    
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
        attn_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)
        x = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        # Project to embedding dimension
        speaker_embedding = self.projection(x)
        
        return speaker_embedding

class TextEncoder(nn.Module):
    """Encoder for processing text input."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use BERT for text encoding
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        # Project to our embedding dimension
        self.projection = nn.Linear(768, config.embedding_dim)
    
    def forward(self, text):
        """Encode text to embeddings."""
        # Tokenize the text
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.config.device) for k, v in tokens.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
        
        # Use CLS token as text representation
        text_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to our embedding dimension
        projected = self.projection(text_embedding)
        return projected

class AttentionModule(nn.Module):
    """Attention mechanism for the decoder."""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_outputs, decoder_hidden):
        """Calculate attention weights."""
        # encoder_outputs: (batch_size, seq_len, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)
        
        # Expand decoder hidden state
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        
        # Calculate attention
        encoder_attn = self.encoder_attn(encoder_outputs)
        decoder_attn = self.decoder_attn(decoder_hidden)
        combined = torch.tanh(encoder_attn + decoder_attn)
        
        # Get attention weights
        attention = self.full_attn(combined).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        
        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding_dim = config.embedding_dim  # Usually 256
        self.n_mels = config.n_mels  # Now 251
        self.hidden_size = config.hidden_size  # 592
        
        # Input size is embedding_dim * 2 (text + speaker) + n_mels (for teacher forcing)
        self.lstm_input_size = (self.embedding_dim * 2) + self.n_mels  # Should be 256 * 2 + 251 = 763
        
        # Projection layers
        self.input_projection = nn.Linear(self.lstm_input_size, self.hidden_size)  # Now 763 -> 592
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        
        # Output projection to mel spectrogram
        self.output_projection = nn.Linear(self.hidden_size, self.n_mels)  # 592 -> 251
        
        # Initial mel token
        self.initial_mel_token = nn.Parameter(torch.randn(1, 1, self.n_mels))  # Shape [1, 1, 251]
    
    def forward(self, combined_embedding, target_mels=None, max_length=None):
        batch_size = combined_embedding.size(0)
        
        print(f"combined_embedding shape: {combined_embedding.shape}")  # Should be [batch_size, embedding_dim * 2]
        print(f"target_mels shape: {target_mels.shape if target_mels is not None else 'None'}")

        # Initialize with learned mel token if no target mels
        if target_mels is None:
            current_mel = self.initial_mel_token.repeat(batch_size, 1, 1)
        else:
            current_mel = target_mels[:, 0:1, :]  # Take first frame
        
        print(f"current_mel shape: {current_mel.shape}")  # Should be [batch_size, 1, n_mels]

        # Initial input combines embeddings with mel frame
        current_input = torch.cat([
            combined_embedding.unsqueeze(1).repeat(1, 1, 1),  # [batch_size, 1, embedding_dim * 2]
            current_mel  # [batch_size, 1, n_mels]
        ], dim=-1)  # Concatenate along the last dimension
        
        print(f"current_input shape before projection: {current_input.shape}")  # Should be [batch_size, 1, (embedding_dim * 2) + n_mels]

        # Project input to correct dimension
        projected_input = self.input_projection(current_input.squeeze(1))  # Remove time dimension for LSTM

        # The rest of your forward method...
        # Initialize hidden state
        hidden = (
            torch.zeros(2, batch_size, self.hidden_size).to(combined_embedding.device),
            torch.zeros(2, batch_size, self.hidden_size).to(combined_embedding.device)
        )
        
        # Set max length
        if max_length is None:
            max_length = 1000 if target_mels is None else target_mels.size(1)
            
        mel_outputs = []
        attention_weights = []
        
        # Generate frames
        for t in range(max_length):
            # LSTM step
            output, hidden = self.lstm(projected_input.unsqueeze(1), hidden)  # Add back time dimension
            
            # Project to mel spectrogram
            mel_output = self.output_projection(output)
            mel_outputs.append(mel_output)
            
            # Teacher forcing during training
            if target_mels is not None and t < target_mels.size(1) - 1:
                current_mel = target_mels[:, t+1:t+2, :]
            else:
                current_mel = mel_output
                
            # Prepare next input
            next_input = torch.cat([
                combined_embedding.unsqueeze(1).repeat(1, 1, 1),
                current_mel
            ], dim=-1)
            projected_input = self.input_projection(next_input.squeeze(1))
        
        mel_outputs = torch.cat(mel_outputs, dim=1)
        return mel_outputs, None  # Return None for attention as we're not using it
    
class Vocoder(nn.Module):
    """Converts mel spectrograms back to audio waveforms."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use HiFi-GAN architecture for mel to waveform conversion
        # Simplified implementation for brevity
        self.conv_in = nn.Conv1d(config.n_mels, 512, kernel_size=7, padding=3)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(512, 512, kernel_size=3, dilation=2**i, padding=2**i),
                nn.LeakyReLU(0.1),
                nn.Conv1d(512, 512, kernel_size=3, dilation=1, padding=1)
            ) for i in range(4)
        ])
        
        # Upsampling blocks
        self.upsample_rates = [4, 4, 2, 2]  # Total upsampling of 64x
        self.upsample_blocks = nn.ModuleList()
        current_channels = 512
        
        for rate in self.upsample_rates:
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        current_channels, current_channels // 2,
                        kernel_size=rate*2, stride=rate, padding=rate//2
                    ),
                    nn.LeakyReLU(0.1)
                )
            )
            current_channels //= 2
        
        # Final convolution
        self.final_conv = nn.Conv1d(current_channels, 1, kernel_size=7, padding=3)
    
    def forward(self, mel_spec):
        """Convert mel spectrogram to waveform."""
        # Transpose for 1D convolution
        x = mel_spec.transpose(1, 2)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = x + res_block(x)
        
        # Upsampling
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Remove channel dimension
        waveform = x.squeeze(1)
        
        return waveform

class VoiceCloningModel(nn.Module):
    """Complete voice cloning model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Components
        self.speaker_encoder = SpeakerEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.decoder = Decoder(config)
        self.vocoder = Vocoder(config)
    
    def forward(self, text, speaker_features, target_mels=None):
        """Forward pass through the complete model."""
        # Encode text and speaker
        text_embedding = self.text_encoder(text)
        speaker_embedding = self.speaker_encoder(speaker_features)
        
        # Combine embeddings
        combined_embedding = torch.cat([text_embedding, speaker_embedding], dim=1)
        
        # Decode to mel spectrogram
        max_length = target_mels.size(1) if target_mels is not None else 400  # Default max length
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

class AudioDataset(Dataset):
    """Dataset for training the voice cloning model."""
    def __init__(self, config, audio_processor, split="train"):
        self.config = config
        self.audio_processor = audio_processor
        self.split = split
        
        # Load dataset
        self.data_dir = os.path.join(config.data_dir, split)
        print(f"Loading {split} dataset from {self.data_dir}")
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            print(f"WARNING: {self.data_dir} does not exist!")
            self.metadata = []
        else:
            self.metadata = self._load_metadata()
            print(f"Loaded {len(self.metadata)} samples for {split}")
        
        # Sample rate and maximum length
        self.sample_rate = config.sample_rate
        self.max_audio_length = config.audio_length * config.sample_rate
    
    def _load_metadata(self):
        """Load metadata about audio samples."""
        metadata_path = os.path.join(self.config.data_dir, f"{self.split}_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        else:
            # Create metadata if it doesn't exist
            metadata = []
            speakers = os.listdir(self.data_dir)
            print(f"Found {len(speakers)} speakers: {speakers}")
            
            for speaker_id in speakers:
                speaker_dir = os.path.join(self.data_dir, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue
                
                audio_files = [f for f in os.listdir(speaker_dir) if f.endswith(".wav")]
                print(f"Found {len(audio_files)} audio files for speaker {speaker_id}")
                
                for audio_file in audio_files:
                    # Get transcript from transcript file
                    transcript_file = os.path.join(speaker_dir, audio_file.replace(".wav", ".txt"))
                    if os.path.exists(transcript_file):
                        with open(transcript_file, "r") as f:
                            transcript = f.read().strip()
                    else:
                        transcript = ""
                    
                    metadata.append({
                        "speaker_id": speaker_id,
                        "audio_path": os.path.join(speaker_dir, audio_file),
                        "transcript": transcript
                    })
            
            # Save metadata
            if metadata:
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
            
            return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        item = self.metadata[idx]
        
        # Load audio
        audio, _ = librosa.load(item["audio_path"], sr=self.sample_rate)
        
        # Pad or truncate audio
        if len(audio) > self.max_audio_length:
            audio = audio[:self.max_audio_length]
        else:
            audio = np.pad(audio, (0, self.max_audio_length - len(audio)))
        
        # Convert to mel spectrogram
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        
        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "mel_spectrogram": torch.tensor(mel_spec, dtype=torch.float32),
            "transcript": item["transcript"],
            "speaker_id": item["speaker_id"]
        }

class VoiceCloner:
    """Main class for voice cloning system."""
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
        self.audio_processor = AudioProcessor(self.config)
        self.model = VoiceCloningModel(self.config).to(self.config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
    
    def record_voice_sample(self, duration=10, output_path="voice_sample.wav"):
        """Record a voice sample for cloning."""
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = self.config.sample_rate
        
        p = pyaudio.PyAudio()
        
        print(f"Recording {duration} seconds of audio...")
        
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
        
        frames = []
        
        for i in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        print("Recording finished.")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recording
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Voice sample saved to {output_path}")
        return output_path
    
    def prepare_training_data(self, data_dir=None):
        """Prepare audio data for training."""
        if data_dir:
            self.config.data_dir = data_dir
        
        print(f"Preparing training data from {self.config.data_dir}...")
        
        # Create dataset splits
        train_dir = os.path.join(self.config.data_dir, "train")
        val_dir = os.path.join(self.config.data_dir, "val")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Process all speakers
        for speaker_id in os.listdir(self.config.data_dir):
            speaker_dir = os.path.join(self.config.data_dir, speaker_id)
            if not os.path.isdir(speaker_dir) or speaker_id in ["train", "val"]:
                continue
            
            # Create speaker directories in splits
            os.makedirs(os.path.join(train_dir, speaker_id), exist_ok=True)
            os.makedirs(os.path.join(val_dir, speaker_id), exist_ok=True)
            
            # Get all audio files
            audio_files = [f for f in os.listdir(speaker_dir) if f.endswith(".wav")]
            
            # Split into train/val (80/20)
            train_files = audio_files[:int(len(audio_files) * 0.8)]
            val_files = audio_files[int(len(audio_files) * 0.8):]
            
            # Process training files
            for audio_file in train_files:
                src_path = os.path.join(speaker_dir, audio_file)
                dst_path = os.path.join(train_dir, speaker_id, audio_file)
                
                if not os.path.exists(dst_path):
                    # Copy audio file
                    os.system(f"cp {src_path} {dst_path}")
                    
                    # Copy transcript if exists
                    transcript_file = audio_file.replace(".wav", ".txt")
                    src_transcript = os.path.join(speaker_dir, transcript_file)
                    dst_transcript = os.path.join(train_dir, speaker_id, transcript_file)
                    
                    if os.path.exists(src_transcript):
                        os.system(f"cp {src_transcript} {dst_transcript}")
            
            # Process validation files
            for audio_file in val_files:
                src_path = os.path.join(speaker_dir, audio_file)
                dst_path = os.path.join(val_dir, speaker_id, audio_file)
                
                if not os.path.exists(dst_path):
                    # Copy audio file
                    os.system(f"cp {src_path} {dst_path}")
                    
                    # Copy transcript if exists
                    transcript_file = audio_file.replace(".wav", ".txt")
                    src_transcript = os.path.join(speaker_dir, transcript_file)
                    dst_transcript = os.path.join(val_dir, speaker_id, transcript_file)
                    
                    if os.path.exists(src_transcript):
                        os.system(f"cp {src_transcript} {dst_transcript}")
        
        print("Training data preparation complete.")
    
    def train(self, epochs=None):
        """Train the voice cloning model."""
        if epochs:
            self.config.max_epochs = epochs
        
        # First, check if data exists
        train_dir = os.path.join(self.config.data_dir, "train")
        val_dir = os.path.join(self.config.data_dir, "val")
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"Training data not found. Please run prepare_training_data() first.")
            print(f"Expected train dir: {train_dir}")
            print(f"Expected val dir: {val_dir}")
            return
            
        # Create datasets
        train_dataset = AudioDataset(self.config, self.audio_processor, split="train")
        val_dataset = AudioDataset(self.config, self.audio_processor, split="val")
        
        # Verify datasets have data
        if len(train_dataset) == 0:
            print("Training dataset is empty. Cannot proceed with training.")
            return
            
        if len(val_dataset) == 0:
            print("Validation dataset is empty. Cannot proceed with training.")
            return
        
        # Fit normalizer
        print("Fitting normalizer...")
        mel_samples = []
        for i in range(min(100, len(train_dataset))):
            mel_samples.append(train_dataset[i]["mel_spectrogram"].numpy())
        
        if mel_samples:
            self.audio_processor.fit_normalizer(mel_samples)
        else:
            print("No mel samples available for normalization")
            return
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Loss functions
        mel_loss_fn = nn.L1Loss()
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.max_epochs):
            self.model.train()
            train_losses = []
            
            start_time = time.time()
            
            for batch in train_loader:
                # Move batch to device
                audio = batch["audio"].to(self.config.device)
                mel_spec = batch["mel_spectrogram"].to(self.config.device)
                transcript = batch["transcript"]
                
                # Forward pass
                outputs = self.model(transcript, audio, mel_spec)
                
                # Calculate losses
                mel_loss = mel_loss_fn(outputs["mel_outputs"], mel_spec)
                
                # Combined loss
                loss = mel_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move batch to device
                    audio = batch["audio"].to(self.config.device)
                    mel_spec = batch["mel_spectrogram"].to(self.config.device)
                    transcript = batch["transcript"]
                    
                    # Forward pass
                    outputs = self.model(transcript, audio, mel_spec)
                    
                    # Calculate losses
                    mel_loss = mel_loss_fn(outputs["mel_outputs"], mel_spec)
                    
                    # Combined loss
                    loss = mel_loss
                    
                    val_losses.append(loss.item())
            
            # Calculate epoch stats
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            # Update scheduler
            self.scheduler.step(avg_val_loss)
            
            # Print stats
            print(f"Epoch {epoch+1}/{self.config.max_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint if improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch
                }, checkpoint_path)
        
        print("Training complete!")
    
    def load_model(self, checkpoint_path=None):
        """Load a trained model."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {checkpoint_path}")
            return True
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return False
    
    def generate_speech(self, text, reference_audio_path, output_path="generated_speech.wav"):
        """Generate speech using the trained model."""
        self.model.eval()
        
        # Load reference audio
        reference_audio, _ = librosa.load(reference_audio_path, sr=self.config.sample_rate)
        
        # Convert to tensor
        reference_audio = torch.tensor(reference_audio, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        
        # Generate speech
        with torch.no_grad():
            outputs = self.model(text, reference_audio)
        
        # Convert to waveform
        waveform = outputs["waveform"].squeeze(0).cpu().numpy()
        
        # Save to file
        self.audio_processor.save_audio(waveform, output_path)
        
        print(f"Generated speech saved to {output_path}")
        return output_path
    
    def voice_similarity_score(self, original_audio_path, generated_audio_path):
        """Calculate similarity score between original and generated voice."""
        # Load audio files
        original_audio, _ = librosa.load(original_audio_path, sr=self.config.sample_rate)
        generated_audio, _ = librosa.load(generated_audio_path, sr=self.config.sample_rate)
        
        # Convert to tensors
        original_tensor = torch.tensor(original_audio, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        generated_tensor = torch.tensor(generated_audio, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        
        # Extract speaker embeddings
        with torch.no_grad():
            original_embedding = self.model.speaker_encoder(original_tensor)
            generated_embedding = self.model.speaker_encoder(generated_tensor)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(original_embedding, generated_embedding).item()
        
        return similarity
    
    def clone_voice_from_sample(self, sample_path, text_to_generate, output_path=None):
        """Clone a voice from a sample and generate speech with it."""
        if output_path is None:
# The output path wasn't explicitly given, generate a unique name
            output_path = f"cloned_speech_{int(time.time())}.wav"
        
        return self.generate_speech(text_to_generate, sample_path, output_path)
    
    def batch_clone(self, sample_path, texts, output_dir="cloned_outputs"):
        """Clone a voice and generate multiple speech samples."""
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"cloned_speech_{i}.wav")
            self.generate_speech(text, sample_path, output_path)
            output_paths.append(output_path)
        
        return output_paths
    
    def analyze_voice(self, audio_path):
        """Analyze voice characteristics."""
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Calculate pitch (f0)
        f0, voiced_flag, _ = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.config.sample_rate
        )
        
        # Filter out unvoiced segments
        f0 = f0[voiced_flag]
        
        # Calculate statistics
        if len(f0) > 0:
            mean_f0 = np.mean(f0)
            min_f0 = np.min(f0)
            max_f0 = np.max(f0)
        else:
            mean_f0 = min_f0 = max_f0 = 0
        
        # Get formants (simplified)
        S = librosa.stft(audio)
        D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.config.sample_rate, n_mfcc=13)
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.config.sample_rate)
        
        return {
            "pitch": {
                "mean": float(mean_f0),
                "min": float(min_f0),
                "max": float(max_f0)
            },
            "timbre": {
                "brightness": float(np.mean(spectral_centroid)),
                "mfccs": [float(np.mean(mfcc)) for mfcc in mfccs]
            },
            "duration": len(audio) / self.config.sample_rate
        }
    
    def visualize_voice_comparison(self, original_path, generated_path, output_path="voice_comparison.png"):
        """Visualize the comparison between original and generated voice."""
        # Load audio files
        original_audio, _ = librosa.load(original_path, sr=self.config.sample_rate)
        generated_audio, _ = librosa.load(generated_path, sr=self.config.sample_rate)
        
        # Calculate mel spectrograms
        original_mel = self.audio_processor.extract_mel_spectrogram(original_audio)
        generated_mel = self.audio_processor.extract_mel_spectrogram(generated_audio)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot original mel spectrogram
        plt.subplot(2, 1, 1)
        plt.imshow(original_mel, aspect='auto', origin='lower')
        plt.title("Original Voice Mel Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        
        # Plot generated mel spectrogram
        plt.subplot(2, 1, 2)
        plt.imshow(generated_mel, aspect='auto', origin='lower')
        plt.title("Generated Voice Mel Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Voice comparison visualization saved to {output_path}")
        return output_path
    
    def adjust_voice_characteristics(self, reference_audio_path, target_characteristics, output_path=None):
        """Adjust voice characteristics of a reference audio."""
        if output_path is None:
            output_path = f"adjusted_voice_{int(time.time())}.wav"
        
        # Load reference audio
        reference_audio, _ = librosa.load(reference_audio_path, sr=self.config.sample_rate)
        
        # Extract current characteristics
        current_analysis = self.analyze_voice(reference_audio_path)
        
        # Adjustments will depend on which characteristics are being modified
        adjusted_audio = reference_audio.copy()
        
        # Adjust pitch if requested
        if "pitch_shift" in target_characteristics:
            # Calculate pitch shift amount
            current_pitch = current_analysis["pitch"]["mean"]
            target_pitch = target_characteristics["pitch_shift"]
            
            # Convert Hz difference to steps
            if current_pitch > 0:
                steps = 12 * np.log2(target_pitch / current_pitch)
                adjusted_audio = librosa.effects.pitch_shift(
                    adjusted_audio, 
                    sr=self.config.sample_rate, 
                    n_steps=steps
                )
        
        # Adjust speed/tempo if requested
        if "tempo_factor" in target_characteristics:
            tempo_factor = target_characteristics["tempo_factor"]
            adjusted_audio = librosa.effects.time_stretch(adjusted_audio, rate=tempo_factor)
        
        # Save adjusted audio
        self.audio_processor.save_audio(adjusted_audio, output_path)
        
        print(f"Adjusted voice saved to {output_path}")
        return output_path
    
    def create_voice_profile(self, audio_path, name, save_dir="voice_profiles"):
        """Create and save a voice profile for future use."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        
        # Extract speaker embedding
        with torch.no_grad():
            speaker_embedding = self.model.speaker_encoder(audio_tensor)
        
        # Analyze voice characteristics
        voice_analysis = self.analyze_voice(audio_path)
        
        # Create profile
        profile = {
            "name": name,
            "embedding": speaker_embedding.cpu().numpy().tolist(),
            "characteristics": voice_analysis,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save profile
        profile_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.json")
        with open(profile_path, "w") as f:
            json.dump(profile, f)
        
        print(f"Voice profile for {name} saved to {profile_path}")
        return profile_path
    
    def list_voice_profiles(self, profile_dir="voice_profiles"):
        """List all available voice profiles."""
        if not os.path.exists(profile_dir):
            print("No voice profiles found.")
            return []
        
        profiles = []
        for filename in os.listdir(profile_dir):
            if filename.endswith(".json"):
                profile_path = os.path.join(profile_dir, filename)
                with open(profile_path, "r") as f:
                    profile = json.load(f)
                profiles.append({
                    "name": profile["name"],
                    "path": profile_path,
                    "created_at": profile["created_at"]
                })
        
        return profiles
    
    def generate_from_profile(self, text, profile_name, output_path=None):
        """Generate speech using a saved voice profile."""
        if output_path is None:
            output_path = f"profile_speech_{int(time.time())}.wav"
        
        # Find profile
        profile_path = os.path.join("voice_profiles", f"{profile_name.replace(' ', '_').lower()}.json")
        
        if not os.path.exists(profile_path):
            print(f"Voice profile for {profile_name} not found.")
            return None
        
        # Load profile
        with open(profile_path, "r") as f:
            profile = json.load(f)
        
        # Extract embedding
        embedding = torch.tensor(profile["embedding"], dtype=torch.float32).to(self.config.device)
        
        # Generate speech
        self.model.eval()
        with torch.no_grad():
            # Encode text
            text_embedding = self.model.text_encoder([text])
            
            # Combine embeddings
            combined_embedding = torch.cat([text_embedding, embedding], dim=1)
            
            # Decode to mel spectrogram
            mel_outputs, _ = self.model.decoder(combined_embedding, max_length=500)
            
            # Convert to waveform
            waveform = self.model.vocoder(mel_outputs).squeeze(0).cpu().numpy()
        
        # Save to file
        self.audio_processor.save_audio(waveform, output_path)
        
        print(f"Generated speech from profile {profile_name} saved to {output_path}")
        return output_path
    
    def merge_voices(self, voice_paths, weights=None, output_path=None):
        """Merge multiple voices with optional weights."""
        if output_path is None:
            output_path = f"merged_voice_{int(time.time())}.wav"
        
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(voice_paths)] * len(voice_paths)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Load all voices and extract embeddings
        embeddings = []
        for path in voice_paths:
            audio, _ = librosa.load(path, sr=self.config.sample_rate)
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                embedding = self.model.speaker_encoder(audio_tensor)
            embeddings.append(embedding)
        
        # Merge embeddings according to weights
        merged_embedding = torch.zeros_like(embeddings[0])
        for i, embedding in enumerate(embeddings):
            merged_embedding += embedding * weights[i]
        
        # Create voice profile from merged embedding
        profile = {
            "name": "merged_voice",
            "embedding": merged_embedding.cpu().numpy().tolist(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save profile
        profile_path = os.path.join("voice_profiles", "merged_voice.json")
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(profile, f)
        
        print(f"Merged voice profile saved.")
        return profile_path
    
    def process_audio_for_tour(self, script, voice_profile, output_dir="tour_audio", segment=True):
        """Process audio for real estate tour with optional segmentation by room."""
        os.makedirs(output_dir, exist_ok=True)
        
        if segment:
            # Split script by room markers
            segments = []
            current_segment = {"room": "Introduction", "text": ""}
            
            for line in script.splitlines():
                if line.startswith("## "):  # Markdown header for room
                    if current_segment["text"]:
                        segments.append(current_segment)
                    room_name = line.replace("## ", "").strip()
                    current_segment = {"room": room_name, "text": ""}
                else:
                    current_segment["text"] += line + "\n"
            
            # Add last segment
            if current_segment["text"]:
                segments.append(current_segment)
            
            # Generate audio for each segment
            outputs = []
            for i, segment in enumerate(segments):
                output_path = os.path.join(output_dir, f"{i:02d}_{segment['room'].replace(' ', '_').lower()}.wav")
                self.generate_from_profile(segment["text"], voice_profile, output_path)
                outputs.append({
                    "room": segment["room"],
                    "path": output_path
                })
            
            return outputs
        else:
            # Generate single audio file
            output_path = os.path.join(output_dir, "complete_tour.wav")
            self.generate_from_profile(script, voice_profile, output_path)
            return [{"room": "Complete Tour", "path": output_path}]
    
    def estimate_required_resources(self, audio_length_seconds, num_voices):
        """Estimate required computational resources."""
        # These are rough estimates
        memory_per_second = 50  # MB
        processing_time_factor = 0.5  # Real-time factor
        
        estimated_memory = audio_length_seconds * memory_per_second * num_voices
        estimated_processing_time = audio_length_seconds * processing_time_factor * num_voices
        
        return {
            "estimated_memory_mb": estimated_memory,
            "estimated_processing_time_seconds": estimated_processing_time,
            "recommended_gpu": estimated_memory > 4000  # Recommend GPU if over 4GB
        }

# Command-line interface
def cli():
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Cloning System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record a voice sample")
    record_parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    record_parser.add_argument("--output", type=str, default="voice_sample.wav", help="Output path")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, help="Data directory")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate speech")
    generate_parser.add_argument("--text", type=str, required=True, help="Text to speak")
    generate_parser.add_argument("--voice", type=str, required=True, help="Reference voice file or profile name")
    generate_parser.add_argument("--output", type=str, default=None, help="Output path")
    
    # Profile commands
    profile_parser = subparsers.add_parser("profile", help="Manage voice profiles")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", help="Profile command")
    
    create_profile_parser = profile_subparsers.add_parser("create", help="Create a voice profile")
    create_profile_parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    create_profile_parser.add_argument("--name", type=str, required=True, help="Profile name")
    
    list_profiles_parser = profile_subparsers.add_parser("list", help="List voice profiles")
    
    # Tour command
    tour_parser = subparsers.add_parser("tour", help="Generate audio for real estate tour")
    tour_parser.add_argument("--script", type=str, required=True, help="Tour script file")
    tour_parser.add_argument("--voice", type=str, required=True, help="Voice profile name")
    tour_parser.add_argument("--output", type=str, default="tour_audio", help="Output directory")
    tour_parser.add_argument("--segment", action="store_true", help="Segment by room")
    
    args = parser.parse_args()
    
    # Initialize voice cloner
    voice_cloner = VoiceCloner()
    
    if args.command == "record":
        voice_cloner.record_voice_sample(args.duration, args.output)
    
    elif args.command == "train":
        if args.data:
            voice_cloner.prepare_training_data(args.data)
        voice_cloner.train(args.epochs)
    
    elif args.command == "generate":
        # Check if voice is a file or profile name
        if os.path.exists(args.voice):
            voice_cloner.generate_speech(args.text, args.voice, args.output)
        else:
            voice_cloner.generate_from_profile(args.text, args.voice, args.output)
    
    elif args.command == "profile":
        if args.profile_command == "create":
            voice_cloner.create_voice_profile(args.audio, args.name)
        elif args.profile_command == "list":
            profiles = voice_cloner.list_voice_profiles()
            for profile in profiles:
                print(f"{profile['name']} (created: {profile['created_at']})")
    
    elif args.command == "tour":
        # Read script file
        with open(args.script, "r") as f:
            script = f.read()
        
        voice_cloner.process_audio_for_tour(script, args.voice, args.output, args.segment)

# Demo script to integrate with real estate video tour
def video_tour_demo():
    voice_cloner = VoiceCloner()
    
    # Record voice sample
    sample_path = voice_cloner.record_voice_sample(15, "narrator_voice.wav")
    
    # Create voice profile
    profile_name = "professional_narrator"
    voice_cloner.create_voice_profile(sample_path, profile_name)
    
    # Example room descriptions
    room_descriptions = {
        "entrance": "Welcome to this stunning modern home. As we enter, notice the spacious foyer with natural light pouring in from the skylights above.",
        "living_room": "Moving into the living room, you'll appreciate the open concept design. The floor-to-ceiling windows offer incredible views and the fireplace creates a cozy atmosphere.",
        "kitchen": "The gourmet kitchen features top-of-the-line stainless steel appliances, quartz countertops, and custom cabinetry. The large island provides ample space for cooking and entertaining.",
        "master_bedroom": "The master suite is truly a retreat, with a sitting area, walk-in closet, and private balcony overlooking the backyard.",
        "bathroom": "The en-suite bathroom includes a double vanity, soaking tub, and a walk-in shower with rainfall showerhead.",
        "backyard": "Finally, step outside to enjoy the beautifully landscaped backyard, complete with a covered patio and built-in BBQ area perfect for outdoor entertaining."
    }
    
    # Generate audio for each room
    audio_segments = {}
    for room, description in room_descriptions.items():
        output_path = f"tour_audio/{room}.wav"
        audio_segments[room] = voice_cloner.generate_from_profile(description, profile_name, output_path)
    
    print("Generated audio segments for video tour:")
    for room, path in audio_segments.items():
        print(f"- {room}: {path}")
    
    # Note: In a real implementation, these audio files would be synchronized 
    # with video footage of each room using a video editing library or tool

if __name__ == "__main__":
    cli()
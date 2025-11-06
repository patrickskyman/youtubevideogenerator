# Complete the ImprovedVoiceCloningModel class
class ImprovedVoiceCloningModel(nn.Module):
    """Improved voice cloning model with better speaker encoding and vocoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            input_dim=config.mel_channels,
            hidden_dim=config.encoder_hidden_dim,
            embedding_dim=config.speaker_embedding_dim,
            num_layers=config.encoder_num_layers
        )
        
        # Text encoder (transformer-based)
        self.text_encoder = TransformerTextEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.text_embedding_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            num_heads=config.transformer_heads,
            dropout=config.dropout
        )
        
        # Decoder (transformer-based)
        decoder_input_dim = config.text_embedding_dim + config.speaker_embedding_dim
        self.decoder = TransformerDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=config.decoder_hidden_dim,
            output_dim=config.mel_channels,
            num_layers=config.decoder_num_layers,
            num_heads=config.transformer_heads,
            dropout=config.dropout
        )
        
        # Vocoder (HiFi-GAN based)
        self.vocoder = ImprovedVocoder(
            input_dim=config.mel_channels,
            hidden_dim=config.vocoder_hidden_dim,
            resblock_kernel_sizes=config.vocoder_resblock_kernel_sizes,
            resblock_dilation_sizes=config.vocoder_resblock_dilation_sizes,
            upsample_rates=config.vocoder_upsample_rates,
            upsample_kernel_sizes=config.vocoder_upsample_kernel_sizes
        )
        
        # Duration predictor 
        self.duration_predictor = DurationPredictor(
            input_dim=config.text_embedding_dim + config.speaker_embedding_dim,
            hidden_dim=config.duration_predictor_hidden_dim,
            num_layers=config.duration_predictor_num_layers,
            dropout=config.dropout
        )
        
        # Pitch predictor 
        self.pitch_predictor = PitchPredictor(
            input_dim=config.text_embedding_dim + config.speaker_embedding_dim,
            hidden_dim=config.pitch_predictor_hidden_dim,
            num_layers=config.pitch_predictor_num_layers,
            dropout=config.dropout
        )
        
        # Energy predictor
        self.energy_predictor = EnergyPredictor(
            input_dim=config.text_embedding_dim + config.speaker_embedding_dim,
            hidden_dim=config.energy_predictor_hidden_dim,
            num_layers=config.energy_predictor_num_layers,
            dropout=config.dropout
        )
    
    def forward(self, text_indices, text_lengths, mel_specs, mel_lengths, reference_mel=None):
        """
        Forward pass through the model
        
        Args:
            text_indices: Tokenized text indices [batch, max_text_len]
            text_lengths: Length of each text sequence [batch]
            mel_specs: Mel spectrograms [batch, max_mel_len, mel_channels]
            mel_lengths: Length of each mel spectrogram [batch]
            reference_mel: Reference mel spectrogram for speaker embedding [batch, ref_mel_len, mel_channels]
                          (if None, use the target mel_specs as reference)
        
        Returns:
            Dictionary of model outputs and intermediate representations
        """
        batch_size = text_indices.size(0)
        
        # Get speaker embeddings from reference mel or target mel
        if reference_mel is not None:
            speaker_embeddings = self.speaker_encoder(reference_mel)
        else:
            speaker_embeddings = self.speaker_encoder(mel_specs)
        
        # Get text encodings
        text_encodings = self.text_encoder(text_indices, text_lengths)
        
        # Combine text and speaker embeddings
        expanded_speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, text_encodings.size(1), -1)
        decoder_inputs = torch.cat([text_encodings, expanded_speaker_embeddings], dim=-1)
        
        # Duration prediction
        log_durations = self.duration_predictor(decoder_inputs)
        
        # Pitch prediction
        pitch_predictions = self.pitch_predictor(decoder_inputs)
        
        # Energy prediction
        energy_predictions = self.energy_predictor(decoder_inputs)
        
        # Expand text encodings according to predicted durations
        if self.training:
            # During training, use ground truth durations from alignment
            # In a real implementation, you'd extract these from force alignment
            # For now, we use a simplified approach
            expanded_encodings = self.expand_encodings_with_durations(
                decoder_inputs, text_lengths, mel_lengths
            )
        else:
            # During inference, use predicted durations
            expanded_encodings = self.expand_encodings_with_predicted_durations(
                decoder_inputs, log_durations, text_lengths
            )
        
        # Generate mel spectrograms
        mel_outputs = self.decoder(expanded_encodings)
        
        # Generate audio waveform (only during inference to save computation)
        if not self.training:
            audio_outputs = self.vocoder(mel_outputs)
        else:
            audio_outputs = None
        
        return {
            "mel_outputs": mel_outputs,
            "audio_outputs": audio_outputs,
            "speaker_embeddings": speaker_embeddings,
            "log_durations": log_durations,
            "pitch_predictions": pitch_predictions,
            "energy_predictions": energy_predictions
        }
    
    def expand_encodings_with_durations(self, encodings, text_lengths, mel_lengths):
        """
        Expand text encodings to match target mel lengths (training time)
        Simplified version - in a real implementation, use alignment information
        """
        batch_size = encodings.size(0)
        expanded = []
        
        for i in range(batch_size):
            # Get sequence lengths for this sample
            text_len = text_lengths[i]
            mel_len = mel_lengths[i]
            
            # Only use valid text encoding for this sample
            valid_encoding = encodings[i, :text_len]
            
            # Simple expansion strategy: repeat each token encoding
            # to approximately match the target mel length
            ratio = mel_len / text_len
            expanded_encoding = []
            
            for j in range(text_len):
                # Calculate how many times to repeat this token
                repeat_count = max(1, int(ratio + (0.5 if j < text_len // 2 else 0)))
                expanded_encoding.append(valid_encoding[j:j+1].repeat(repeat_count, 1))
            
            # Concatenate and trim/pad to mel_len
            expanded_encoding = torch.cat(expanded_encoding, dim=0)
            if expanded_encoding.size(0) > mel_len:
                expanded_encoding = expanded_encoding[:mel_len]
            elif expanded_encoding.size(0) < mel_len:
                pad_len = mel_len - expanded_encoding.size(0)
                expanded_encoding = torch.cat([
                    expanded_encoding,
                    expanded_encoding[-1:].repeat(pad_len, 1)
                ], dim=0)
            
            expanded.append(expanded_encoding)
        
        # Pad to maximum mel length in batch
        max_len = max(mel_lengths)
        result = []
        for exp in expanded:
            if exp.size(0) < max_len:
                pad_len = max_len - exp.size(0)
                exp = torch.cat([
                    exp, 
                    torch.zeros(pad_len, exp.size(1), device=exp.device)
                ], dim=0)
            result.append(exp.unsqueeze(0))
        
        return torch.cat(result, dim=0)
    
    def expand_encodings_with_predicted_durations(self, encodings, log_durations, text_lengths):
        """
        Expand text encodings based on predicted durations (inference time)
        """
        batch_size = encodings.size(0)
        expanded = []
        
        for i in range(batch_size):
            # Get valid encoding and durations for this sample
            text_len = text_lengths[i]
            valid_encoding = encodings[i, :text_len]
            valid_durations = torch.exp(log_durations[i, :text_len]).round().long()
            
            # Ensure minimum duration
            valid_durations = torch.clamp(valid_durations, min=1)
            
            # Expand each token according to its predicted duration
            expanded_encoding = []
            for j in range(text_len):
                expanded_encoding.append(valid_encoding[j:j+1].repeat(valid_durations[j], 1))
            
            expanded_encoding = torch.cat(expanded_encoding, dim=0)
            expanded.append(expanded_encoding)
        
        # Pad to maximum expanded length
        max_len = max([exp.size(0) for exp in expanded])
        result = []
        for exp in expanded:
            if exp.size(0) < max_len:
                pad_len = max_len - exp.size(0)
                exp = torch.cat([
                    exp, 
                    torch.zeros(pad_len, exp.size(1), device=exp.device)
                ], dim=0)
            result.append(exp.unsqueeze(0))
        
        return torch.cat(result, dim=0)
    
    def synthesize(self, text, reference_mel=None, reference_audio=None):
        """
        Synthesize speech from text using a reference voice
        
        Args:
            text: Text to synthesize (string)
            reference_mel: Reference mel spectrogram (tensor) or None
            reference_audio: Reference audio (numpy array) or None
        
        Returns:
            Dictionary with synthesis results
        """
        self.eval()
        
        # Process text input
        text_processor = TextProcessor()
        text_indices = text_processor.text_to_indices(text)
        text_indices = torch.LongTensor(text_indices).unsqueeze(0).to(next(self.parameters()).device)
        text_lengths = torch.LongTensor([text_indices.size(1)]).to(next(self.parameters()).device)
        
        # Process reference input
        if reference_mel is None and reference_audio is not None:
            # Convert reference audio to mel spectrogram
            audio_processor = ImprovedAudioProcessor(self.config)
            reference_mel = audio_processor.extract_mel_spectrogram(reference_audio)
            reference_mel = torch.FloatTensor(reference_mel).unsqueeze(0).to(next(self.parameters()).device)
        
        if reference_mel is None:
            raise ValueError("Either reference_mel or reference_audio must be provided")
        
        # Forward pass
        with torch.no_grad():
            # Get speaker embedding
            speaker_embedding = self.speaker_encoder(reference_mel)
            
            # Get text encoding
            text_encoding = self.text_encoder(text_indices, text_lengths)
            
            # Combine text and speaker embeddings
            expanded_speaker_embedding = speaker_embedding.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
            decoder_input = torch.cat([text_encoding, expanded_speaker_embedding], dim=-1)
            
            # Predict durations, pitch, and energy
            log_durations = self.duration_predictor(decoder_input)
            pitch_prediction = self.pitch_predictor(decoder_input)
            energy_prediction = self.energy_predictor(decoder_input)
            
            # Expand encodings with predicted durations
            expanded_encoding = self.expand_encodings_with_predicted_durations(
                decoder_input, log_durations, text_lengths
            )
            
            # Generate mel spectrogram
            mel_output = self.decoder(expanded_encoding)
            
            # Generate audio
            audio_output = self.vocoder(mel_output)
        
        # Convert tensors to numpy arrays
        audio_output = audio_output.squeeze().cpu().numpy()
        mel_output = mel_output.squeeze().cpu().numpy()
        
        return {
            "audio": audio_output,
            "mel_spectrogram": mel_output,
            "sample_rate": self.config.sample_rate,
            "duration_seconds": len(audio_output) / self.config.sample_rate
        }

# Define the various components
class SpeakerEncoder(nn.Module):
    """Speaker encoder to extract speaker embeddings"""
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=3):
        super().__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Projection layer
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)
        
        # Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, mel_specs):
        """
        Extract speaker embedding from mel spectrogram
        
        Args:
            mel_specs: Mel spectrogram [batch, seq_len, mel_channels]
        
        Returns:
            Speaker embedding [batch, embedding_dim]
        """
        # Pass through LSTM
        lstm_outputs, (hidden, _) = self.lstm(mel_specs)
        
        # Get last hidden state from each direction
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Project to embedding dimension
        embedding = self.projection(hidden_concat)
        
        # Normalize
        embedding = self.layer_norm(embedding)
        
        return embedding

class TransformerTextEncoder(nn.Module):
    """Text encoder using transformer architecture"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, text_indices, text_lengths):
        """
        Encode text sequences
        
        Args:
            text_indices: Text token indices [batch, max_len]
            text_lengths: Length of each sequence [batch]
        
        Returns:
            Text encodings [batch, max_len, embedding_dim]
        """
        batch_size, max_len = text_indices.size()
        
        # Create attention mask
        mask = torch.arange(max_len, device=text_indices.device).expand(batch_size, max_len) >= text_lengths.unsqueeze(1)
        
        # Embed text
        embeddings = self.embedding(text_indices)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Pass through transformer
        encodings = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        
        # Pass through output layer
        outputs = self.output_layer(encodings)
        
        return outputs

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    """Decoder using transformer architecture"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim, dropout=dropout)
        
        # Transformer decoder layers
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, encodings, target_mask=None):
        """
        Decode sequence to mel spectrogram
        
        Args:
            encodings: Encoded inputs [batch, seq_len, input_dim]
            target_mask: Optional mask for targets
        
        Returns:
            Decoded mel spectrogram [batch, seq_len, output_dim]
        """
        # Add positional encoding
        x = self.pos_encoding(encodings)
        
        # Create memory mask to prevent attending to padding
        memory_mask = None
        
        # Transformer decoding
        # For a simplified implementation, we're using the encodings directly as memory
        # In a full implementation, you would have separate memory from the encoder
        decoded = self.transformer_decoder(x, encodings, tgt_mask=target_mask, memory_mask=memory_mask)
        
        # Project to output dimension
        outputs = self.output_layer(decoded)
        
        return outputs

class ImprovedVocoder(nn.Module):
    """HiFi-GAN based vocoder for high-quality audio synthesis"""
    def __init__(self, input_dim, hidden_dim, resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_rates, upsample_kernel_sizes):
        super().__init__()
        
        # Initial projection
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        curr_dim = hidden_dim
        
        for i, (upsample_rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsample_layers.append(nn.ConvTranspose1d(
                curr_dim,
                curr_dim // 2,
                kernel_size=kernel_size,
                stride=upsample_rate,
                padding=(kernel_size - upsample_rate) // 2
            ))
            curr_dim //= 2
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(resblock_kernel_sizes)):
            self.resblocks.append(ResBlock(
                channels=curr_dim,
                kernel_size=resblock_kernel_sizes[i],
                dilations=resblock_dilation_sizes[i]
            ))
        
        # Output convolution
        self.output_conv = nn.Conv1d(curr_dim, 1, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()
    
    def forward(self, mel_specs):
        """
        Generate audio from mel spectrogram
        
        Args:
            mel_specs: Mel spectrogram [batch, seq_len, mel_channels]
        
        Returns:
            Audio waveform [batch, seq_len * upsample_rate]
        """
        # Transpose for 1D convolution
        x = mel_specs.transpose(1, 2)  # [batch, mel_channels, seq_len]
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Upsampling
        for upsample_layer in self.upsample_layers:
            x = F.leaky_relu(upsample_layer(x), 0.1)
        
        # Residual blocks
        for resblock in self.resblocks:
            x = resblock(x)
        
        # Output convolution
        x = self.output_conv(x)
        x = self.tanh(x)
        
        # Return audio waveform
        return x.squeeze(1)

class ResBlock(nn.Module):
    """Residual block for vocoder"""
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.conv_layers.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    channels, channels, kernel_size,
                    padding=padding, dilation=dilation
                ),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=1)
            ))
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            residual = x
            x = conv_layer(x)
            x = x + residual
        return x

class DurationPredictor(nn.Module):
    """Duration predictor for phoneme/token durations"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ))
        
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Predict log durations for each token
        
        Args:
            x: Input features [batch, seq_len, input_dim]
        
        Returns:
            Log durations [batch, seq_len, 1]
        """
        for layer in self.layers:
            x = layer(x)
        
        # Output log durations
        log_durations = self.output_layer(x)
        
        return log_durations.squeeze(-1)  # [batch, seq_len]

class PitchPredictor(nn.Module):
    """Pitch predictor for prosody control"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ))
        
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Predict pitch contour
        
        Args:
            x: Input features [batch, seq_len, input_dim]
        
        Returns:
            Pitch values [batch, seq_len, 1]
        """
        for layer in self.layers:
            x = layer(x)
        
        # Output pitch values
        pitch = self.output_layer(x)
        
        return pitch.squeeze(-1)  # [batch, seq_len]

class EnergyPredictor(nn.Module):
    """Energy predictor for prosody control"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ))
        
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Predict energy contour
        
        Args:
            x: Input features [batch, seq_len, input_dim]
        
        Returns:
            Energy values [batch, seq_len, 1]
        """
        for layer in self.layers:
            x = layer(x)
        
        # Output energy values
        energy = self.output_layer(x)
        
        return energy.squeeze(-1)  # [batch, seq_len]

# Define Text Processor
class TextProcessor:
    """Process text input for voice cloning"""
    def __init__(self, vocab_file=None):
        # Default vocabulary
        self.char_to_idx = {
            '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, ' ': 4,
            'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11,
            'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'n': 18,
            'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 't': 24, 'u': 25,
            'v': 26, 'w': 27, 'x': 28, 'y': 29, 'z': 30, "'": 31, '!': 32,
            ',': 33, '.': 34, '?': 35, '-': 36
        }
        
        # Load custom vocabulary if provided
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
                self.char_to_idx = vocab
        
        # Create reverse mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Initialize tokenizer or use a simple character-level tokenizer
        self.vocab_size = len(self.char_to_idx)
    
    def text_to_indices(self, text):
        """Convert text to token indices"""
        # Lowercase and add start/end tokens
        text = f"<s> {text.lower()} </s>"
        
        # Convert to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<unk>'])
        
        return indices
    
    def indices_to_text(self, indices):
        """Convert token indices back to text"""
        text = []
        for idx in indices:
            if idx in self.idx_to_char:
                text.append(self.idx_to_char[idx])
            else:
                text.append('<unk>')
        
        # Remove special tokens
        text = ''.join(text)
        text = text.replace('<s>', '').replace('</s>', '').strip()
        
        return text

# Define Audio Processor
class ImprovedAudioProcessor:
    """Process audio for voice cloning with improved quality"""
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_mels = config.mel_channels
        self.fmin = config.fmin
        self.fmax = config.fmax
        
        # Normalization statistics
        self.mel_mean = None
        self.mel_std = None
    
    def load_audio(self, audio_path):
        """Load audio file"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram from audio"""
        # Ensure audio is numpy array
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        # Compute stft
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Convert to magnitude
        magnitude = np.abs(stft)
        
        # Convert to mel scale
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize if statistics are available
        if self.mel_mean is not None and self.mel_std is not None:
            mel_spec = (mel_spec - self.mel_mean) / self.mel_std
        
        return mel_spec.T  # [time, mel_dim]
    
    def compute_normalization_stats(self, mel_specs):
        """Compute mean and std for normalization"""
        mel_concatenated = np.vstack(mel_specs)
        self.mel_mean = np.mean(mel_concatenated, axis=0)
        self.mel_std = np.std(mel_concatenated, axis=0)
        
     
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import cv2
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class AudioVisualDataset(Dataset):
    """Dataset for audio-visual training data"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load filenames
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    audio_path = os.path.join(root, file.replace('.mp4', '.wav'))
                    
                    if os.path.exists(audio_path):
                        self.samples.append((video_path, audio_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, audio_path = self.samples[idx]
        
        # Extract audio features
        audio_features = extract_audio_features(audio_path)
        
        # Extract video mouth landmarks
        mouth_landmarks = extract_mouth_landmarks(video_path)
        
        # Create synchronized data
        synced_data = synchronize_data(audio_features, mouth_landmarks)
        
        if self.transform:
            synced_data = self.transform(synced_data)
        
        return synced_data

def extract_audio_features(audio_path):
    """Extract MFCC features from audio"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract additional features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Combine features
    features = np.concatenate([mfccs, chroma, mel])
    
    return features

def extract_mouth_landmarks(video_path):
    """Extract mouth landmarks from video frames"""
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Mouth landmark indices
    outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91]
    mouth_indices = list(set(outer_lip_indices + inner_lip_indices))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Extract mouth landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get mouth landmarks
            mouth_landmarks = []
            for idx in mouth_indices:
                landmark = face_landmarks.landmark[idx]
                mouth_landmarks.append([landmark.x, landmark.y])
            
            landmarks_sequence.append(np.array(mouth_landmarks))
        else:
            # If no face detected, use the previous landmarks or zeros
            if landmarks_sequence:
                landmarks_sequence.append(landmarks_sequence[-1])
            else:
                landmarks_sequence.append(np.zeros((len(mouth_indices), 2)))
    
    cap.release()
    return np.array(landmarks_sequence)

def synchronize_data(audio_features, mouth_landmarks):
    """Synchronize audio features with mouth landmarks"""
    # Resample to ensure same length
    audio_len = audio_features.shape[1]
    video_len = mouth_landmarks.shape[0]
    
    if audio_len > video_len:
        # Downsample audio to match video
        audio_features = librosa.resample(audio_features, orig_sr=audio_len, target_sr=video_len)
    elif video_len > audio_len:
        # Downsample video to match audio
        indices = np.linspace(0, video_len-1, audio_len, dtype=int)
        mouth_landmarks = mouth_landmarks[indices]
    
    return {
        'audio_features': audio_features,
        'mouth_landmarks': mouth_landmarks
    }

class VisemeModel(nn.Module):
    """Model to predict mouth shapes from audio features"""
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=40):
        super(VisemeModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_viseme_model(data_dir, epochs=50, batch_size=16, learning_rate=0.001):
    """Train a viseme prediction model"""
    # Create dataset and dataloader
    dataset = AudioVisualDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = dataset[0]['audio_features'].shape[0]  # Number of audio features
    output_dim = dataset[0]['mouth_landmarks'].shape[1] * 2  # x, y for each landmark
    
    model = VisemeModel(input_dim=input_dim, output_dim=output_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                
                # Get data
                audio_features = batch['audio_features']
                mouth_landmarks = batch['mouth_landmarks']
                
                # Forward pass
                outputs = model(audio_features)
                loss = criterion(outputs, mouth_landmarks.view(outputs.shape))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "viseme_model.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    # Set data directory
    data_dir = "path/to/audio_visual_dataset"
    
    # Train the model
    train_viseme_model(data_dir)
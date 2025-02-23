import os
import requests
import bz2

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_bz2(filepath):
    newfilepath = filepath[:-4]  # Remove .bz2 extension
    with bz2.BZ2File(filepath) as fr, open(newfilepath, 'wb') as fw:
        fw.write(fr.read())
    os.remove(filepath)  # Remove the bz2 file after extraction
    return newfilepath

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download face landmark model
    landmark_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    landmark_bz2_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    
    print("Downloading face landmark model...")
    download_file(landmark_url, landmark_bz2_path)
    
    print("Extracting face landmark model...")
    extract_bz2(landmark_bz2_path)
    
    # Download face recognition model
    recognition_url = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
    recognition_bz2_path = "models/dlib_face_recognition_resnet_model_v1.dat.bz2"
    
    print("Downloading face recognition model...")
    download_file(recognition_url, recognition_bz2_path)
    
    print("Extracting face recognition model...")
    extract_bz2(recognition_bz2_path)
    
    print("Done! Models downloaded and extracted successfully.")

if __name__ == "__main__":
    main()
import os
import numpy as np
from tensorflow.keras.models import load_model

# Test model loading
MODEL_PATH = 'models/fer2013_big_XCEPTION.54-0.66.hdf5'

print("🔧 Testing model compatibility...")

try:
    print("1. Trying normal load...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
except Exception as e:
    print(f"❌ Normal load failed: {e}")
    
    try:
        print("2. Trying load without compilation...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded without compilation!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Test with dummy input
        print("3. Testing dummy prediction...")
        dummy_input = np.random.random((1, 64, 64, 1))
        predictions = model.predict(dummy_input, verbose=0)
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Sample prediction: {predictions[0]}")
        print("✅ Model works for prediction!")
        
    except Exception as e2:
        print(f"❌ Load without compilation failed: {e2}")
        
        print("\n🔧 Checking model file...")
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"   File exists: {MODEL_PATH}")
            print(f"   File size: {file_size:.1f} MB")
        else:
            print(f"   File not found: {MODEL_PATH}")

print("\n" + "="*50)

# -*- coding: utf-8 -*-
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import base64

class EnsembleImageDetector:
    def __init__(self):
        """Load multiple models for better accuracy"""
        print("Loading ensemble image detectors...")
        
        self.models = []
        model_names = [
            "umm-maybe/AI-image-detector",
            "Organika/sdxl-detector"
        ]
        
        for model_name in model_names:
            try:
                print(f"  Loading {model_name}...")
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
                model.eval()
                self.models.append({
                    'name': model_name,
                    'processor': processor,
                    'model': model
                })
                print(f"  ✓ {model_name} loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
        
        if len(self.models) == 0:
            raise Exception("Failed to load any models!")
        
        print(f"Loaded {len(self.models)} models for ensemble\n")
    
    def detect_from_base64(self, base64_string):
        """Detect using ensemble voting"""
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return self.detect_from_image(image)
        except Exception as e:
            print(f"Error decoding image: {e}")
            raise
    
    def detect_from_image(self, image):
        """Ensemble detection with voting and metadata analysis"""
        width, height = image.size
        total_pixels = width * height
        megapixels = total_pixels / 1000000
        
        print(f"Analyzing: {width}x{height} ({megapixels:.1f}MP)")
        
        # Get predictions from all models
        predictions = []
        for model_info in self.models:
            try:
                inputs = model_info['processor'](images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model_info['model'](**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                if probs.shape[1] == 2:
                    ai_prob = probs[0][1].item()
                else:
                    ai_prob = probs[0][0].item()
                
                predictions.append(ai_prob)
                print(f"  Model prediction: {ai_prob*100:.1f}% AI")
            except Exception as e:
                print(f"  Model error: {e}")
        
        if not predictions:
            raise Exception("All models failed!")
        
        # Average predictions
        avg_ai_prob = sum(predictions) / len(predictions)
        
        # Metadata analysis
        has_exif = False
        exif_count = 0
        try:
            exif = image.getexif()
            if exif:
                exif_count = len(exif)
                has_exif = exif_count > 8
        except:
            pass
        
        # Check AI characteristics
        aspect_ratio = width / height
        is_square = 0.95 < aspect_ratio < 1.05
        common_ai_sizes = [512, 768, 1024, 1536, 2048]
        is_ai_size = width in common_ai_sizes and height in common_ai_sizes
        
        # Strong indicators
        strong_real = sum([has_exif, megapixels > 8, not is_ai_size])
        strong_ai = sum([exif_count == 0, is_square, is_ai_size])
        
        # Apply calibration
        final_prob = avg_ai_prob
        
        if strong_real >= 2:
            final_prob = final_prob * 0.5
        elif has_exif:
            final_prob = final_prob * 0.6
        
        if strong_ai >= 2:
            final_prob = final_prob * 1.3
        
        final_prob = final_prob * 0.9
        final_prob = max(0.05, min(0.95, final_prob))
        
        print(f"Final: {final_prob*100:.1f}% AI")
        
        # Generate explanations
        explanations = self._generate_explanations(
            has_exif, is_square, is_ai_size, megapixels, width, height, final_prob
        )
        
        distance = abs(final_prob - 0.5)
        confidence = "High" if distance > 0.3 else "Medium" if distance > 0.2 else "Low"
        
        return {
            'prediction': 'AI' if final_prob > 0.5 else 'Real',
            'ai_probability': round(final_prob * 100, 2),
            'real_probability': round((1 - final_prob) * 100, 2),
            'confidence': confidence,
            'explanations': explanations
        }
    
    def _generate_explanations(self, has_exif, is_square, is_ai_size, mp, w, h, prob):
        """Generate user-friendly explanations"""
        explanations = []
        
        if has_exif:
            explanations.append({
                'indicator': 'Camera Metadata Detected',
                'description': 'Image contains extensive EXIF data with camera settings, strongly suggesting authentic photograph.',
                'type': 'Real'
            })
        else:
            explanations.append({
                'indicator': 'No Camera Metadata',
                'description': 'Missing EXIF data normally present in photos from cameras and smartphones.',
                'type': 'AI'
            })
        
        if is_ai_size:
            explanations.append({
                'indicator': 'AI-Standard Dimensions',
                'description': f'Image size ({w}x{h}) matches common AI generation formats.',
                'type': 'AI'
            })
        else:
            explanations.append({
                'indicator': 'Unique Dimensions',
                'description': f'Non-standard dimensions ({w}x{h}) typical of real camera sensors.',
                'type': 'Real'
            })
        
        if mp > 8:
            explanations.append({
                'indicator': 'High Camera Resolution',
                'description': f'Very high resolution ({mp:.1f}MP) typical of modern cameras.',
                'type': 'Real'
            })
        elif mp < 2:
            explanations.append({
                'indicator': 'Low Resolution',
                'description': f'Low resolution ({mp:.1f}MP) common in AI-generated images.',
                'type': 'AI'
            })
        
        if prob > 0.7:
            explanations.append({
                'indicator': 'Strong AI Patterns',
                'description': 'Multiple models detected characteristic AI generation patterns.',
                'type': 'AI'
            })
        elif prob < 0.3:
            explanations.append({
                'indicator': 'Authentic Photography',
                'description': 'Multiple models confirmed natural photographic characteristics.',
                'type': 'Real'
            })
        else:
            explanations.append({
                'indicator': 'Uncertain',
                'description': 'Modern AI generation is extremely realistic. Consider other evidence.',
                'type': 'Neutral'
            })
        
        return explanations[:5]

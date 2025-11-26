# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class SentenceBasedTextDetector:
    def __init__(self, model_name="Hello-SimpleAI/chatgpt-detector-roberta"):
        """
        Best working models:
        1. "Hello-SimpleAI/chatgpt-detector-roberta" - Best overall (RECOMMENDED)
        2. "C:/Users/Kush/Desktop/ai-text-detector-model 2" - Good for formal AI
        """
        print(f"Loading model: {model_name}")
        
        if os.path.exists(str(model_name)):
            print("[*] Loading from local path...")
        else:
            print("[*] Downloading from Hugging Face (first time only)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        print("[OK] Model loaded successfully")
    
    def split_into_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def analyze_sentence(self, sentence):
        """Get AI probability for a sentence with calibration"""
        inputs = self.tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            if probs.shape[1] == 2:
                ai_prob = probs[0][1].item()
            else:
                ai_prob = probs[0][0].item()
        
        # ✅ CALIBRATION: Adjust predictions based on patterns
        sentence_lower = sentence.lower()
        
        # Strong human indicators - reduce AI score
        informal_markers = ['lol', 'haha', 'omg', 'btw', 'tbh', 'gonna', 'wanna', 
                           'kinda', 'sorta', 'yeah', 'nah', 'idk', ' u ', 'ngl',
                           '???', '!!!', '...']
        informal_count = sum(1 for marker in informal_markers if marker in sentence_lower)
        
        if informal_count >= 3:
            ai_prob *= 0.3  # Very informal - definitely human
        elif informal_count >= 2:
            ai_prob *= 0.5  # Somewhat informal - likely human
        elif informal_count >= 1:
            ai_prob *= 0.7  # Some informality
        
        # Check for contractions (human trait)
        contractions = ["can't", "won't", "ain't", "shouldn't", "wouldn't", 
                       "i'm", "we're", "they're", "it's", "that's"]
        if any(c in sentence_lower for c in contractions):
            ai_prob *= 0.8
        
        # Questions often human
        if '?' in sentence and len(sentence.split()) < 15:
            ai_prob *= 0.8
        
        # Very short casual sentences are human
        if len(sentence.split()) < 10 and any(m in sentence_lower for m in ['hey', 'hi', 'yo', 'sup']):
            ai_prob *= 0.5
        
        # Strong AI indicators - increase score
        formal_transitions = ['furthermore', 'moreover', 'additionally', 'consequently',
                            'therefore', 'thus', 'hence', 'nevertheless', 'nonetheless']
        if any(t in sentence_lower for t in formal_transitions):
            ai_prob = min(ai_prob * 1.3, 0.95)
        
        # AI buzzwords
        ai_buzzwords = ['facilitate', 'utilize', 'leverage', 'comprehensive', 
                       'optimize', 'strategic', 'framework', 'methodology']
        buzzword_count = sum(1 for word in ai_buzzwords if word in sentence_lower)
        if buzzword_count >= 2:
            ai_prob = min(ai_prob * 1.4, 0.95)
        elif buzzword_count >= 1:
            ai_prob = min(ai_prob * 1.2, 0.95)
        
        # Clamp between 5% and 95%
        ai_prob = max(0.05, min(0.95, ai_prob))
        
        return ai_prob
    
    def get_sentence_explanation(self, sentence, ai_score):
        """Generate explanation for sentence classification"""
        sentence_lower = sentence.lower()
        reasons = []
        
        # AI Indicators
        formal_transitions = ['furthermore', 'moreover', 'additionally', 'consequently',
                            'therefore', 'thus', 'hence', 'nevertheless', 'nonetheless',
                            'in conclusion', 'to summarize', 'it is important to note']
        
        ai_buzzwords = ['delve', 'utilize', 'leverage', 'facilitate', 'implement',
                       'comprehensive', 'robust', 'seamless', 'streamline', 'optimize',
                       'strategic', 'framework', 'methodology', 'paramount']
        
        passive_voice = ['is known', 'are made', 'was created', 'were developed',
                        'can be found', 'has been', 'have been', 'will be']
        
        # Human Indicators
        informal_markers = ['lol', 'haha', 'omg', 'btw', 'tbh', 'gonna', 'wanna',
                           'kinda', 'sorta', 'yeah', 'nah', 'idk', ' u ', 'ngl',
                           '...', '!!', '??', 'bruh', 'fr', 'lowkey']
        
        contractions = ["can't", "won't", "ain't", "shouldn't", "wouldn't",
                       "i'm", "we're", "they're", "it's"]
        
        # Check patterns
        if any(m in sentence_lower for m in informal_markers):
            reasons.append("Informal conversational language")
        
        if any(c in sentence_lower for c in contractions):
            reasons.append("Natural contractions")
        
        if any(t in sentence_lower for t in formal_transitions):
            reasons.append("Formal tone and structure")
        
        if any(w in sentence_lower for w in ai_buzzwords):
            reasons.append("Technical/corporate vocabulary")
        
        if any(p in sentence_lower for p in passive_voice):
            reasons.append("Passive voice construction")
        
        if len(sentence.split()) > 25:
            reasons.append("Very long, complex sentence")
        
        if sentence.count(',') >= 3:
            reasons.append("Multiple clauses")
        
        if '?' in sentence:
            reasons.append("Direct question")
        
        # Default reasons
        if not reasons:
            if ai_score > 0.7:
                reasons.append("Formulaic structure")
            elif ai_score < 0.3:
                reasons.append("Natural expression")
            else:
                reasons.append("Mixed characteristics")
        
        return ". ".join(reasons[:2]) + "."
    
    def explain(self, text):
        """Analyze text and return sentence-level explanations"""
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return self._analyze_whole_text(text)
        
        sentence_results = []
        
        for sentence in sentences:
            score = self.analyze_sentence(sentence)
            reason = self.get_sentence_explanation(sentence, score)
            
            sentence_results.append({
                'sentence': sentence,
                'ai_probability': score,
                'reason': reason
            })
        
        # Calculate overall score as weighted average
        total_weight = 0
        weighted_sum = 0
        
        for result in sentence_results:
            weight = abs(result['ai_probability'] - 0.5) + 0.5
            weighted_sum += result['ai_probability'] * weight
            total_weight += weight
        
        overall_ai_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Sort by AI probability
        sentence_results.sort(key=lambda x: x['ai_probability'], reverse=True)
        
        # Get indicators
        ai_indicators = [s for s in sentence_results if s['ai_probability'] > 0.55][:5]
        human_indicators = [s for s in sentence_results if s['ai_probability'] < 0.45][:5]
        
        # Calculate confidence
        distance = abs(overall_ai_prob - 0.5)
        confidence = "High" if distance > 0.25 else "Medium" if distance > 0.15 else "Low"
        
        return {
            'prediction': 'AI' if overall_ai_prob > 0.5 else 'Human',
            'ai_probability': round(overall_ai_prob * 100, 2),
            'human_probability': round((1 - overall_ai_prob) * 100, 2),
            'confidence': confidence,
            'ai_indicators': ai_indicators,
            'human_indicators': human_indicators
        }
    
    def _analyze_whole_text(self, text):
        """Fallback for short text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            overall_ai_prob = probs[0][1].item() if probs.shape[1] == 2 else probs[0][0].item()
        
        distance = abs(overall_ai_prob - 0.5)
        confidence = "High" if distance > 0.25 else "Medium" if distance > 0.15 else "Low"
        
        return {
            'prediction': 'AI' if overall_ai_prob > 0.5 else 'Human',
            'ai_probability': round(overall_ai_prob * 100, 2),
            'human_probability': round((1 - overall_ai_prob) * 100, 2),
            'confidence': confidence,
            'ai_indicators': [] if overall_ai_prob <= 0.5 else [{
                'sentence': text,
                'score': overall_ai_prob,
                'reason': self.get_sentence_explanation(text, overall_ai_prob)
            }],
            'human_indicators': [] if overall_ai_prob > 0.5 else [{
                'sentence': text,
                'score': overall_ai_prob,
                'reason': self.get_sentence_explanation(text, overall_ai_prob)
            }]
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AI Text Detection - Testing with Calibration")
    print("="*70)
    
    MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
    
    detector = SentenceBasedTextDetector(MODEL_NAME)
    
    # Test cases
    tests = [
        ("omg i cant believe what happened yesterday lol", "Human"),
        ("Furthermore, it is important to note that comprehensive analysis", "AI"),
        ("hey whats up? wanna hang out later?", "Human"),
        ("The strategic framework facilitates optimal outcomes", "AI")
    ]
    
    for text, expected in tests:
        result = detector.explain(text)
        status = "[OK]" if result['prediction'] == expected else "[FAIL]"
        print(f"\n{status} {text[:50]}...")
        print(f"Expected: {expected}, Got: {result['prediction']} ({result['ai_probability']:.1f}%)")

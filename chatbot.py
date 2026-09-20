"""
chatbot.py

Enhanced intent-classification chatbot for the Precision Agriculture
Advisor. Uses TF-IDF + cosine similarity with expanded intents covering
most farming questions. Now supports farmer name greeting and fallback
to a more flexible response system.
"""

import re
import random
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# EXPANDED INTENTS - Covers most farming questions
# ============================================================================

INTENTS = {
    # ----- IRRIGATION -----
    "irrigation_timing": {
        "examples": [
            "when should I irrigate",
            "do I need to water my crops",
            "is it time to irrigate",
            "should I water today",
            "how much water does my crop need",
            "when is the best time to water",
            "should I irrigate now",
            "how often should I water",
            "irrigation schedule",
            "watering my crops",
            "when to water maize",
            "irrigation advice",
            "water my plants",
            "how much irrigation",
            "irrigation frequency",
        ],
        "response": "Based on current rainfall and soil moisture, {irrigation_advice}"
    },
    
    # ----- YIELD -----
    "yield_forecast": {
        "examples": [
            "what will my yield be",
            "how much will I harvest",
            "is my yield on track",
            "predict my crop yield",
            "how many tons per hectare will I get",
            "yield prediction",
            "expected harvest",
            "crop yield forecast",
            "what is my yield",
            "yield estimate",
            "how many tons",
            "harvest prediction",
            "will my crop produce well",
            "yield outlook",
        ],
        "response": "Your estimated yield for this season is {predicted_yield} tons/ha."
    },
    
    # ----- DISEASE -----
    "disease_check": {
        "examples": [
            "is my plant sick",
            "what does this leaf spot mean",
            "my crop looks diseased",
            "check my plant for disease",
            "pest problem on my crop",
            "plant disease",
            "leaf spots",
            "my plant is dying",
            "what is wrong with my crop",
            "disease symptoms",
            "is this a disease",
            "crop infection",
            "fungus on my plant",
            "my leaves are turning yellow",
            "brown spots on leaves",
        ],
        "response": "Upload a leaf photo using the Disease Check tab and I'll assess it. (Note: this module is still being trained - see the CNN section of the project.)"
    },
    
    # ----- FERTILIZER -----
    "fertilizer_advice": {
        "examples": [
            "how much fertilizer should I use",
            "am I using too much fertilizer",
            "fertilizer recommendation",
            "what fertilizer do I need",
            "nitrogen fertilizer",
            "phosphorus fertilizer",
            "potassium fertilizer",
            "how to fertilize",
            "fertilizer rate",
            "nutrient management",
            "soil nutrients",
            "compost advice",
            "organic fertilizer",
            "chemical fertilizer",
        ],
        "response": "Your current fertilizer intensity is {fertilizer_note}"
    },
    
    # ----- PLANTING -----
    "planting_advice": {
        "examples": [
            "when should I plant",
            "best time to plant",
            "planting season",
            "what month to plant maize",
            "when to sow",
            "planting date",
            "when should I sow",
            "planting calendar",
            "germination time",
            "planting depth",
            "how deep to plant",
            "spacing between plants",
            "plant population",
            "seed rate",
        ],
        "response": "For {crop_type}, the optimal planting window is between October and December. Plant at 4-5cm depth with 25-30cm spacing for best results."
    },
    
    # ----- HARVEST -----
    "harvest_advice": {
        "examples": [
            "when to harvest",
            "harvest time",
            "is my crop ready to harvest",
            "harvesting advice",
            "when should I harvest",
            "harvest date",
            "crop maturity",
            "harvest season",
            "when to pick",
        ],
        "response": "Your crop will be ready for harvest in approximately {days_to_harvest} days. Look for {harvest_signs} as indicators of readiness."
    },
    
    # ----- PEST CONTROL -----
    "pest_control": {
        "examples": [
            "how to control pests",
            "pest management",
            "get rid of pests",
            "pesticide advice",
            "bugs on my crop",
            "insect control",
            "caterpillar problem",
            "aphid infestation",
            "natural pest control",
            "organic pest management",
            "how to stop pests",
            "pest treatment",
        ],
        "response": "For pest control, I recommend monitoring weekly and using integrated pest management (IPM). For specific pests, try neem oil as a natural option. If the infestation is severe, consider a targeted pesticide application."
    },
    
    # ----- SOIL -----
    "soil_advice": {
        "examples": [
            "soil type",
            "what is my soil",
            "soil health",
            "soil preparation",
            "how to improve soil",
            "soil testing",
            "soil pH",
            "soil nutrients",
            "soil quality",
            "soil management",
            "tillage advice",
            "soil conservation",
        ],
        "response": "Your soil type is {soil_type}. To improve soil health, consider adding organic matter, practicing crop rotation, and testing soil nutrients regularly. Maintain pH between 6.0-7.0 for most crops."
    },
    
    # ----- WEATHER -----
    "weather_advice": {
        "examples": [
            "what is the weather forecast",
            "will it rain",
            "rainfall forecast",
            "temperature forecast",
            "weather report",
            "climate outlook",
            "is it going to rain",
            "what is the temperature",
            "weather conditions",
            "seasonal forecast",
        ],
        "response": "Check the Rainfall Forecast tab for detailed predictions. Based on current models, the next 7 days will see approximately {rainfall_7_day}mm of rain."
    },
    
    # ----- CROP SPECIFIC -----
    "crop_specific_maize": {
        "examples": [
            "maize advice",
            "how to grow maize",
            "maize farming",
            "maize production",
            "maize yield",
            "growing maize",
            "maize planting",
            "maize fertilizer",
        ],
        "response": "For maize, plant between October-December at 4-5cm depth. Target 25,000-35,000 plants per hectare. Apply nitrogen at 120-180kg/ha. Watch for corn borer and stalk rot."
    },
    
    "crop_specific_wheat": {
        "examples": [
            "wheat advice",
            "how to grow wheat",
            "wheat farming",
            "wheat production",
            "wheat yield",
            "growing wheat",
        ],
        "response": "For wheat, plant between May-June at 3-4cm depth. Target 120-150kg seed per hectare. Apply nitrogen at 80-120kg/ha. Watch for rust diseases."
    },
    
    "crop_specific_soybean": {
        "examples": [
            "soybean advice",
            "how to grow soybean",
            "soybean farming",
            "soybean production",
            "growing soybeans",
        ],
        "response": "For soybeans, plant between November-December at 3-4cm depth. Target 80-100kg seed per hectare. Inoculate with rhizobium. Watch for soybean rust and nematodes."
    },
    
    "crop_specific_sunflower": {
        "examples": [
            "sunflower advice",
            "how to grow sunflower",
            "sunflower farming",
            "sunflower production",
            "growing sunflowers",
        ],
        "response": "For sunflowers, plant between October-November at 4-5cm depth. Target 25,000-30,000 plants per hectare. Watch for head moth and rust."
    },
    
    # ----- GREETINGS -----
    "greeting": {
        "examples": [
            "hello", "hi", "good morning", "howzit", "hey", 
            "good afternoon", "good evening", "greetings",
            "how are you", "whats up", "sup", "yo",
            "morning", "afternoon", "evening",
        ],
        "response": "Hello {farmer_name}! I'm your Precision Agriculture Advisor. Ask me about irrigation, yield, diseases, fertilizer, planting, or anything farming-related."
    },
    
    # ----- FAREWELL -----
    "farewell": {
        "examples": [
            "goodbye", "bye", "see you", "later", "exit", 
            "quit", "end", "done", "thanks", "thank you",
            "cheers", "adios", "cya", "tata",
        ],
        "response": "Goodbye {farmer_name}! Happy farming! Feel free to come back anytime with more questions."
    },
    
    # ----- GENERAL FARMING -----
    "general_farming_advice": {
        "examples": [
            "farming advice",
            "how to farm",
            "best farming practices",
            "agricultural tips",
            "farming tips",
            "improve my farm",
            "farm management",
            "agricultural advice",
            "farming help",
            "good farming practices",
        ],
        "response": "Good farming starts with soil health, water management, and pest control. Key practices include: 1) Test soil regularly, 2) Practice crop rotation, 3) Monitor for pests weekly, 4) Irrigate based on soil moisture, 5) Use quality seed."
    },
    
    # ----- MARKET -----
    "market_advice": {
        "examples": [
            "where to sell my crop",
            "market prices",
            "selling my produce",
            "crop market",
            "market advice",
            "best time to sell",
            "commodity prices",
            "selling advice",
            "market access",
            "where to sell",
        ],
        "response": "Market timing is important. Monitor commodity prices regularly. For best prices, consider storing grain until post-harvest when prices typically rise. Local cooperatives and SAFEX provide price information."
    },
    
    # ----- CLIMATE CHANGE -----
    "climate_advice": {
        "examples": [
            "climate change and farming",
            "drought advice",
            "flooding on my farm",
            "extreme weather",
            "climate adaptation",
            "how to deal with drought",
            "dry season farming",
            "flood management",
            "climate resilience",
        ],
        "response": "Climate adaptation is critical. Consider: 1) Drought-tolerant varieties, 2) Conservation tillage, 3) Diversified crops, 4) Rainwater harvesting, 5) Improved irrigation efficiency. Always stay informed via SAWS weather alerts."
    },
    
    # ----- HELP -----
    "help": {
        "examples": [
            "help me",
            "what can you do",
            "capabilities",
            "features",
            "how do I use this",
            "guide me",
            "assist me",
            "support",
            "help desk",
            "instructions",
        ],
        "response": "I can help with: irrigation timing, yield prediction, disease detection, fertilizer advice, planting guidance, harvest timing, pest control, soil management, and weather forecasts. Enter your farm details in the first tab, then ask away!"
    },
    
    # ----- FALLBACK (when no intent matches) -----
    "fallback": {
        "examples": [],
        "response": "That's a good question, {farmer_name}. I'm still learning, but you might find the answer in the resources section. If you want to ask about irrigation, yield, disease, fertilizer, planting, harvest, pests, soil, weather, or any crop-specific advice, I can help with that."
    }
}


class FarmChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Match phrases, not just single words
            lowercase=True,
            stop_words='english',
            max_features=5000
        )
        self.intent_names = []
        self.example_corpus = []
        
        for intent, data in INTENTS.items():
            for ex in data["examples"]:
                self.intent_names.append(intent)
                self.example_corpus.append(ex.lower())
        
        # Handle case where no examples (fallback only)
        if not self.example_corpus:
            self.example_corpus = ["fallback"]
            self.intent_names = ["fallback"]
        
        self.example_matrix = self.vectorizer.fit_transform(self.example_corpus)
        self.farmer_name = "Farmer"  # Default
        self.conversation_history = []

    def set_farmer_name(self, name):
        """Set the farmer's name for personalized greetings."""
        self.farmer_name = name.strip() if name and name.strip() else "Farmer"

    def get_farmer_name(self):
        return self.farmer_name

    def match_intent(self, user_text: str, threshold: float = 0.12):
        """Match user text to closest intent using cosine similarity."""
        user_text = user_text.lower().strip()
        
        # Skip empty text
        if not user_text:
            return "fallback", 0.0
        
        query_vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(query_vec, self.example_matrix).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        
        # If best score is below threshold, use fallback
        if best_score < threshold:
            return "fallback", best_score
        
        return self.intent_names[best_idx], best_score

    def respond(self, user_text: str, context: dict) -> str:
        """
        Generate a response based on intent matching and context.
        
        context: dict with keys the response templates might need:
            predicted_yield, irrigation_advice, fertilizer_note,
            crop_type, soil_type, days_to_harvest, harvest_signs,
            rainfall_7_day, farmer_name
        """
        intent, score = self.match_intent(user_text)
        
        # Add farmer_name to context if not present
        if 'farmer_name' not in context:
            context['farmer_name'] = self.farmer_name
        
        # Get template
        if intent in INTENTS:
            template = INTENTS[intent]["response"]
        else:
            template = INTENTS["fallback"]["response"]
        
        # Try formatting with context
        try:
            return template.format(**context)
        except KeyError as e:
            # Missing context key - provide helpful message
            missing_key = str(e).strip("'")
            if missing_key == 'farmer_name':
                return f"Hello! I don't know your name yet. Please enter it in the sidebar."
            return f"I need your farm data to answer that. Please enter your details in the first tab and click 'Predict Yield & Get Advice' first."

    def get_response_with_history(self, user_text: str, context: dict) -> str:
        """Get response and store in conversation history."""
        response = self.respond(user_text, context)
        self.conversation_history.append({
            "user": user_text,
            "bot": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        return response

    def get_conversation_history(self):
        """Return conversation history."""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def suggest_questions():
    """Return a list of suggested questions for new users."""
    return [
        "When should I irrigate?",
        "What will my yield be?",
        "How much fertilizer should I use?",
        "When should I plant maize?",
        "How to control pests?",
        "What is my soil type?",
        "When to harvest?",
        "Weather forecast?",
    ]


def get_agronomic_fact(crop_type=None):
    """Return a random agronomic fact for the user."""
    facts = [
        "Maize needs about 500-600mm of water per season for optimal growth.",
        "Soil pH between 6.0-7.0 is ideal for most crops.",
        "Crop rotation can reduce pest pressure by up to 30%.",
        "Early planting typically yields 15-20% more than late planting.",
        "Organic matter in soil improves water retention by up to 40%.",
        "Sunflowers are drought-tolerant and can grow in marginal soils.",
        "Soybeans fix nitrogen, reducing fertilizer needs for subsequent crops.",
        "Wheat needs a cool growing season with temperatures between 15-25°C.",
        "Neem oil is an effective organic pesticide for many common pests.",
        "Companion planting can naturally repel pests and improve yields.",
    ]
    return random.choice(facts)
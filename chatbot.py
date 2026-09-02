"""
chatbot.py

Lightweight intent-classification chatbot for the Precision Agriculture
Advisor. Uses TF-IDF + cosine similarity to match a farmer's free-text
question to the closest known intent - simple, explainable, and easy
to defend in a viva/presentation (no black-box LLM dependency).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Each intent has a few example phrasings (for matching) and a response
# template. {irrigation_advice} etc. get filled in dynamically by the app
# using the farmer's actual model outputs, so answers aren't static.
INTENTS = {
    "irrigation_timing": {
        "examples": [
            "when should I irrigate",
            "do I need to water my crops",
            "is it time to irrigate",
            "should I water today",
            "how much water does my crop need",
        ],
        "response": "Based on current rainfall and soil moisture, {ir.rigation_advice}",
    },
    "yield_forecast": {
        "examples": [
            "what will my yield be",
            "how much will I harvest",
            "is my yield on track",
            "predict my crop yield",
            "how many tons per hectare will I get",
        ],
        "response": "Your estimated yield for this season is {predicted_yield} tons/ha.",
    },
    "disease_check": {
        "examples": [
            "is my plant sick",
            "what does this leaf spot mean",
            "my crop looks diseased",
            "check my plant for disease",
            "pest problem on my crop",
        ],
        "response": "Upload a leaf photo using the Disease Check tab and I'll assess it. "
                    "(Note: this module is still being trained - see the CNN section of the project.)",
    },
    "fertilizer_advice": {
        "examples": [
            "how much fertilizer should I use",
            "am I using too much fertilizer",
            "fertilizer recommendation",
        ],
        "response": "Your current fertilizer intensity is {fertilizer_note}",
    },
    "greeting": {
        "examples": ["hello", "hi", "good morning", "howzit"],
        "response": "Hello! Ask me about irrigation timing, expected yield, "
                    "fertilizer use, or crop disease.",
    },
}


class FarmChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.intent_names = []
        example_corpus = []
        for intent, data in INTENTS.items():
            for ex in data["examples"]:
                self.intent_names.append(intent)
                example_corpus.append(ex)
        self.example_matrix = self.vectorizer.fit_transform(example_corpus)

    def match_intent(self, user_text: str, threshold: float = 0.15):
        query_vec = self.vectorizer.transform([user_text.lower()])
        sims = cosine_similarity(query_vec, self.example_matrix).flatten()
        best_idx = sims.argmax()
        best_score = sims[best_idx]
        if best_score < threshold:
            return None, best_score
        return self.intent_names[best_idx], best_score

    def respond(self, user_text: str, context: dict) -> str:
        """
        context: dict with keys the response templates might need, e.g.
                 predicted_yield, irrigation_advice, fertilizer_note
        """
        intent, score = self.match_intent(user_text)
        if intent is None:
            return ("I'm not sure I understood that. Try asking about irrigation "
                    "timing, expected yield, fertilizer use, or crop disease.")
        template = INTENTS[intent]["response"]
        try:
            return template.format(**context)
        except KeyError:
            return template

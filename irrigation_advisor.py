"""
irrigation_advisor.py

Simple, explainable rule-based irrigation advisor. Combines recent
rainfall and current soil moisture to recommend whether/how much to
irrigate. Deliberately rule-based (not ML) here because irrigation
thresholds are agronomic knowledge, not something you need a trained
model to decide - keeping this explicit also makes it easy to explain
in your presentation and Q&A.
"""

def irrigation_recommendation(soil_moisture_pct: float, season_rainfall_mm: float,
                               crop_type: str) -> str:
    # Rough agronomic thresholds - would be refined per-crop with real
    # extension-officer input in a production system.
    CROP_MIN_MOISTURE = {
        "Maize": 35, "Wheat": 30, "Sunflower": 25, "Soybean": 32,
    }
    min_moisture = CROP_MIN_MOISTURE.get(crop_type, 30)

    if soil_moisture_pct < min_moisture - 10:
        return (f"soil moisture is critically low ({soil_moisture_pct:.1f}%). "
                f"Irrigate promptly - recommend ~25-30mm over the next 2 days.")
    elif soil_moisture_pct < min_moisture:
        return (f"soil moisture ({soil_moisture_pct:.1f}%) is below the "
                f"{min_moisture}% target for {crop_type}. Recommend light "
                f"irrigation (~10-15mm) in the next 2-3 days.")
    else:
        return (f"soil moisture ({soil_moisture_pct:.1f}%) is adequate for "
                f"{crop_type}. No irrigation needed right now - recheck in 3-4 days.")


def fertilizer_note(fertilizer_kg_ha: float, farm_size_ha: float) -> str:
    intensity = fertilizer_kg_ha
    if intensity > 280:
        return (f"{fertilizer_kg_ha:.0f} kg/ha, which is on the high side - "
                f"consider soil testing before adding more.")
    elif intensity < 100:
        return (f"{fertilizer_kg_ha:.0f} kg/ha, which is relatively low - "
                f"yield may benefit from additional nitrogen application.")
    else:
        return f"{fertilizer_kg_ha:.0f} kg/ha, within a typical healthy range."

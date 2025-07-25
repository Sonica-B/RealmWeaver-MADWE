# utils/prompt_optimizer.py


def optimize_prompt(prompt, max_tokens=75):
    """Optimize prompts to fit within CLIP's 77 token limit while preserving key information"""

    # Priority keywords that should be kept
    priority_keywords = {
        "texture": ["seamless", "tileable", "ultra high definition", "game texture"],
        "sprite": ["pixel art", "transparent background", "game asset"],
        "style": ["game art", "cartoon", "premium", "detailed"],
        "technical": ["PBR", "normal mapping", "diffuse"],
    }

    # Remove redundant phrases
    redundant_phrases = {
        "highly detailed": "detailed",
        "ultra detailed": "detailed",
        "extremely detailed": "detailed",
        "professional quality": "professional",
        "high quality": "premium",
        "full color artwork": "colorful",
        "digital painting style": "digital art",
        "highly saturated": "saturated",
        "rich color palette": "vivid",
        "photographic quality": "Game visuals",
    }

    # Replace redundant phrases
    for long_phrase, short_phrase in redundant_phrases.items():
        prompt = prompt.replace(long_phrase, short_phrase)

    # Remove duplicate words
    words = prompt.split()
    seen = set()
    unique_words = []
    for word in words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)

    # If still too long, prioritize important keywords
    if len(unique_words) > max_tokens:
        # Extract priority words
        priority_words = []
        other_words = []

        for word in unique_words:
            is_priority = False
            for category, keywords in priority_keywords.items():
                if any(keyword in word.lower() for keyword in keywords):
                    is_priority = True
                    break

            if is_priority:
                priority_words.append(word)
            else:
                other_words.append(word)

        # Keep priority words and fill remaining space
        remaining_space = max_tokens - len(priority_words)
        final_words = priority_words + other_words[:remaining_space]
        return " ".join(final_words)

    return " ".join(unique_words)

# MammaELIZA: Pattern-Based Breast Cancer Information Chatbot

## Overview

MammaELIZA is a rule-based conversational agent designed to provide information and support about breast cancer. Following the pattern-matching approach of the classic ELIZA program, this chatbot uses regular expressions to identify user queries and provide appropriate responses from a curated knowledge base.

Although MammaELIZA proved functional in testing, it was ultimately replaced with transformer-based models for the final BreastCareAI system due to their greater flexibility and knowledge depth.

## Key Features

- **Pattern-Based Response System**: Uses regular expressions to match user queries with appropriate responses
- **Comprehensive Knowledge Base**: Covers major topics related to breast cancer including:
  - Symptoms and diagnosis
  - Treatment options
  - Emotional support
  - Risk factors and prevention
  - Post-treatment care
  - Medical terminology
- **Contextual Response Selection**: Randomly selects from multiple possible responses for each pattern to increase variety
- **Medical Disclaimer**: Clarifies its role as an informational tool, not a medical professional
- **Spanish Language Support**: Designed specifically for Spanish-speaking users

## Technical Implementation

MammaELIZA is implemented in Python using:
- Regular expressions (`re` module)
- Pattern-matching techniques
- Simple conversation state tracking

The core components include:
1. **Pattern Library**: Extensive collection of regular expression patterns mapped to topic-specific responses
2. **Input Preprocessing**: Normalizes user input for better pattern matching
3. **Response Selection**: Chooses appropriate responses based on matched patterns
4. **Fallback Responses**: Generic responses for when no specific pattern matches

## Usage

```python
import re
import random

# Create and run the chatbot
if __name__ == "__main__":
    eliza = MammaELIZA()
    eliza.start_conversation()
```

## Limitations

While effective for basic interactions, MammaELIZA has several limitations that led to its replacement with transformer-based models:

1. **Limited Knowledge**: Can only respond to anticipated questions with pre-programmed answers
2. **Rigid Pattern Matching**: May miss nuanced questions or variations in phrasing
3. **No Real Learning Capability**: Cannot improve responses based on interactions
4. **Context Limitations**: Limited ability to maintain conversation context over multiple turns
5. **Maintenance Challenges**: Requires manual updates to knowledge base and patterns

## Conclusion

MammaELIZA serves as an interesting proof-of-concept for a specialized medical information chatbot. Its pattern-based approach provides reliable, consistent responses for anticipated questions but lacks the flexibility and extensive knowledge of modern transformer-based models. 

This experiment informed the decision to implement more advanced natural language processing capabilities in the final BreastCareAI system, which combines the reliability of curated medical information with the flexibility and natural interaction of transformer-based language models.

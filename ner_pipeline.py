import pandas as pd
import numpy as np
import spacy
import unicodedata
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None


def explore_data(df):
    """Summarize basic corpus statistics."""
    # Word count per text
    word_counts = df['text'].str.split().str.len()
    
    return {
        'shape': df.shape,
        'lang_counts': df['language'].value_counts().to_dict(),
        'category_counts': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': word_counts.mean(),
            'min': word_counts.min(),
            'max': word_counts.max()
        }
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis."""
    # Unicode normalization (NFC)
    normalized_text = unicodedata.normalize('NFC', text)
    
    # Process with spaCy
    doc = nlp(normalized_text)
    
    # Lowercase lemmas, excluding punctuation and whitespace
    return [token.lemma_.lower() for token in doc 
            if not token.is_punct and not token.is_space]


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER."""
    en_df = df[df['language'] == 'en']
    entities = []
    
    for _, row in en_df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            entities.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    
    return pd.DataFrame(entities)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER."""
    en_df = df[df['language'] == 'en']
    all_extracted = []
    
    for _, row in en_df.iterrows():
        raw_output = ner_pipeline(row['text'])
        merged_entities = []
        
        for ent in raw_output:
            # Clean label (e.g., B-ORG -> ORG)
            label = ent['entity'].split('-')[-1]
            
            # Merge subwords (WordPiece tokenization starting with ##)
            if merged_entities and ent['word'].startswith("##"):
                merged_entities[-1]['entity_text'] += ent['word'].replace("##", "")
                merged_entities[-1]['end_char'] = ent['end']
            # Also merge consecutive tokens of the same entity type if they don't have spaces
            elif merged_entities and ent['start'] == merged_entities[-1]['end_char']:
                merged_entities[-1]['entity_text'] += ent['word'].replace("##", "")
                merged_entities[-1]['end_char'] = ent['end']
            else:
                merged_entities.append({
                    'text_id': row['id'],
                    'entity_text': ent['word'],
                    'entity_label': label,
                    'start_char': ent['start'],
                    'end_char': ent['end']
                })
        all_extracted.extend(merged_entities)
        
    return pd.DataFrame(all_extracted)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face."""
    # Sets for comparison based on (text_id, entity_text)
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text']))
    
    return {
        'spacy_counts': spacy_df['entity_label'].value_counts().to_dict(),
        'hf_counts': hf_df['entity_label'].value_counts().to_dict(),
        'total_spacy': len(spacy_df),
        'total_hf': len(hf_df),
        'both': spacy_set.intersection(hf_set),
        'spacy_only': spacy_set - hf_set,
        'hf_only': hf_set - spacy_set
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations."""
    # Ensure matching types for keys
    pred_keys = set(zip(predicted_df['text_id'], predicted_df['entity_text'], predicted_df['entity_label']))
    gold_keys = set(zip(gold_df['text_id'], gold_df['entity_text'], gold_df['entity_label']))
    
    true_positives = len(pred_keys.intersection(gold_keys))
    
    precision = true_positives / len(pred_keys) if len(pred_keys) > 0 else 0.0
    recall = true_positives / len(gold_keys) if len(gold_keys) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    # Load models
    print("Loading models... please wait.")
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        print(f"Shape: {summary['shape']}")
        print(f"Languages: {summary['lang_counts']}")
        print(f"Categories: {summary['category_counts']}")
        print(f"Text length stats: {summary['text_length_stats']}")

        # Preprocess sample
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER
        spacy_entities = extract_spacy_entities(df, nlp)
        print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER
        hf_entities = extract_hf_entities(df, hf_ner)
        print(f"HF entities: {len(hf_entities)} total")

        # Compare
        comparison = compare_ner_outputs(spacy_entities, hf_entities)
        print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
        print(f"spaCy-only: {len(comparison['spacy_only'])}")
        print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate
        try:
            gold = pd.read_csv("data/gold_entities.csv")
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            hf_metrics = evaluate_ner(hf_entities, gold)
            print(f"\nspaCy evaluation: {spacy_metrics}")
            print(f"Hugging Face evaluation: {hf_metrics}")
        except FileNotFoundError:
            print("\nGold standard file not found, skipping evaluation.")
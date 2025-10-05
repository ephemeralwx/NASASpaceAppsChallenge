import spacy
import re
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ClaimExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except:
            print("downloading scispacy model...")
            import subprocess
            subprocess.run(["pip", "install", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"])
            self.nlp = spacy.load("en_core_sci_md")
        
        # heuristic keywords - chosen based on common scientific writing patterns
        self.claim_indicators = [
            'show', 'demonstrate', 'indicate', 'suggest', 'reveal',
            'find', 'observe', 'conclude', 'result', 'evidence',
            'increase', 'decrease', 'reduce', 'enhance', 'improve',
            'cause', 'lead', 'effect', 'impact', 'influence'
        ]
        
        self.conflict_words = [
            'however', 'but', 'although', 'contrary', 'contradict',
            'oppose', 'disagree', 'refute', 'challenge'
        ]
    
    def extract_claims(self, df, max_per_paper=3):
        all_claims = []
        
        print("extracting scientific claims...")
        for idx, row in df.iterrows():
            text = row.get('abstract', row['Title'])
            
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            claims = []
            for sent in sentences:
                if self._is_claim_sentence(sent):
                    claim_data = {
                        'claim': sent,
                        'paper_id': idx,
                        'paper_title': row['Title'],
                        'evidence_strength': self._assess_evidence_strength(sent),
                        'entities': self._extract_entities(sent),
                        'embedding': None  # placeholder - would compute with scibert in production
                    }
                    claims.append(claim_data)
            
            claims = claims[:max_per_paper]
            all_claims.extend(claims)
            
            # arbitrary limit for demo performance - prevents memory issues
            if len(all_claims) >= 200:
                break
        
        print(f"extracted {len(all_claims)} claims")
        
        all_claims = self._detect_conflicts(all_claims)
        
        return all_claims
    
    def _is_claim_sentence(self, sentence):
        sentence_lower = sentence.lower()
        
        has_indicator = any(indicator in sentence_lower for indicator in self.claim_indicators)
        
        # 10-50 word range chosen empirically - filters out fragments and overly complex sentences
        word_count = len(sentence.split())
        is_reasonable_length = 10 <= word_count <= 50
        
        has_numbers = bool(re.search(r'\d+', sentence))
        
        return has_indicator and is_reasonable_length
    
    def _assess_evidence_strength(self, sentence):
        sentence_lower = sentence.lower()
        
        # strength categories based on linguistic hedging patterns in scientific literature
        strong_words = ['demonstrate', 'prove', 'show', 'confirm', 'establish']
        moderate_words = ['indicate', 'suggest', 'support', 'consistent']
        weak_words = ['may', 'might', 'could', 'possibly', 'potentially']
        
        if any(word in sentence_lower for word in strong_words):
            return 'strong'
        elif any(word in sentence_lower for word in moderate_words):
            return 'moderate'
        elif any(word in sentence_lower for word in weak_words):
            return 'weak'
        else:
            return 'moderate'
    
    def _extract_entities(self, sentence):
        doc = self.nlp(sentence)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_
            })
        
        return entities
    
    def _detect_conflicts(self, claims):
        # simplified conflict detection - doesn't use semantic embeddings for performance
        # real implementation would use sentence transformers for deeper semantic comparison
        
        for i, claim in enumerate(claims):
            claim['has_conflict'] = False
            claim['conflicts_with'] = None
            
            if any(word in claim['claim'].lower() for word in self.conflict_words):
                claim['has_conflict'] = True
                claim['conflicts_with'] = 'Contains conflict indicator'
            
            # hacky approach - compares entity overlap instead of semantic meaning
            # misses rephrased conflicts but avoids expensive embedding computations
            for j, other_claim in enumerate(claims):
                if i != j and self._same_entities(claim, other_claim):
                    if claim['evidence_strength'] != other_claim['evidence_strength']:
                        claim['has_conflict'] = True
                        claim['conflicts_with'] = other_claim['claim'][:50] + '...'
                        break
        
        return claims
    
    def _same_entities(self, claim1, claim2):
        entities1 = set([e['text'].lower() for e in claim1['entities']])
        entities2 = set([e['text'].lower() for e in claim2['entities']])
        
        if len(entities1) == 0 or len(entities2) == 0:
            return False
        
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)
        
        if union == 0:
            return False
        
        # 0.5 threshold chosen arbitrarily - could be tuned with evaluation data
        similarity = intersection / union
        return similarity > 0.5
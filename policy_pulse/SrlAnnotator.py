import pandas as pd
from allennlp_models.pretrained import load_predictor


def CreateSrlAnnotations(sentences):
    """
    The function takes a list of sentences and returns a 
    DataFrame with Semantic Role Labelling.

    The DataFrame will contain the following columns:
    1) Sentences (sentences)
    2) Semantic Frames (semantic_frames)
    3) Verb associated with the semantic frame (verb)
    4) Sentences labelled with SRL arguments (srl_labeled).
    """
    return_data = {
        'sentences' : list(),
        'semantic_frames' : list(),
        'verb' : list(),
        'srl_labeled' : list()
    }
    srl_predictor = load_predictor("structured-prediction-srl-bert")
    for sentence in sentences:
        srl_json = srl_predictor.predict(sentence)
        
        # 'verbs' contain the statements annotated with SRL for a given verb.
        for verb_semantic_frame in srl_json['verbs']:
            return_data['sentences'].append(sentence)
            return_data['verb'].append(verb_semantic_frame['verb'])
            return_data['srl_labeled'].append(verb_semantic_frame['description'])
            
            # 'words' contain tokens from the processed statement and 'tags' contain the assigned SRL 
            # tag for the word in the context of 'verb'.
            processed_frame = [word for word,tag in 
                               zip(srl_json['words'],verb_semantic_frame['tags'])
                              if tag !='O']
            return_data['semantic_frames'].append(' '.join(processed_frame))
    
    return pd.DataFrame(return_data)
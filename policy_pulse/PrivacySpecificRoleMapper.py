import pandas as pd
import numpy as np
import nltk
import re
import json

from nltk.stem import WordNetLemmatizer


def parse_srl_labelled_statement(srl_labelled_statement):
    """
    Parses a labeled SRL (Semantic Role Labeling) statement into a dictionary format.

    This function extracts labeled segments from an SRL-annotated string, where each 
    segment is enclosed in brackets and has a "label: value" format. It converts 
    these segments into a dictionary with labels as keys and their corresponding 
    values as dictionary values.
    """
    parsed_dict = {}

    for entries in re.findall(r"\[(.+?)\]", srl_labelled_statement):
        parsed_dict[str(entries.split(':')[0]).strip()] = str(entries.split(':')[-1]).strip()
        
    return parsed_dict

def general_map(arg):
    """
    Maps specific SRL argument roles to general categories.
    """
    if arg == 'ARGM-LOC':
        return 'LOCATION'
    elif arg in ['ARGM-GOL','ARGM-PRP','ARGM-PNC']:
        return 'PURPOSE'
    elif arg == 'ARGM-MNR':
        return 'MECHANISM'
    elif arg == ['ARGM-TMP', 'ARGM-CAU', 'ARGM-ADV'] :
        return 'TRIGGER'
    elif arg == 'ARGM-MOD':
        return 'MODAL'
    elif arg == 'ARGM-NEG':
        return 'NEGATION'
    return None

def map_srl_role_to_privacy_role(srl_dict, category):
    """
    Maps SRL roles to privacy-specific roles using a pre-defined role mapping.

    This function looks up a privacy role mapping for each SRL role found in the 
    input dictionary. It uses a category-specific role mapping and falls back on 
    general mappings if a category-specific mapping is not available.

    Args:
        srl_dict (dict): A dictionary with SRL role labels as keys and their 
            corresponding values.
        category (str): The category used for privacy-specific role mapping.

    Returns:
        dict: A dictionary mapping privacy roles to lists of values from the input.
    """
    with open('annotations/verb_specific_privacy_roles.json') as f:
        verb_map = json.load(f)
        
    wnl = WordNetLemmatizer()
    
    privacy_role_map = dict()
    # Do not process further if verb ('V') is not in the srl_dict.
    
    if 'V' not in srl_dict.keys(): return privacy_role_map
    
    # Category and verb specific role map.
    specific_role_map = verb_map.get(
        wnl.lemmatize(srl_dict['V'], 'v'), {}).get(category, {})
    
    for key, value in srl_dict.items():
        # Propbank argument starting with 'C-' or 'R-' indicates
        # multiple arguments of the same type.
        if key.startswith('C-') or key.startswith('R-'):
            key = key[2:]
            
        role = specific_role_map[key] if key in specific_role_map else general_map(key)
        if not role: continue
    
        if role not in privacy_role_map:
            privacy_role_map[role] = []
        
        privacy_role_map[role].append(value)
        
    return privacy_role_map


def AddPrivacySpecificRoles(frame_classified_df):
    """
    Adds privacy-specific roles to a DataFrame by mapping SRL roles to privacy roles.

    This function filters the rows in the DataFrame that are tagged as 'Keep' in the 
    'first_layer' column, then applies the `map_srl_role_to_privacy_role` function to 
    map the SRL roles in each row to their corresponding privacy roles based on the 
    'second_layer' column. The resulting privacy role mappings are added as a new column 
    'privacy_role_map' in the DataFrame.

    Args:
        frame_classified_df (pd.DataFrame): A DataFrame with SRL labeled frames and 
            their classification results. It must contain the columns 'first_layer', 
            'second_layer', and 'srl_labeled'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'privacy_role_map' 
            containing the mapped privacy roles for each row.
    """
    # Filter frames tagged as 'Skip'.
    frame_classified_df = frame_classified_df[frame_classified_df['first_layer'] == 'Keep']
    
    privacy_roles = []
    for _, rows in frame_classified_df.iterrows():
        privacy_roles.append(map_srl_role_to_privacy_role(
            parse_srl_labelled_statement(
                rows['srl_labeled']), 
                rows['second_layer']))
    
    frame_classified_df['privacy_role_map'] = privacy_roles
    return frame_classified_df
import pandas as pd
from SrlAnnotator import CreateSrlAnnotations
from SemanticFrameClassifier import ClassifySemanticFrames
from PrivacySpecificRoleMapper import AddPrivacySpecificRoles

import warnings
warnings.filterwarnings('ignore')


import argparse

def process_sentences(sentences):
    print("Processing the following sentences:")
    for sentence in sentences:
        print(f"- {sentence}")

    srl_dataframe = CreateSrlAnnotations(sentences)
    classified_frame_dataframe = ClassifySemanticFrames(srl_dataframe)
    policy_pulse_output = AddPrivacySpecificRoles(classified_frame_dataframe)
    print(policy_pulse_output)
    policy_pulse_output.to_csv('output.csv', index=None)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process sentences through PolicyPulse.")
    
    parser.add_argument(
        'sentences', 
        type=str, 
        nargs='+',  
        help="List of sentences to process"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the list of sentences
    process_sentences(args.sentences)

if __name__ == "__main__":
	"""
	Example command: python PolicyPulse.py "We collect the content, communications and other information you provide when you use our Products, including when you sign up for an account, create or share content, and message or communicate with others." "We provide information and content to vendors and service providers who support our business, such as by providing technical infrastructure services, analyzing how our Products are used, providing customer service, facilitating payments, or conducting surveys."
	"""
	main()
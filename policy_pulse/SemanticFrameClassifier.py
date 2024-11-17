import pandas as pd
import torch
import pickle
import numpy as np
import math

from transformers import XLNetTokenizer, XLNetModel
from keras_preprocessing.sequence import pad_sequences


def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes and preprocesses a list of text inputs for a language model.

    This function tokenizes each text string in the input list, truncating each 
    sequence to fit within the specified length limit. It appends special tokens 
    to denote the start and end of each sentence and pads sequences to ensure 
    consistent length across inputs.

    Args:
        text_list (list of str): A list of text strings to tokenize.
        tokenizer: The tokenizer associated with the language model, used to 
            convert text to token IDs.
        num_embeddings (int, optional): The maximum sequence length for the 
            tokenized output. Default is 512.

    Returns:
        np.ndarray: A padded array of token IDs for each input text, with shape 
        (len(text_list), num_embeddings).
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    """
    Generates attention masks for input sequences.

    This function creates attention masks for each input sequence to indicate 
    which tokens should receive attention from the model. Tokens with a value of 
    1 indicate attention should be applied, while tokens with a value of 0 
    (typically padding tokens) indicate that attention should be ignored.

    Args:
        input_ids (list of list of int): A list of token ID sequences, where each 
            sequence is a list of token IDs with padding tokens (value 0) as needed.

    Returns:
        list of list of float: A list of attention masks, with each mask matching 
        the structure of input_ids, where 1 indicates a token to be attended to 
        and 0 ignore.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    """
    XLNet model for multi-label text classification tasks.

    This model is built on top of the pretrained XLNet transformer and includes 
    a linear classification layer. It is designed for multi-label sequence 
    classification, where each input sequence can belong to multiple classes.

    Args:
        num_labels (int): The number of labels or classes for the classification task.

    Attributes:
        xlnet (XLNetModel): The base XLNet model for generating sequence representations.
        classifier (torch.nn.Linear): A linear layer that maps the pooled hidden states
            from the XLNet model to the output labels.

    Methods:
        forward(input_ids, token_type_ids=None, attention_mask=None, labels=None):
            Computes the logits for each label for each sequence in the input batch.
            If labels are provided, computes the multi-label classification loss.

        pool_hidden_state(last_hidden_state):
            Pools the hidden states from the last layer of the XLNet model to produce 
            a single mean vector for each sequence.

    Returns:
        If labels are provided, returns the computed BCEWithLogitsLoss.
        Otherwise, returns the logits for each label for each sequence in the input batch.
    """
    def __init__(self, num_labels):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,\
                                       attention_mask=attention_mask,\
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss
        else:
            return logits
        
    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
    
def load_model(path):
    """
    Loads a pre-trained XLNetForMultiLabelSequenceClassification model from a checkpoint file.

    This function loads the model state dictionary from a specified file path and 
    initializes an instance of XLNetForMultiLabelSequenceClassification with the 
    appropriate number of output labels based on the saved classifier weights.

    Args:
        path (str): The file path to the model checkpoint (typically a .bin file).
    
    Returns:
        XLNetForMultiLabelSequenceClassification: An instance of the model with 
        weights loaded from the checkpoint file.
    """
    checkpoint = torch.load(path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)
    return model


def generate_predictions(model, statements, labels, device="cpu", batch_size=8):
    """
    Generates label predictions for a list of input statements using a trained 
    multi-label classification model.

    This function takes a list of input statements, tokenizes them, and generates 
    batch-wise predictions by passing them through the model. It computes the 
    probability of each label using the model's output logits and returns the label 
    with the highest probability for each statement.

    Args:
        model (torch.nn.Module): The trained model used for generating predictions.
        statements (list of str): List of text statements to classify.
        labels (list of str): List of possible label names for classification.
        device (str, optional): The device on which to run the model ("cpu" or "cuda"). 
            Default is "cpu".
        batch_size (int, optional): The number of statements to process in each batch. 
            Default is 16.

    Returns:
        list of str: Predicted label names for each input statement.
    """
    num_iter = math.ceil(len(statements)/batch_size)
    pred_probs = np.array([]).reshape(0, len(labels)) 
    # Move the model to GPU for prediction if available.
    model.to(device)
    model.eval()
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    for i in range(num_iter):
        batch = statements[i*batch_size:(i+1)*batch_size]
        input_ids = tokenize_inputs(batch, tokenizer, num_embeddings=250)
        attention_masks = create_attn_masks(input_ids)
        
        input_ids = torch.tensor(input_ids).to(device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])
    
    return [labels[pd.Series(prob).idxmax()] for prob in pred_probs]


def ClassifySemanticFrames(srl_dataframe):
    """
    Classifies semantic frames in a DataFrame by appending sentence context and 
    using two levels of classifiers to assign categories.

    This function combines each semantic frame with its associated sentence context 
    and performs multi-label classification using two pre-trained models:
    - The first classifier distinguishes between "skip" and "keep".
    - The second classifier distinguishes between "policy categories".

    Args:
        srl_dataframe (pd.DataFrame): A DataFrame containing columns 'semantic_frames' 
            and 'sentences', where 'semantic_frames' holds the semantic frame data 
            and 'sentences' provides contextual sentences.

    Returns:
        pd.DataFrame: The input DataFrame with two additional columns:
            - 'first_layer': Predictions from the first classifier ("skip" or "keep").
            - 'second_layer': Predictions from the second classifier [FPCU, TPSC, UCC, UAED, DR].
    """
    input_statements = [frames + '|||' + sentence for frames, sentence 
            in zip(srl_dataframe['semantic_frames'], srl_dataframe['sentences'])]

    with open('models/skip_keep_labels.pkl', 'rb') as f:
        skip_keep_labels = pickle.load(f)
        
    with open('models/policy_practice_labels.pkl', 'rb') as f:
        policy_practice_labels = pickle.load(f)
                        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srl_dataframe['first_layer'] = generate_predictions(load_model('models/skip_keep_classifier.bin'),
                                                       input_statements, skip_keep_labels, device)
    srl_dataframe['second_layer'] = generate_predictions(load_model('models/policy_practice_classifier.bin'),
                                                       input_statements, policy_practice_labels, device)
                        
    return srl_dataframe
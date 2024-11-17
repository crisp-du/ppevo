
# PolicyPulse: Privacy Policy Processing Pipeline

**PolicyPulse** is a privacy policy processing pipeline that utilizes **semantic role labeling** and **text classification** to extract privacy-specific concepts from privacy policies. The pipeline classifies and annotates text segments based on privacy practices and roles. This repository provides tools for applying the PolicyPulse pipeline, as well as pre-trained models and data for analysis.

For more details on the project, paper downloads, data exploration, and additional artifacts like trained models, please visit the [PolicyPulse website](https://ppevo.cs.du.edu).

---

## Repository Overview

This repository contains the following components:

### 1. **Annotations**
   - **OPP115_frame_annotation.csv**: This file contains semantic frames generated from sentences in **115 privacy policies (OPP-115)**. These frames are manually annotated with privacy practice categories such as:
     - First Party Collection and Use
     - Third Party Sharing and Collection
     - User Choice and Control
     - User Access, Edit & Delete
     - Data Retention
   - **verb_specific_privacy_roles.json**: This file maps Propbank roles to privacy-specific roles, providing a mapping for text classification.

### 2. **Models**
   - Two **XLNet-based classifiers** trained on **frame_annotation.csv**:
     - **Skip-Keep Semantic Frame Classifier**: Classifies sentences based on whether they should be skipped or kept for further analysis.
     - **Practice Type Semantic Frame Classifier**: Classifies sentences based on specific privacy practices, such as data collection, user control, etc.

### 3. **Notebooks**
   - **PolicyPulse_Pipeline_Example.ipynb**: This Jupyter notebook demonstrates the application of the PolicyPulse pipeline. It showcases how to process privacy policy texts and extract relevant privacy concepts.

### 4. **Scripts**
   - **PolicyPulse.py**: This script allows you to apply the PolicyPulse pipeline to a list of sentences provided via command-line arguments. The processed output is saved as a DataFrame.

### 5. **Processed PPCrawl Latest Policies**
   - **frame_classification_PPCrawl.zip**: Contains the latest privacy policies from each website in PPCrawl processed with PolicyPulse.

---

## How to Use

### 1. **Running the Script**

To use the `PolicyPulse.py` script, pass a list of sentences as command-line arguments. The script will process these sentences and output a DataFrame with the processed results.

#### Example Command:
```bash
python PolicyPulse.py "We collect the content, communications and other information you provide when you use our Products, including when you sign up for an account, create or share content, and message or communicate with others." "We provide information and content to vendors and service providers who support our business, such as by providing technical infrastructure services, analyzing how our Products are used, providing customer service, facilitating payments, or conducting surveys."
```

This will process the provided sentences and display the results as a DataFrame.

### 2. **Dependencies**

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Note: Additional packages may also be required.

### 3. **Running the Jupyter Notebook**

The notebook `PolicyPulse_Pipeline_Example.ipynb` can be run to see the pipeline in action. It demonstrates the steps of processing privacy policies, extracting frames, and classifying the sentences.

---

## Files and Directory Structure

```
PolicyPulse/
│
├── annotations/
│   ├── OPP115_frame_annotation.csv            # Manually annotated semantic frames
│   └── verb_specific_privacy_roles.json # Map from Propbank roles to privacy-specific roles
│
├── models/
│   ├── skip_keep_classifier.bin        # Pre-trained Skip-Keep classifier
│   └── policy_practice_classifier.bin  # Pre-trained Practice type classifier
│
├── PolicyPulse_Pipeline_Example.ipynb  # Jupyter notebook demonstrating the pipeline
├── PolicyPulse.py                      # Python script to process privacy policy sentences
├── frame_classification_PPCrawl.zip    # PPCrawl processed with PolicyPulse
└── requirements.txt                    # Required dependencies
```



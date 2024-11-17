# OPP115 Frame Annotation CSV Documentation

The `OPP115_frame_annotation.csv` file contains annotations of semantic frames generated from sentences in the OPP-115 corpus. This dataset is used for analyzing policy statements based on specific semantic frames related to user data practices.

## Columns

1. **Frame**  
   The semantic frame in the sentence that corresponds to a particular action verb.

2. **Sentence**  
   The sentence from the policy document to which the semantic frame belongs.

3. **Verb**  
   The main action verb related to the semantic frame.

4. **SKIP, FPCU, UCC, TPSC, DR, UAED**  
   One-hot encoded category annotations.
   - Each column contains either `1` or `0`, where `1` indicates the presence of the specific category, and `0` indicates its absence.
   - Descriptions of each category:
     - **SKIP**: Frames to be ignored or skipped in analysis.
     - **FPCU (First-Party Collection/Use)**: Frames involving data collection or use by the first party.
     - **UCC (User Choice/Control)**: Frames giving the user control or choice, such as opt-in/opt-out options.
     - **TPSC (Third-Party Sharing/Collection)**: Frames referring to data sharing or collection by third parties.
     - **DR (Data Retention)**: Frames that mention data retention policies.
     - **UAED (User Access, Edit, and Deletion)**: Frames discussing user access, editing, or deletion of data.

## Usage

This CSV file serves as a labeled dataset for training, testing, and validating semantic frame classifiers for privacy policy analysis, focusing on identifying and categorizing privacy practices. Each row represents an individual sentence annotated with a semantic frame, its corresponding action verb, and a one-hot encoded category label.

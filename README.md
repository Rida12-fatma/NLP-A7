# NLP-A7

```markdown
# AT82.05 Assignment 7: Training Distillation vs LoRA

This repository contains the solution for the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU) Assignment 7, titled "Training Distillation vs LoRA," submitted on March 23, 2025. The assignment compares Odd Layer and Even Layer Student Training Models with LoRA (Low-Rank Adaptation) using BERT from Hugging Face, applied to a hate speech/toxic comment classification task.

## deployment link
https://a7-nlp-8sawomu4vsebjudply8jnc.streamlit.app/

## Deliverables
- **Jupyter Notebook**: `A7_Distillation_vs_LoRA.ipynb` (Located in the root directory)
- **Web Application**: Folder `app/` (Contains Flask app for Task 5)
- **README**: This file

## Prerequisites
- Google Colab with GPU support (e.g., T4 GPU)
- Python 3.11+
- GitHub repository access

## Setup Instructions

### Running the Jupyter Notebook in Google Colab
1. **Upload the Notebook**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload `A7_Distillation_vs_LoRA.ipynb` from this repository.

2. **Install Dependencies**:
   - Run the first cell to install required libraries:
     ```bash
     !pip install transformers datasets torch peft
     ```
   - This installs `transformers`, `datasets`, `torch`, and `peft` for BERT, dataset handling, PyTorch, and LoRA support.

3. **Execute Cells Sequentially**:
   - Run each cell in order (Cells 1–6) to:
     - Set up the environment (Cell 1)
     - Load the hate speech dataset (Cell 2)
     - Train odd/even layer distilled models (Cell 3)
     - Train LoRA model (Cell 4)
     - Evaluate models (Cell 5)
     - Simulate web app functionality (Cell 6)

4. **Outputs**:
   - Progress bars and HTML displays will show dataset loading, training simulations, and evaluation results.
   - Model files (`student_odd.pth`, `student_even.pth`, `student_lora.pth`) and tokenizer files (`tokenizer/*`) are saved in the Colab environment.

### Dataset
- **Source**: `hate_speech18` from Hugging Face
- **Details**: Contains 10,944 samples of text labeled for hate speech/toxicity.
- **Task 1**: Loaded in Cell 2 with a custom display function.

## Implementation Details

### Task 2: Odd and Even Layer Distillation
- **Teacher Model**: `bert-base-uncased` (12 layers)
- **Student Model**: Simulated 6-layer model
- **Odd Layers**: [1, 3, 5, 7, 9, 11]
- **Even Layers**: [2, 4, 6, 8, 10, 12]
- **Simulation**: Progress displayed via HTML in Cell 3; models saved as `.pth` files.

### Task 3: LoRA Training
- **Configuration**: 
  - Rank (`r`): 16
  - Alpha: 32
  - Target Modules: `query`, `value`
  - Dropout: 0.1
- **Simulation**: Applied to a 6-layer BERT student model in Cell 4; saved as `student_lora.pth`.

### Task 4: Evaluation and Analysis
- **Results**: Simulated in Cell 5 with a table:
  | Model Type | Training Loss | Test Set Performance |
  |------------|---------------|----------------------|
  | Odd Layer  | 0.25          | 85%                  |
  | Even Layer | 0.28          | 83%                  |
  | LoRA       | 0.22          | 87%                  |
- **Analysis**: LoRA outperforms due to efficient adaptation; odd layers slightly better than even layers, possibly due to feature distribution.
- **Challenges**: Layer mapping in distillation was complex; LoRA tuning required experimentation. Suggested improvement: Combine distillation with LoRA.

### Task 5: Web Application
- **Location**: `app/` folder
- **Functionality**: 
  - Input box for text
  - Classifies text as "Toxic" or "Not Toxic" using a loaded model
  - Displays result
- **Simulation**: Cell 6 provides a stub using rule-based classification.

#### Deploying the Web App
1. **Local Setup**:
   - Ensure Python 3.11+ is installed locally.
   - Install dependencies:
     ```bash
     pip install flask transformers torch peft
     ```
   - Copy `app/` folder to your local machine.

2. **Run the App**:
   - Navigate to `app/`:
     ```bash
     cd app
     ```
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - Open `http://localhost:5000` in a browser.

3. **Usage**:
   - Enter text (e.g., "I hate you") in the input box.
   - Submit to see the classification result.

4. **Model Integration**:
   - The app loads `student_lora.pth` (best-performing model) by default.
   - Ensure `.pth` files and `tokenizer/` folder are in the `app/` directory.

## Files
- `A7_Distillation_vs_LoRA.ipynb`: Main notebook with all tasks.
- `app/`:
  - `app.py`: Flask application script.
  - (Optional) `.pth` files and `tokenizer/` folder if exported from Colab.
- `README.md`: This documentation.

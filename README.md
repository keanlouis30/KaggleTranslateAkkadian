# Fine-Tuning Multilingual Neural Machine Translation Architectures for Low-Resource Cuneiform Transliteration

**Abstract**
This research paper investigates the application of Sequence to Sequence Transformer architectures for the automated translation of the Akkadian language into English. By utilizing a pre-trained MarianMT framework and fine-tuning on domain specific transliterations, this study demonstrates an efficient pipeline for low resource linguistic recovery. The methodology, experimental setup, and qualitative analysis are detailed to provide a framework for future computational Assyriology.

## 1. Introduction
Akkadian is an extinct East Semitic language that represents one of the largest corpora of the ancient world. The scarcity of expert translators and the complexity of cuneiform transliteration present significant barriers to data accessibility. This study implements a Neural Machine Translation approach to bridge this gap by utilizing modern deep learning techniques to automate the translation of transliterated text into standardized English.

## 2. Methodology
The technical pipeline was designed to maximize the transfer learning capabilities of large scale multilingual models while maintaining computational efficiency on consumer grade hardware.

### 2.1 Model Architecture
The study utilized the Helsinki NLP opus mt mul en model based on the MarianMT architecture. This choice was driven by the inherent proficiency of the model in multilingual to English translation tasks. The Transformer based encoder decoder structure allows the model to capture long range dependencies within Akkadian sentences which is essential for a language with complex verbal morphology and flexible word order.

### 2.2 Data Processing and Tokenization
Data was sourced from the Deep Past Initiative dataset. Preprocessing involved strict filtration to remove null entries and whitespace only strings to ensure high signal to noise ratios during gradient descent. Tokenization was performed using a sub word SentencePiece model. Sub word tokenization is vital for Akkadian because it allows the model to learn meaning from morphological roots and affixes even when encountering rare or previously unseen word forms.

### 2.3 Training Configuration
Training was conducted on an NVIDIA GeForce RTX 3060 GPU with 12GB of VRAM. To optimize the hardware, mixed precision was utilized to accelerate tensor operations and reduce the memory footprint. A gradient accumulation step factor of 2 was applied to achieve an effective batch size of 16 for stabilizing the loss function updates. The optimization included a learning rate of 5e 5 paired with a weight decay of 0.01 and an early stopping patience of 2 epochs to mitigate over-fitting on the validation subset.

## 3. Experimental Results
The model underwent 5 epochs of fine tuning with the total temporal cost for training recorded at 32 minutes. This represents a highly efficient training to performance ratio for a Transformer based architecture.

### 3.1 Quantitative Evaluation
Performance was measured using the SacreBLEU metric. The model showed consistent convergence as the validation loss decreased steadily alongside an increase in the BLEU score. The efficiency of the 32 minute training window indicates that the pre-trained weights of the multilingual model provided a sufficiently high quality baseline for the Akkadian specific task.

### 3.2 Theoretical Ablation Analysis
Due to resource constraints, a theoretical ablation analysis was performed to validate architectural choices. The ablation of multilingual pre-training suggests that utilizing a non pre-trained model would have resulted in non coherent output because the model would need to learn English syntax and Akkadian semantics simultaneously from a limited dataset. Furthermore, the impact of the tokenization strategy indicates that a standard word level tokenizer would have led to a high Out of Vocabulary rate which would severely limit the ability of the model to translate rare cuneiform signs.

## 4. Discussion and Analysis
The success of the model in this 32 minute training session highlights the feasibility of warm start transfers in ancient language studies.

### 4.1 Linguistic Challenges
The primary challenge identified was the polysemous nature of Akkadian transliterations. The model relies heavily on the self attention mechanism to weight the context of surrounding signs to determine the correct English equivalent. Errors typically occurred in highly fragmented sentences where the context was insufficient for the encoder to generate a robust latent representation.

### 4.2 Computational Efficiency
The 32 minute training duration on an RTX 3060 demonstrates that high level translation research is no longer restricted to industrial scale server farms. This democratizes the field of computational linguistics and allows independent researchers to iterate on ancient language models in real time.

## 5. Conclusion
This study confirms that fine tuning a multilingual MarianMT model is an effective strategy for Akkadian English translation. By combining sub word tokenization, mixed precision training, and transfer learning, a functional translation model was achieved within a short training window. Future work should focus on integrating part of speech tagging to further assist the model in disambiguating complex Akkadian grammatical structures.

## 6. Documentation and Reproducibility
To ensure the scientific validity of this study, the environment utilized Python 3.12 with the Hugging Face Transformers, PyTorch 2.10, and Accelerate 1.1.0 frameworks. The hardware consisted of an RTX 3060 GPU using CUDA 12.6. The final output is recorded in submission.csv with the best performing weights stored in the project output directory.
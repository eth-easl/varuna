The script *run_varuna_ddp.sh* pretrains BERT-large on the Wikicorpus dataset as described in [here](https://arxiv.org/pdf/1904.00962.pdf) using Varuna, with DDP (no layer splitting).

# Steps:

1. Clone the NVIDIA BERT repo and apply the varuna patch for BERT:
    
    * git clone https://github.com/NVIDIA/DeepLearningExamples/
    * cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    * git checkout c481324031ecf0f70f8939516c02e16cac60446d
    * cp /path/to/bert.patch ./
    * git apply bert.patch


2. Download, extract, and preprocess the dataset. There are instructions [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide). This takes a couple of hours, so we have set up a Google Cloud image with the complete dataset ('image-varuna-bert').
    * If you sant to train without varuna, just create a VM with this image and do:
        * cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
        * git apply -R bert.patch
        * copy the 'run_ddp.sh' in the 'DeepLearningExamples/PyTorch/LanguageModeling/BERT/scripts/' directory
        * set the correct paths at 'scripts/run_ddp.sh', and number of GPUs
        * run './scripts/run_ddp.sh'


3. Copy the *run_pretraining.py* script in the BERT working folder (it has some minor fixes):
    
    cp run_pretraining.py DeepLearningExamples/PyTorch/LanguageModeling/BERT/

4. Change the paths and number of GPUs per node in the *run_varuna_ddp.sh* script


5. Train BERT by running *./run_varuna_ddp.sh*


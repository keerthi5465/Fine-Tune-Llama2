# Fine-Tuning Llama 2 on Google Colab Using PEFT and QLoRA

This guide walks through fine-tuning the Llama 2 model using parameter-efficient fine-tuning (PEFT) techniques, specifically **LoRA** and **QLoRA**, on a Google Colab instance. Due to limited VRAM in free Colab sessions, we’ll use 4-bit precision for training to make the process feasible.

## Requirements

Ensure the following libraries are installed:

```
pip install peft transformers accelerate bitsandbytes
```
Since Google Colab offers a 15GB GPU (barely enough to store Llama 2–7B’s weights), using parameter-efficient fine-tuning (PEFT) techniques like LoRA or QLoRA is essential to reduce the computational cost.

##Model and Dataset
We'll be using the OpenAssistant Guanaco dataset for training. You can access the datasets from the following links:
- [OpenAssistant-Guanaco Dataset](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Ftimdettmers%2Fopenassistant-guanaco)
- [Reformatted Guanaco Dataset](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fmlabonne%2Fguanaco-llama2)
  
##Why PEFT (LoRA/QLoRA)?
Full fine-tuning of the model is not feasible due to hardware constraints. PEFT methods like LoRA freeze most of the model weights and only train a subset of them, drastically reducing the VRAM usage. QLoRA further reduces the memory footprint by allowing fine-tuning at 4-bit precision.

##Training Parameters
The model will be trained for 25 logging steps, with a batch size of 4, using the following parameters for supervised fine-tuning:
```
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    logging_steps=25,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,  # Enable 16-bit precision training for better performance
)
```
###Steps for Fine-Tuning
####Prepare Dataset:
Download and load the dataset using datasets from Hugging Face.

####Set Up Model with PEFT:
Use PEFT to freeze most weights and train only specific layers.

####Use 4-bit Quantization:
QLoRA enables model training at 4-bit precision, significantly reducing the memory required to store and update the model’s parameters.

###Train:
Run training on the available GPU, keeping track of the logging steps and training loss.

###Custom Prompts
You can pass custom prompts to the fine-tuned model using the following format:
```
<s>[INST] <<SYS>> System prompt <<SYS>> User prompt [/INST] Model answer </s>
```
##Notes
####VRAM Considerations: Keep in mind the overhead due to optimizer states, gradients, and forward activations when running the training.
####Limited Colab Resources: Due to the limited GPU resources available in free Google Colab, consider optimizing the model as much as possible using lower precision techniques like 4-bit quantization.

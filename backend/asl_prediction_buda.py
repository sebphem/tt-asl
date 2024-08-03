import os
import torch
from torch import nn
import pybuda
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader


# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

def setup_image(image: Image):
    image.convert("RGB")
    processed_tensor = processor(images=image, return_tensors='pt')
    return processed_tensor['pixel_values']

def get_prediction_given_tensor(input_tensor):
    tt0.push_to_inputs((input_tensor,))
    # output = pybuda_module.run(input_tensor)  # executes compilation (if first time) + runtime
    output_q = pybuda.run_inference()
    output = output_q.get()
    output_tensor = output[0].value()
    pred = output_tensor.argmax(-1).item()
    return id_to_label[pred]

def get_all_file_abspaths(directory):
    return [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained("dyllanesl/ASL_Classifier")
    model = AutoModelForImageClassification.from_pretrained("dyllanesl/ASL_Classifier")

    tt0 = pybuda.TTDevice(
        name="tt_device_0",  # here we can give our device any name we wish, for tracking purposes
        arch=pybuda.BackendDevice.Grayskull
    )


    # Create module
    pybuda_module = pybuda.PyTorchModule(
        name = "asl_model",  # give the module a name, this will be used for tracking purposes
        module=model  # specify the model that is being targeted for compilation
    )


    tt0.place_module(module=pybuda_module)


    label_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_to_id)}

    # Example usage:
    directory_path = '/home/user/tt-hackathon-2024/image_queue'
    all_file_paths = get_all_file_abspaths(directory_path)
    predicted_letters = []

    for file_path in all_file_paths:
        tmp_img = Image.open(file_path)
        readied_tensor = setup_image(tmp_img)
        guessed_label = get_prediction_given_tensor(readied_tensor)
        predicted_letters.append(guessed_label)

    print('predicted word: ', predicted_letters)
    tt0.remove_modules()
    # pybuda.shutdown()
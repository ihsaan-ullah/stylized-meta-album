import glob
import os
import pandas as pd
import numpy as np
import time
from task import category, domain

def run(output_dir, input_dir):
    input_dir = f'{input_dir}'
    print(input_dir)
    start_time = time.time()
    print("Starting experiment")
    current_path = os.getcwd()
    print(current_path)
    print(os.listdir(current_path))
    print("output dir : ", output_dir)

    import sys
    sys.path.append(f"{input_dir}/Pytorch_AdaIN_master")
    os.chdir(f"{input_dir}/Pytorch_AdaIN_master")

    #!cp ./vgg19-dcbb9e9d.pth /root/.cache/torch/hub/checkpoints/

    task_dir = f"{input_dir}/{domain}"
    print(f"Starting {task_dir.split('/')[-1]}")

    ##############################################################################################################################
    ##############################################################################################################################
    ############################################## Load Model ######################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    import argparse
    from PIL import Image
    import torch
    from torchvision import transforms
    from torchvision.utils import save_image
    from model import Model

    import warnings
    warnings.filterwarnings("ignore")


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([transforms.ToTensor(),
                                normalize])


    def denorm(tensor, device):
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
        res = torch.clamp(tensor * std + mean, 0, 1)
        return res

    model_state_path = "model_state.pth"
    gpu = 0
    alpha = 1

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f'cuda:0')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if model_state_path is not None:
        model.load_state_dict(torch.load(model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    def stylized(content,style_image,outputs_path):
        try:
            c_img = Image.open(content)
            s = Image.open(style_image)
            c_tensor = trans(c_img).unsqueeze(0).to(device)
            s_tensor = trans(s).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.generate(c_tensor, s_tensor, alpha)
            out_denorm = denorm(out, device)
            save_image(out_denorm, outputs_path, nrow=1)
        except RuntimeError:
                print('Images are too large to transfer. Size under 1000 are recommended ')
                print(style_image)

    df = pd.read_csv(f"{task_dir}/labels.csv")
    #df = df[df.CATEGORY == category].reset_index()
    #print(category)

    # Create output directory
    output_stylized_dir = f"{output_dir}/{domain}/stylized"

    cmpt = 0
    for i, row in df.iterrows():
        pair_name = f"{row.CATEGORY}_{row.STYLE}"
        os.makedirs(f"{output_stylized_dir}/{pair_name}", exist_ok=True)
        location_style_image = f"{task_dir}/styles/{row.STYLE_PATH}"
        location_content_image = f"{task_dir}/content/{row.FILE_PATH}"
        location_output_image = f"{output_stylized_dir}/{pair_name}/{row.FILE_NAME}"

        out = stylized(location_content_image, location_style_image, location_output_image)
        if i%100 == 0:
        	print(f"Execution time: {time.time() - start_time}")

    print(f"Finished {task_dir.split('/')[-1]}")
    os.chdir(current_path)
    print("Finished experiment")

    with open(f"{output_dir}/time.txt", "w") as f: f.write(f"{time.time() - start_time}")

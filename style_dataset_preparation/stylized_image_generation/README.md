# Stylize Image Generation code

To run the generate code, please follow the steps below:

1. Put your dataset on "input_data/datasets" (there is a README.md file there)
2. Replace the "ROOT_DIR" variable  by the absolute root path of the project in "ingestion.py". The root folder being the one containing "competition.yaml", 'input_data" and so on.
3. Download [VGG19 weights](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) and put them in `style_dataset_preparation/stylized_image_generation/input_data/Pytorch_AdaIN_master/`
4. Then put yourself on the "ingestion_program" file and execute "ingestion.py".
5. The generated data will be stored in "sample_result_submission".

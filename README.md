## Higgs Binary Classification
This project implements a neural network for binary classification of Higgs boson events using PyTorch. The dataset is derived from the Higgs Boson Machine Learning Challenge on Kaggle.
## Dataset
The dataset consists of features extracted from simulated particle collisions, with the goal of classifying events as either signal (Higgs boson) or background (non-Higgs events). The dataset is available on Kaggle: [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson).
## Goal
The goal of this project is to build a neural network that can accurately classify Higgs boson events from background events. The model will be trained on a subset of the dataset and evaluated on a separate test set.
## Requirements
Install the required packages using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate HiggsBinaryClassification
```
## Usage
To run the training script, use the following command:
```bash
python train.py
```
This will train the neural network on the training dataset and evaluate its performance on the validation set.
## Results
The model's performance will be logged using TensorBoard. You can visualize the training process by running the following command:
```bash
tensorboard --logdir=runs
```
Open your web browser and navigate to `http://localhost:6006` to view the TensorBoard dashboard.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Acknowledgments
This project is inspired by the Higgs Boson Machine Learning Challenge on Kaggle. This project is solely for educational purposes and is not affiliated with Kaggle or the organizers of the challenge.
## Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.
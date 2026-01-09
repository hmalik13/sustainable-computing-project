## ResNet-18 Training Script

This script trains a ResNet-18 convolutional neural network on the CIFAR-10 dataset using PyTorch while measuring the environmental impact of the training process.

### Overview

* Trains a ResNet-18 image classification model for a fixed workload (100 epochs, batch size 64).
* Uses CodeCarbon to track energy consumption and estimate carbon emissions during training.
* Collects metadata about the run, including cloud provider, region, instance type, GPU model, runtime, and final training loss.
* Saves:

  * Carbon emissions data to a CSV file
  * Experiment metadata to a JSON file

### Role in Sustainable Computing Final Project

This script serves as the controlled AI workload used to compare the sustainability of different cloud providers (AWS, Azure, and GCP) and regions. By running the same training job on identical hardware across multiple cloud environments, the results can be used to analyze differences in energy use, carbon emissions, and water footprint.

### Usage

Run the script from the command line with cloud-specific metadata:

```bash
python train.py --provider AWS --region us-east-1 --instance g4dn.xlarge
```

This ensures each experiment is labeled consistently for cross-provider and cross-region comparisons.

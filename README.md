# Vision 'n Language Testbed

Welcome to the Vision and Language Testbed repository. This is a playground and evaluation framework for various vision and language models, focused on controllable, synthetic datasets for more rigorous academic analysis. Dive into our curated test environment and explore the frontiers of cross-modal AI capabilities.

## Key Features 🌟
- **Unified Interface**: Every model adopts a unified structure for easy evaluation.
- **Rich Datasets**: Featuring "multi-MNIST" – a dataset where images consist of multiple MNIST digits with a corresponding descriptive text label.
- **Expandable**: Seamlessly add more models for evaluation.

## Getting Started 🚀

### Setup
1. Clone this repository:
```bash
git clone https://github.com/your-github-handle/vision-language-testbed
cd vision-language-testbed
```

2. Install necessary dependencies (TODO, once dependencies are finalized):
```bash
pip install -r requirements.txt
```

### Add New Vision+Language Models

* Ensure your model is saved under the `models` folder.
* Your model should inherit the `VisionLanguageModel` class located in `models/parent.py`.
* Use `models/clip.py` as a reference for implementing your model.


## Datasets 📦

**multi-MNIST**:
A novel dataset where each image comprises multiple MNIST digits with an associated text label. Labels describe spatial and visual attributes of the digits, such as "a 2 to the left of a red 4". This allows for controllability to explore various properties of vision+language models.


## Evaluating Models 📊
(TODO: evaluations we'll be doing on models)


## License 📄
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

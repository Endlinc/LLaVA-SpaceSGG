# LLaVA-SpaceSGG
[Paper](https://arxiv.org/abs/2412.06322) | [Dataset](https://huggingface.co/datasets/Endlinc/SpaceSGG) | [Benchmark](https://huggingface.co/datasets/Endlinc/SpaceSGG-Val) | Models

## Overview

LLaVA-SpaceSGG is a multimodal large language model (MLLM) designed to tackle the challenges of Scene Graph Generation (SGG) by improving spatial relation modeling and enabling open-vocabulary generalization. SGG converts visual scenes into structured graph representations, providing deeper scene understanding for complex vision tasks.

### Key Features
- **Enhanced Spatial Relation Modeling**: Incorporates object locations, relations, and depth information for better spatial reasoning.
- **Open-Vocabulary Generalization**: Excels in generating structured scene graphs in open-vocabulary contexts.
- **Custom Dataset: SpaceSGG**: A novel instruction-tuning dataset that includes spatial descriptions, question-answering (QA), and conversations.
- **Two-Stage Training Paradigm**: Improves model transferability to SGG tasks by leveraging MLLMs' native capabilities.

### Achievements
- **Performance**: LLaVA-SpaceSGG outperforms existing methods with a 4.5% improvement in recall and a 1.4% increase in mean recall.
- **Dataset**: SpaceSGG is constructed using a pipeline that integrates object locations, spatial relations, and depth information from public datasets and open-source models.

---

## Installation

Clone the repository and set up the environment:
```bash
git clone https://github.com/Endlinc/LLaVA-SpaceSGG.git
cd LLaVA-SpaceSGG
pip install -r requirements.txt
```

---

## Data Preparation

### Stage 1: Generate Point Clouds and Layered Objects
The scene graph description generation process in Stage 1 is built upon [the All-Seeing v2 project](https://github.com/OpenGVLab/all-seeing/tree/main/all-seeing-v2). Please refer to their repository for detailed instructions and implementation.
1. **Generate Point Cloud from RGB and Depth Image**:
   ```bash
   python d2p.py --dataset-path dataset/coco --scale-factor 5000 --world-coordinates
   ```
2. **Cluster Objects by Depth into Layers**:
   ```bash
   python layers_aggregation.py \
       --input-file asv2_level.json \
       --depth-dir ./depth-output \
       --mask-dir ./mask-output \
       --output-file processed_annotations.json \
       --dataset-base /home/ming/Datasets/all-seeing-v2/materials/ \
       --data-prefix ../data/
   ```
3. **Generate Multiview Layered Objects**:
   ```bash
   python multiview_layers.py \
       --input-file asv2_level.json \
       --point-cloud-dir ./point_clouds \
       --mask-dir ./mask-output \
       --output-file processed_annotations.json \
       --dataset-base /home/ming/Datasets/all-seeing-v2/materials/ \
       --data-prefix ../data/
   ```

### Stage 2: Generate Training Data Formats
1. **Generate Layered Descriptions**:
   ```bash
   python llm_based_query.py \
       --anno-file annotations.json \
       --prompt-function create_layer_prompt \
       --output-file layer_description.json
   ```
2. **Generate Question-Answering (QA) Data**:
   ```bash
   python llm_based_query.py \
       --anno-file annotations.json \
       --prompt-function create_between_prompt \
       --output-file between_qa.json
   ```
3. **Generate Conversation Data**:
   ```bash
   python llm_based_query.py \
       --anno-file annotations.json \
       --prompt-function create_rotation_prompt \
       --output-file rotation_prompts.json
   ```

---

## Usage

After preparing the dataset, train the LLaVA-SpaceSGG model using the scripts provide in project [LLaVA](https://github.com/haotian-liu/LLaVA) and [The All-Seeing Project V2](https://github.com/OpenGVLab/all-seeing/tree/main/all-seeing-v2)

---

## Citation

If you use LLaVA-SpaceSGG or SpaceSGG dataset in your research, please cite our work:
```bibtex
@inproceedings{llava_spacesgg2025,
  title={LLaVA-SpaceSGG: Visual Instruct Tuning for Open-vocabulary Scene Graph Generation with Enhanced Spatial Relations},
  author={Your Name and Co-authors},
  booktitle={Proceedings of WACV 2025},
  year={2025}
}
```

---

## License

This project is licensed under the [Apache License](LICENSE).

---

## Contact

For questions or feedback, please contact [parasolohalo@gmail.com](mailto:parasolohalo@gmail.com).

Let me know if you need adjustments!

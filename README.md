# Visual Question Answering on 3D Scenes

A capstone project that extends traditional Visual Question Answering (VQA) from single 2D RGB images to multi-view 3D scenes, aimed at improving perception accuracy for Human-Robot Interaction (HRI) applications.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Goals](#goals)
- [Approach](#approach)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Getting Started](#getting-started)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

## Introduction

Humans perceive the world using their eyes, ears, and other senses, and share their thoughts through spoken or written language. Getting machines to do something similar, understanding their surroundings and communicating insights about them, is a much harder problem. Computer Vision (CV) and Natural Language Processing (NLP) are the two fields pushing this forward, letting machines perceive an environment and respond to it in a more human way.

Visual Question Answering (VQA) is the task of a machine answering natural language questions about an image. It's been called an "AI-complete" task since it needs both CV and NLP working together. Most VQA systems today only look at a single 2D RGB image, which becomes a real limitation in practice. Poor lighting, occlusion, or just a bad camera angle can hide the exact detail a question is asking about.

Human-Robot Interaction (HRI) systems need something closer to how humans actually perceive their surroundings, using depth and multiple viewpoints instead of one flat image. This project extends VQA into a 3D setting by using four different viewpoints of a scene to build a fuller picture before answering questions about it.

We use the CLEVR dataset, a synthetic dataset of 3D-rendered scenes provided in JSON format along with generated questions and answers. Given hardware and execution-time constraints, we avoided datasets with direct point-cloud or 3D geometry, and instead generated multiple 2D renders per scene to approximate 3D understanding.

## Problem Statement

In standard VQA systems, a single fixed 2D image limits how much a model can actually understand about a scene. Occluded objects, ambiguous depth, and poor viewpoints all hurt accuracy. Using 4-viewpoint 3D scenes, this project tries to:

- Give a more complete and detailed representation of objects and how they relate spatially.
- Improve the model's ability to perceive a scene correctly and answer questions about it.

## Goals

- **Enhanced Perception**: use multi-view 3D images to surface object placement and relationships that a single view would hide.
- **Advanced VQA**: push VQA capability past the limits of single 2D images.
- **Robotics Applications**: lay groundwork for using these techniques in real-world human-robot interaction systems.

## Approach

1. **Dataset generation**: Extended the original CLEVR dataset generation pipeline (Blender) with a custom `static_scene_generator` module that renders 4 camera viewpoints per scene, spaced 90 degrees apart at 45 degrees elevation, instead of the original single view. Object attribute generation (color, size, material, shape) was also customized through `properties_customised.json`.
2. **Question generation**: Used the standard CLEVR `question_generation` pipeline, instantiating question templates (`CLEVR_1.0_templates`) against the generated scenes to produce question-answer pairs automatically.
3. **Image feature extraction**: Used a pretrained DenseNet121 CNN to extract features from each viewpoint image.
4. **Question processing**: Tokenized and lemmatized questions, then built hierarchical word, phrase, and sentence-level representations (embeddings plus a custom phrase-level module plus an LSTM for sentence-level).
5. **Joint feature representation**: Custom attention layers (`AttentionMaps`, `ContextVector`) fuse image and question features at each linguistic level, feeding forward hierarchically before a final dense classifier.
6. **Multi-view aggregation**: Each viewpoint is scored independently, and the 4 prediction vectors are stacked and combined with argmax to get the final answer.

## Dataset

The dataset used is CLEVR (Compositional Language and Elementary Visual Reasoning), a synthetic VQA benchmark of 3D-rendered scenes with ground-truth object attributes, relationships, and generated Q&A pairs.

- **Training set**: ~2,000 images, ~5,000 questions
- **Validation set**: ~400 images, ~4,000 questions
- **Test set**: ~400 images, ~4,000 questions

Each scene was re-rendered into 4 distinct viewpoints using a custom multi-view extension (`static_scene_generator/render_images_mv.py`) built on top of the original CLEVR dataset generator.

## Repository Structure

```
Visual-Question-Answering-On-3D-Scenes/
├── docs/
│   └── VQA_Research_Paper.pdf        # Research Paper
├── clevr-dataset-gen-main/
│   ├── assets/                       # Demo GIFs/images from the base CLEVR repo
│   │
│   ├── image_generation/             # Original single-view CLEVR image generation (Blender)
│   │   ├── data/
│   │   │   ├── materials/            # MyMetal.blend, Rubber.blend
│   │   │   ├── shapes/              # SmoothCube_v2, SmoothCylinder, Sphere
│   │   │   ├── base_scene.blend
│   │   │   ├── CoGenT_A.json / CoGenT_B.json
│   │   │   └── properties.json
│   │   ├── collect_scenes.py
│   │   ├── render_images.py
│   │   ├── utils.py
│   │   └── README.md
│   │
│   ├── question_generation/          # CLEVR question/answer generation
│   │   ├── CLEVR_1.0_templates/      # Question templates (one_hop, two_hop, comparison, etc.)
│   │   ├── generate_questions.py
│   │   ├── question_engine.py
│   │   ├── metadata.json
│   │   ├── synonyms.json
│   │   └── README.md
│   │
│   ├── static_scene_generator/       # Custom multi-view extension (core contribution)
│   │   ├── data/
│   │   │   ├── materials/
│   │   │   ├── shapes/               # Extended shape set: Diamond, Dolphin, Duck, Horse,
│   │   │   │                         # Monkey, Mug, SmoothCube_v2, SmoothCylinder, Sphere, Teapot
│   │   │   ├── base_scene_mv.blend   # Multi-view base scene
│   │   │   ├── base_scene_ori.blend
│   │   │   ├── base_scene.blend
│   │   │   ├── CoGenT_A.json / CoGenT_B.json
│   │   │   ├── properties.json
│   │   │   └── properties_customised.json   # Customized object attributes
│   │   ├── output/
│   │   │   ├── images/               # Rendered viewpoints: CLEVR_new_XXXXXX_00.png ... _03.png
│   │   │   └── scenes/               # Scene metadata: CLEVR_new_XXXXXX.json
│   │   ├── collect_scenes.py
│   │   ├── get_b_box.py
│   │   ├── make_mp4.py
│   │   ├── postproc_masks.py
│   │   ├── render_images_mv.py       # Multi-view rendering script
│   │   ├── run_gen.sh                # Batch scene generation entry point
│   │   └── utils.py
│   │
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── PATENTS
│   └── README.md
│
├── VQA_3_Model_Final.ipynb           # Final VQA model: training, evaluation, inference
├── test/
└── README.md
```

- To generate multi-view 3D scenes, run `clevr-dataset-gen-main/static_scene_generator/run_gen.sh` (see that folder for details).
- To generate the corresponding questions and answers, use `clevr-dataset-gen-main/question_generation/generate_questions.py` against the scene JSON files produced above.
- `VQA_3_Model_Final.ipynb` contains the full model pipeline: image feature extraction, question processing, attention-based fusion, and evaluation.

## Results

The model was evaluated using F1 score and categorical cross-entropy loss.

| Metric   | Training | Validation |
|----------|----------|------------|
| Accuracy | ~56%     | ~38%       |

The gap between training and validation accuracy is mostly a dataset size issue. Standard VQA benchmarks usually have far more images and questions than we could generate given hardware and time constraints.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nikhil20012/Visual-Question-Answering-On-3D-Scenes.git
   cd Visual-Question-Answering-On-3D-Scenes
   ```
2. **Install dependencies**: Blender for scene rendering, plus TensorFlow/Keras, NumPy, and the other packages referenced in `VQA_3_Model_Final.ipynb`.
3. **Generate the multi-view 3D dataset**
   ```bash
   cd clevr-dataset-gen-main/static_scene_generator
   bash run_gen.sh
   ```
4. **Generate questions and answers**
   ```bash
   cd ../question_generation
   python generate_questions.py --input_scene_file <path_to_scene_json> --output_questions_file <output_path>
   ```
5. **Train and evaluate the model**: open and run `VQA_3_Model_Final.ipynb`.

## Technologies Used

- **Machine Learning**: TensorFlow / Keras, CNN (DenseNet121), LSTM/GRU, custom attention mechanisms
- **NLP**: Tokenization, lemmatization, hierarchical word/phrase/sentence embeddings
- **3D Image Processing / Rendering**: Blender, custom multi-view CLEVR dataset generation pipeline

## Future Work

Multi-view 3D VQA is still a pretty underexplored area compared to standard 2D VQA, so there's a lot of room to keep building on this. Next steps would be improving accuracy with a larger generated dataset and trying out different multi-view fusion architectures, with the eventual goal of getting this to a point where it could actually be deployed in real HRI applications.

---

**Authors:** Y Nikhil Bharadwaj, Shubham M Mahale, Altaf Abdul Razak Kandagal, Yamajala Siddhardha
**Guidance:** Dr. Surabhi Narayan, PES University

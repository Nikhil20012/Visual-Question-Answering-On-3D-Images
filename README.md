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
3. **Image feature extraction**: Used a pretrained DenseNet121 CNN to extract (49, 1024)-dimensional feature maps from each viewpoint image. The 49 comes from the 7x7 spatial grid of DenseNet121's final conv layer, and 1024 is the channel depth.
4. **Question processing**: Tokenized and lemmatized questions, then built hierarchical word, phrase, and sentence-level representations (embeddings plus a custom phrase-level module plus an LSTM for sentence-level).
5. **Joint feature representation**: Custom attention layers (`AttentionMaps`, `ContextVector`) fuse image and question features at each linguistic level, feeding forward hierarchically before a final dense classifier.
6. **Multi-view aggregation**: Each viewpoint is scored independently, and the 4 prediction vectors are stacked and combined with argmax to get the final answer.

## Dataset

The dataset used is CLEVR (Compositional Language and Elementary Visual Reasoning), a synthetic VQA benchmark of 3D-rendered scenes with ground-truth object attributes, relationships, and generated Q&A pairs.

- **Total**: ~2,000 images, ~5,000 questions (21 distinct answer types)
- **Train/Validation split**: 99/1 stratified split (4,950 train, 50 validation samples)

The split is small because the dataset itself is small. Standard VQA benchmarks have orders of magnitude more data; we were constrained by hardware and Blender rendering time.

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

During training, the model is monitored using F1 score (micro-averaged) and categorical cross-entropy loss. Final accuracy is measured using a custom VQA accuracy function based on the standard VQA evaluation protocol: `min(count of matching human answers / 3, 1)` per question, averaged across the dataset.

| Metric       | Training | Validation |
|--------------|----------|------------|
| VQA Accuracy | ~56.2%   | ~46.7%     |

The gap between training and validation accuracy is largely a dataset size issue. The validation set contains only 50 samples (due to the 99/1 split), making it a noisy estimate. Standard VQA benchmarks have far more images and questions than we could generate given hardware and rendering time constraints.

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

- **Machine Learning**: TensorFlow / Keras, CNN (DenseNet121), LSTM, custom attention mechanisms (co-attention)
- **NLP**: Tokenization, lemmatization, hierarchical word/phrase/sentence embeddings
- **3D Image Processing / Rendering**: Blender, custom multi-view CLEVR dataset generation pipeline

## Future Work

This project was completed in 2022, before the wave of large vision-language models
(GPT-4V, Claude, etc.) that have since transformed multimodal reasoning. The 3D VQA
space has progressed considerably since then, with benchmarks like ScanQA and MV-ScanQA,
and fusion approaches that bridge 2D and 3D representations using LLM backbones. If
revisiting this project today, the natural next step would be replacing the
DenseNet121 + co-attention pipeline with a VLM-based architecture, scaling up the
dataset with more rendered scenes, and evaluating on established 3D VQA benchmarks
rather than a custom CLEVR subset.

Some production-oriented directions that could build on this work:

- **Warehouse inventory audit**: mount 4 fixed cameras at each storage bay, run
  multi-view VQA to answer queries like "is the top shelf fully stocked?" or "are
  any items misplaced?" without manual walkthroughs. Plug into an existing WMS
  (warehouse management system) via a FastAPI endpoint that accepts a bay ID and a
  natural language question, returns an answer plus the viewpoint that was most
  informative.

- **Retail shelf compliance**: similar multi-camera setup in store aisles. Brand
  managers could ask "is Product X at eye level?" or "how many facings does Product Y
  have?" through a simple dashboard. The multi-view approach directly solves the
  occlusion problem that single-camera planogram systems struggle with.

- **Industrial quality inspection**: position cameras around an assembly line station
  to catch defects that are only visible from certain angles. A technician could ask
  "is the weld on the left joint complete?" and get an answer grounded in whichever
  viewpoint best shows that joint, along with a confidence score.

- **Assistive robotics for accessibility**: a mobile robot with multiple cameras could
  help visually impaired users navigate indoor spaces by answering spatial questions
  like "is there a chair blocking the hallway?" or "which door is the elevator?"
  using real-time multi-view fusion.

Each of these could be prototyped today using a VLM API (Claude or GPT-4V) for the
reasoning layer, with the multi-view rendering and aggregation pipeline from this
project adapted to handle real camera feeds instead of Blender-generated scenes.

---

**Authors:** Y Nikhil Bharadwaj, Shubham M Mahale, Altaf Abdul Razak Kandagal, Yamajala Siddhardha  
**Guidance:** Dr. Surabhi Narayan, PES University

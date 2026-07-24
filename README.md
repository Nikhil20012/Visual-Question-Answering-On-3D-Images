# Visual Question Answering on 3D Images

> A capstone project extending traditional Visual Question Answering (VQA) from single 2D RGB images to multi-view 3D scenes, aimed at improving perception accuracy for Human-Robot Interaction (HRI) applications.

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

It's quite natural for humans to perceive the world using their eyes, ears, and other senses, and to share their thoughts through spoken or written language. But how can machines perform such complex tasks — understanding their surroundings and communicating insights about them? Computer Vision (CV) and Natural Language Processing (NLP) are key to advancing this capability, enabling machines to perceive environments and respond in a human-like manner.

**Visual Question Answering (VQA)** — where a machine answers natural language questions about an image — has drawn significant attention from the AI community as an "AI-complete" task, since it integrates both CV and NLP. Most VQA systems today work with a single 2D RGB image, but this becomes a limitation in real-world settings: poor lighting, occlusion, or an unfavorable viewpoint can hide critical information.

Human-Robot Interaction (HRI) systems, by contrast, need to perceive and recognize their surroundings the way humans do — using depth and multiple viewpoints, not a single flat image. This project extends traditional VQA into a **3D setting**, using **four different viewpoints of a scene** to build a richer, more complete representation before answering questions about it.

We use the **CLEVR dataset**, a synthetic dataset of 3D-rendered scenes provided in JSON format along with generated questions and answers. Due to hardware and execution-time constraints, we avoided datasets with direct point-cloud/3D geometry, and instead generated multiple 2D renders per scene to approximate 3D understanding.

## Problem Statement

In standard VQA systems, a single fixed 2D image limits how much an agent can understand about a scene — occluded objects, ambiguous depth, and poor viewpoints all hurt accuracy. By using **4-viewpoint 3D scenes**, this project aims to:

- Provide a more complete and detailed representation of objects and their spatial relationships.
- Improve the model's ability to perceive and correctly answer questions about a scene.

## Goals

- **Enhanced Perception** — use multi-view 3D images to reveal object placement and relationships that a single view would hide.
- **Advanced VQA** — push VQA capability beyond the limitations of single 2D images.
- **Robotics Applications** — lay groundwork for using these techniques in real-world human-robot interaction systems.

## Approach

1. **Dataset generation** — Extended the original CLEVR dataset generation pipeline (Blender) with a custom `static_scene_generator` module that renders **4 camera viewpoints per scene** (spaced 90° apart, at 45° elevation), instead of the original single view. Object attribute generation (color, size, material, shape) was also customized via `properties_customised.json`.
2. **Question generation** — Used the standard CLEVR `question_generation` pipeline, instantiating question templates (`CLEVR_1.0_templates`) against the generated scenes to auto-produce question-answer pairs.
3. **Image feature extraction** — Used a pretrained **DenseNet121** CNN to extract features from each viewpoint image.
4. **Question processing** — Tokenized and lemmatized questions, then built hierarchical word/phrase/sentence-level representations (embeddings + a custom phrase-level module + LSTM for sentence-level).
5. **Joint feature representation** — Custom attention layers (`AttentionMaps`, `ContextVector`) fuse image and question features at each linguistic level, feeding forward hierarchically before a final dense classifier.
6. **Multi-view aggregation** — Each viewpoint is scored independently, and the 4 prediction vectors are stacked and combined via argmax to produce the final answer.

## Dataset

The dataset used is **CLEVR** (Compositional Language and Elementary Visual Reasoning) — a synthetic VQA benchmark of 3D-rendered scenes with ground-truth object attributes, relationships, and generated Q&A pairs.

- **Training set:** ~2,000 images, ~5,000 questions
- **Validation set:** ~400 images, ~4,000 questions
- **Test set:** ~400 images, ~4,000 questions

Each scene was re-rendered into **4 distinct viewpoints** using a custom multi-view extension (`static_scene_generator/render_images_mv.py`) built on top of the original CLEVR dataset generator.

## Repository Structure

```
Visual-Question-Answering-On-3D-Images/
├── clevr-dataset-gen-main/
│   ├── assets/                       # Demo GIFs/images from the base CLEVR repo
│   │
│   ├── image_generation/             # Original single-view CLEVR image generation (Blender)
│   │   ├── data/
│   │   │   ├── materials/            # MyMetal.blend, Rubber.blend
│   │   │   ├── shapes/                # SmoothCube_v2, SmoothCylinder, Sphere
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
│   ├── static_scene_generator/       # ★ Custom multi-view extension (core contribution)
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
│   │   ├── render_images_mv.py       # ★ Multi-view rendering script
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
- `VQA_3_Model_Final.ipynb` contains the full model pipeline — image feature extraction, question processing, attention-based fusion, and evaluation.

## Results

The model was evaluated using F1 score and categorical cross-entropy loss.

| Metric   | Training | Validation |
|----------|----------|------------|
| Accuracy | ~56%     | ~38%       |

The gap between training and validation accuracy is largely attributed to dataset size — standard VQA benchmarks typically use far more images and questions than were feasible to generate here given hardware/time constraints.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nikhil20012/Visual-Question-Answering-On-3D-Images.git
   cd Visual-Question-Answering-On-3D-Images
   ```
2. **Install dependencies** — Blender (for scene rendering), plus TensorFlow/Keras, NumPy, and other packages referenced in `VQA_3_Model_Final.ipynb`.
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
5. **Train and evaluate the model** — open and run `VQA_3_Model_Final.ipynb`.

## Technologies Used

- **Machine Learning:** TensorFlow / Keras, CNN (DenseNet121), LSTM/GRU, custom attention mechanisms
- **NLP:** Tokenization, lemmatization, hierarchical word/phrase/sentence embeddings
- **3D Image Processing / Rendering:** Blender, custom multi-view CLEVR dataset generation pipeline

## Future Work

This project remains an active area of research, as multi-view 3D VQA is still relatively underexplored compared to standard 2D VQA. Planned directions include improving accuracy through larger generated datasets and exploring alternative multi-view fusion architectures, with the eventual goal of deployment in real-world HRI applications.

---

**Authors:** Y Nikhil Bharadwaj, Shubham M Mahale, Altaf Abdul Razak Kandagal, Yamajala Siddhardha
**Guidance:** Dr. Surabhi Narayan, PES University

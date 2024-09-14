# Visual Question Answering on 3D Images

## Introduction

It's quite natural for humans to perceive the world around them using their eyes, ears, and other senses. Humans share their thoughts with others through spoken or written language. But how can machines perform such complex tasks, like understanding their surroundings and sharing their insights? The fields of Computer Vision (CV) and Natural Language Processing (NLP) are key to advancing this capability, allowing machines to perceive environments and communicate in a human-like manner.

"Visual Question Answering" (VQA) involves machines answering questions about images, and it has garnered significant attention from the AI community as an "AI-complete" task. This task integrates both CV and NLP techniques, with potential applications in intelligent robots to simplify daily human tasks. Most current VQA systems focus on answering questions about single RGB (2D) images. However, in real-world applications, 2D images can be insufficient, especially under poor lighting or inappropriate viewpoints.

Human-Robot Interaction (HRI) systems rely on 3D information to perceive and recognize their surroundings. Thus, it's crucial for robots to understand and interpret their environment similarly to how humans do. This project aims to advance traditional VQA by applying it to 3D images. By utilizing multiple viewpoints to create a comprehensive 3D view of the scene, we seek to enhance the accuracy of question answering.

The dataset used in this project is the CLEVR dataset, which consists of 3D-rendered images in JSON format, along with various questions and answers about the scenes. Due to hardware limitations and execution time considerations, we avoided using datasets with direct 3D information.

## Project Overview

This project focuses on enhancing Visual Question Answering (VQA) by incorporating 3D images. Unlike traditional VQA systems that work with 2D images, our approach uses multiple views of an environment to provide a richer understanding of the scene.

## Problem Statement

In standard VQA systems, a fixed 2D image often limits the agent's understanding of an environment. By using 3D images—where each scene consists of four different views—we aim to:

- Provide a more complete and detailed representation of objects and their positions.
- Improve the agent's ability to perceive and answer questions about the environment accurately.

## Goals

- **Enhanced Perception**: Use 3D images to reveal more information about object placements and relationships.
- **Advanced VQA**: Improve VQA capabilities by moving beyond the limitations of 2D images.
- **Robotics Applications**: Apply these advancements to improve human-robot interactions and real-world robotic systems.

## Getting Started

To get started with this project, you'll need to:

1. Clone the repository.
2. Install the required dependencies.
3. Prepare your 3D image dataset.
4. Train and evaluate the VQA model.

## Technologies Used

- Machine Learning (ML)
- Natural Language Processing (NLP)
- 3D Image Processing

For more details on setup and usage, please refer to the instructions in the repository.

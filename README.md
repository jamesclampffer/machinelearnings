# machinelearnings

Welcome to machinelearnings – a repository documenting my ongoing ventures into understanding the nuts and bolts of machine learning. This is a collection of tangible projects used for fun (and maybe profit).

Projects here are inherently works-in-progress, with newer iterations often building directly on the lessons (and occasional frustrations) of older code. While some of these efforts trace their origins back to late 2024, the commit history might not always tell the full story. This repository is a consolidation of several disparate efforts, so expect a timeline that’s more about conceptual progression than strict chronological commits.

## Convolutional Neural Network Exploration

I knew a few details about CNNs prior to starting this:
- It's a bunch of linear algebra.
- They use convolutions in a sliding window as a means of feature extraction.
  - Prior use of Fourier and Laplace transforms in other domains helped out here.

Project goals:
- Learn end-to-end architecture and training.
- Get to a point where I can make some models for domain-specific use cases that have business value.


### [Naivenet](cifar-example/README.md)

This project (and its earlier iterations) served as my ground-up introduction to CNNs. Starting with surprisingly simple architectures, it progresses through fundamental concepts like residual blocks and data augmentation. A key early learning involved the significant performance gains observed by simply upscaling low-resolution CIFAR images (e.g., 32x32 to 128x128) before applying affine transformations, giving augmentation plenty of "room to work" without losing critical information.

### [MiniNet](mininet/README.md)

Building directly on insights gained from NaiveNet and some reading, this lays out a more optimized and modular CNN framework. The objective here was to construct a highly efficient, ResNet-inspired architecture that prioritizes parameter reduction and rapid prototyping. It incorporates architectural patterns like depthwise-separable convolutions and Squeeze-and-Excitation (SE) blocks, aiming for high performance even on constrained hardware (e.g. robotics on a power budget). This is designed to be a useful toolkit for future CNN experimentation.

## Large Language Model (LLM) Explorations

Beyond computer vision, it's hard to ignore the LLM hype. I used to be dismissive of it; I still am dismissive of it, but I used to too. The whole notion of shoehorning symbolic logic into a language model trained on the wisdom of the internet is a dead end and overlooks a lot of real utility, IMHO. That said, I understand there is also real value in having non-experts bang out scripts to make their workflow easier.

LLMs are a uniquely marketable piece of advanced R&D. And I need to inform myself how much effort I should put into solving utilitarian problems using them — it seems like utility is inversely proportional to amount of capital one can raise these days. The other option would be building a good marketing team, spending minimal effort fine-tuning an existing model, and then make outlandish claims.

### [Twelve Angry LLMs](lang/README.md)

~~A set of chatbots~~ A distributed team of cutting-edge models powering agentic AIs to walk through a silly, yet surprisingly insightful, simulation of "Twelve Angry Men". This project is a hands-on dive into the practicalities of working with LLMs.

It's a direct experiment designed to see how fast I could get something running with no prior knowledge of tooling (a few days) that provides a window to observe these models. Also, it's a tongue-in-cheek exploration of "Prompt Engineering"(TM).

### [Federated Similarity Search](inverted_similarity_index/README.md)

An initial dive into building an on-premise semantic similarity search engine. Leveraging text embeddings and Approximate Nearest Neighbor (ANN) indexing, it aims to quickly find documents by their meaning rather than exact keywords. While serving as a learning vehicle, it explores potential end uses such as localized RAG (Retrieval Augmented Generation) for LLMs and robust document search, laying groundwork for these applications. The focus is significantly on practical, platform-level concerns: infrastructure control, scalability, and federated search across multiple data shards, rather than relying on external hosted services.
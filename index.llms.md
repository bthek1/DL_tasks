# DL Tasks

One page per machine learning task: what it is for, which models currently lead, how they are measured, and code that runs on a single consumer GPU.

Author

Benedict Thekkel

This site is organised by **task** rather than by architecture. The question it answers is “I have this input and want that output, what should I use”, which is how the problem usually arrives. The categories follow the Hugging Face task taxonomy, so a page here maps directly onto a filterable set of models on the Hub.

Every page follows the same structure: real-world use cases, recent research, the models currently worth using, the metrics that evaluate them, runnable code, and a benchmark comparison at the end. Where hardware allows, the page finishes with a live demo against the webcam or microphone.

The code is written for a small machine, an RTX 3060 with 12 GB of VRAM alongside 20 GB of system RAM, so models are loaded in half precision, one at a time, and freed afterwards. Anything that will not fit is discussed in prose rather than run.

For the architectures underneath these tasks see [DL Methods](https://bthek1.github.io/DL_methods/).

------------------------------------------------------------------------

## Natural Language Processing

| Page | Task |
|----|----|
| [Text Classification](Natural_Language_Processing/text_classification.llms.md) | Assigning a label to a document, which covers sentiment and intent |
| [Token Classification](Natural_Language_Processing/token_classification.llms.md) | Labelling each token, as in named entity recognition |
| [Question Answering](Natural_Language_Processing/question_answering.llms.md) | Extracting an answer from a passage |
| [Table Question Answering](Natural_Language_Processing/table_question_answering.llms.md) | The same against structured tables |
| [Zero-Shot Classification](Natural_Language_Processing/zero_shot_classification.llms.md) | Classifying into labels the model was never trained on |
| [Translation](Natural_Language_Processing/translation.llms.md) | Between languages |
| [Summarization](Natural_Language_Processing/summarization.llms.md) | Long text to short |
| [Text Generation](Natural_Language_Processing/text_generation.llms.md) | The autoregressive case, which is what an LLM does |
| [Fill-Mask](Natural_Language_Processing/fill_mask.llms.md) | Predicting a hidden token, the objective BERT was trained on |
| [Feature Extraction](Natural_Language_Processing/feature_extraction.llms.md) | Turning text into vectors |
| [Sentence Similarity](Natural_Language_Processing/sentence_similarity.llms.md) | Comparing those vectors, which is what retrieval is built on |
| [Text Ranking](Natural_Language_Processing/text_ranking.llms.md) | Re-ordering candidates, the second stage of a RAG pipeline |

------------------------------------------------------------------------

## Computer Vision

The largest section, split by what goes in and what comes out.

**Understanding an image**

| Page | Task |
|----|----|
| [Image Classification](Computer_Vision/image_classification.llms.md) | One label for the whole image |
| [Object Detection](Computer_Vision/object_detection.llms.md) | Boxes around things, with labels |
| [Image Segmentation](Computer_Vision/image_segmentation.llms.md) | Labels per pixel |
| [Mask Generation](Computer_Vision/mask_generation.llms.md) | Segmenting without being told what to look for |
| [Keypoint Detection](Computer_Vision/keypoint_detection.llms.md) | Locating specific points, as in pose estimation |
| [Depth Estimation](Computer_Vision/depth_estimation.llms.md) | Recovering distance from a single image |
| [Image Feature Extraction](Computer_Vision/image_feature_extraction.llms.md) | Turning an image into a vector |
| [Zero-Shot Image Classification](Computer_Vision/zero_shot_image_classification.llms.md) | Classifying against labels described in words, which is what CLIP enables |
| [Zero-Shot Object Detection](Computer_Vision/zero_shot_object_detection.llms.md) | The same for boxes |
| [Image-to-Text](Computer_Vision/image_to_text.llms.md) | Captioning |

**Generating an image or video**

| Page | Task |
|----|----|
| [Text-to-Image](Computer_Vision/text_to_image.llms.md) | The diffusion case everyone knows |
| [Image-to-Image](Computer_Vision/image_to_image.llms.md) | Editing, style transfer, and super-resolution |
| [Unconditional Image Generation](Computer_Vision/unconditional_image_generation.llms.md) | Sampling with no prompt at all |
| [Text-to-Video](Computer_Vision/text_to_video.llms.md) | Video from a prompt |
| [Image-to-Video](Computer_Vision/image_to_video.llms.md) | Animating a still |
| [Video-to-Video](Computer_Vision/video_to_video.llms.md) | Transforming existing footage |
| [Text-to-3D](Computer_Vision/text_to_3d.llms.md) | Geometry from a prompt |
| [Image-to-3D](Computer_Vision/image_to_3d.llms.md) | Geometry from a photograph |
| [Video Classification](Computer_Vision/video_classification.llms.md) | Labelling a clip, which needs the time dimension |

------------------------------------------------------------------------

## Audio

| Page | Task |
|----|----|
| [Automatic Speech Recognition](Audio/automatic_speech_recognition.llms.md) | Speech to text. This is the reference page the others are modelled on |
| [Text-to-Speech](Audio/text_to_speech.llms.md) | The reverse |
| [Text-to-Audio](Audio/text_to_audio.llms.md) | Generating sound and music rather than speech |
| [Audio-to-Audio](Audio/audio_to_audio.llms.md) | Enhancement, separation, and voice conversion |
| [Audio Classification](Audio/audio_classification.llms.md) | Labelling a sound |
| [Voice Activity Detection](Audio/voice_activity_detection.llms.md) | Finding where speech is, which is the first stage of most audio pipelines |

------------------------------------------------------------------------

## Multimodal

Where more than one modality goes in, which is where most current research sits.

| Page | Task |
|----|----|
| [Image-Text-to-Text](Multimodal/image_text_to_text.llms.md) | A vision language model answering about an image |
| [Visual Question Answering](Multimodal/visual_question_answering.llms.md) | The narrower, older framing of the same |
| [Document Question Answering](Multimodal/document_question_answering.llms.md) | Questions against a scanned page, layout included |
| [Visual Document Retrieval](Multimodal/visual_document_retrieval.llms.md) | Finding the right page without transcribing it first |
| [Audio-Text-to-Text](Multimodal/audio_text_to_text.llms.md) | Speech in, text out, without a separate transcription step |
| [Video-Text-to-Text](Multimodal/video_text_to_text.llms.md) | The same over video |
| [Image-Text-to-Image](Multimodal/image_text_to_image.llms.md) | Instructed image editing |
| [Image-Text-to-Video](Multimodal/image_text_to_video.llms.md) | A still plus a prompt, animated |
| [Any-to-Any](Multimodal/any_to_any.llms.md) | Single models handling arbitrary input and output modalities |

------------------------------------------------------------------------

## Tabular, Reinforcement Learning, and Graphs

| Page | Task |
|----|----|
| [Tabular Classification](Tabular/tabular_classification.llms.md) | Deep learning against the data type where gradient boosting usually still wins |
| [Tabular Regression](Tabular/tabular_regression.llms.md) | The continuous version |
| [Time Series Forecasting](Tabular/time_series_forecasting.llms.md) | Foundation models for forecasting, complementing the classical treatment in [Time Series](https://bthek1.github.io/ML_Timeseries/) |
| [Reinforcement Learning](Reinforcement_Learning/reinforcement_learning.llms.md) | Learning from reward rather than labels |
| [Robotics](Reinforcement_Learning/robotics.llms.md) | Policies that act on physical hardware |
| [Graph Machine Learning](Other/graph_machine_learning.llms.md) | Learning over nodes and edges |

------------------------------------------------------------------------

## Frameworks and Theory

| Page | Covers |
|----|----|
| [Pipecat](Frameworks/pipecat.llms.md) | The framework for real-time voice and multimodal agents, which is what stitches the audio pages into a conversation |
| [Piper vs Kokoro](Frameworks/piper_vs_kokoro.llms.md) | Both small TTS engines run on plain ONNX Runtime, with side-by-side audio, hard-text examples, and measured speed, memory and round-trip WER |
| [Mathematical Foundations](Theory/mathematics_foundations.llms.md) | The symbols and notation the papers assume you already read |
| [Deep Learning Equations](Theory/deep_learning_equations.llms.md) | The equations behind the architectures |
| [Discrete Maths](Theory/discrete_maths.llms.md) | Sets, logic, and combinatorics |

The physics, electromagnetism and electricity reference sheets moved to [Electrical Docs](https://bthek1.github.io/Electrical_Docs/).

------------------------------------------------------------------------

## Not Covered Yet

- **The benchmark numbers are indicative, not rigorous.** Each page compares models on a small sample on one GPU, which is enough to choose between candidates but not to make a claim about a leaderboard.
- **Nothing is fine-tuned.** Every page uses pretrained checkpoints as they come, so there is no coverage of adapting a model to a specific dataset.
- **No deployment or serving.** The pages stop at a working model in a notebook. Nothing on inference servers, batching, quantisation for production, or latency budgets.
- **Large models are described but not run.** Anything past roughly 3B parameters in half precision, or an 8 GB download, is discussed in prose rather than executed, so the biggest models in each category are unmeasured here.
- **[ForResNet](Theory/forresnet.llms.md) is a placeholder** marked “coming soon”.
- The theory folder is a reference sheet rather than an explanation.

------------------------------------------------------------------------

Back to top

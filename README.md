# Towards Behavioral-Alignment via Representation Alignment

## Can Neural Encoding Strategies in Humans be Useful for AI Alignment?

Representational alignment is an emerging field at the intersection of neuroscience and artificial intelligence, focusing on the parallels between neural representations in the human brain and those in deep neural networks.
In recent years, researchers in this field have observed that as state-of-the-art AI models have become increasingly capable at completing the tasks they were trained for, their learned latent representations learned have become increasingly predictive of neural activity in the brain of primates.
This phenomenon has been observed in image models and language models, for instance, in which latent representations in these models has been observed to correlate with, and be predictive of, neural activity measured in brain areas of primates dedicated vision/language processing.
Ultimately, the goal of representational alignment research, however, is not solely to reveal fun insights like this (joking aside, these insights have actually been very valuable in developing better models of the brain and revealing insights in neuroscience), but also to understand the intricate relationships between these representations and system behavior.

With this in mind, I ask **can we make progress in behavioral alignment through representational alignment**?
That is, if we are to develop models that encode similar information to information encoded in human brains (or maybe more accurately, the areas of human brains associated with cognition and agency), can this contribute to behavior alignment?
In this project, I take a first step towards investigating this big question by studying a much more specific question: **can we encourage image models to learn more interpretable features by increasing their alignment with neural activity in the visual cortex (brain areas primariliy dedicated to visual processing) of humans?**

Below, I walk though the problem setup, execution, and analysis used in this project to study this question.
In this readme, I seek to keep the content concise, in effort to give you the gist of this work without eating into the time you may use to read the other 12 billion AI papers published today.
If, by doing this, there are some details I skipped over that you are curious about, please feel welcome to message me, yell your question into the void, send me a letter... you do you.

## Hypothesis

**A quick backround before the hypothesis**: The ventral stream is a processing pathway in the human visual cortex theorized to play a particularly important role in visual object recognition.  Neurons in higher-order cortical areas in this processing pathway are thought to encode information relevant to shape, texture, object parts, faces, and plenty more.  Although plenty of these neurons are thought to exhibit mixed selectivity (e.g., are polysemantic, responding to unrelated concepts), some suggest (such as in [here for example](https://www.jneurosci.org/content/43/10/1731)) that these neurons encode semantically meaningful information.

**Hypothesis**: By fine-tuning an image model to predict neural activity in high-order areas of the ventral stream, the image model will learn to encode features that tend to be more interpretable (semantically more meaningful).

## Methods, Short and Sweet
[(skip to the approach?)](#approach)

### Models used for analysis
In this preliminary work, I focus on investigating interpretability of simple CNN image models.
Investigated models include:
- ResNet-18
- More to come soon...

### Data
- CIFAR-10: A classic. 60000 32x32 colour images from 10 classes.
- [Natural Scenes Dataset](https://naturalscenesdataset.org/): large-scale fMRI dataset consisting of whole-brain fMRI measurements of 8 humans while viewing images from MS COCO.

### Evaluating Interpretability
Quantifying interpretability is an open challenge.  In this project, I use an automated metric, the [Interpretability Index](https://arxiv.org/pdf/2310.11431) (II) (David Klindt et al.) to quickly quantify how interpretable model features are.  In short, this metric quantifies how interpretable a neuron is based on similarities among the neuron's Maximally Exciting Images (i.e., the images that cause maximal activation for the neuron).  Multiple metrics can be used to evaluate MEI similarity, but here we focus on II-LPIPS, the pairwise learned perceptual image patch similarity ([LPIPS](https://github.com/richzhang/PerceptualSimilarity)) across MEIs, which was shown by Klindt et al. to correlated well with human measures of interpretability.

In this 

Is this metric perfect?  Probably not.  Is it insightful?  I think so.  The paper is pretty cool, check it out!

### Approach
1. Train a randomly initialized model on CIFAR-10.  We'll call this the "Baseline Model" (I know, how creative).
2. Make a copy of the baseline model, and fine tune it to predict neural activity from the Natural Scences Dataset (NSD).  In effort to maintain task-performance on original task (since NSD doesnt have CIFAR labels), this fine-tuning was done following the Learning Without Forgetting Knowledge Distillation apporach (distilling knowledge from the baseline model).  Let's call this new model the "Neural-Tuned Model".
3. Compare the two models based on original task performance and interpretability indices (IIs).

Some small details:
- NSD neural activity was predicted using a linear prediction from the image-model's final feature representation (deeper layers in image models tend to be more predictive of higher cortical areas than earlier layers).  More rigorous studies will also consider additional layers in the model
- The NSD dataset contains activity in voxels from all across the brain.  In this preliminary work, I focused on fine-tuning based on activity in visual area [V4](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7501212/) (technically a mid-level area).  Future work will study tuning with different brain areas and also multiple brain areas at once

## Results

| Model | Condition | CIFAR-10 Val Acc | II-LPIPS |
|:--|:--:|:--:|:--:|
| ResNet-18 | Baseline     | $0.9325$ | $0.354 \pm 0.050$ |
| ResNet-18 | Neural-Tuned | $0.8743$ | $0.322 \pm 0.040$ |

Notes:
- II submetrics reported as mean +/- std dev across all neurons from ResNet-18 Layer 4 (the neuron activations that directly predict NSD neural activity) of the network
- Difference in II-LPIPS is statistically significiant ($p < 0.01$, one-way anova)

## So What?

So thats pretty neat!  Quantitatively, according to II metrics, the neural tuned ResNet appears to be slightly more interpretable than the baseline model (future work on this repo will involve translating these II metrics into something a bit more meaningful).

Original task performance does drop quite a bit, but this may be something that could be managed with a larger tuning dataset (here, I am only using a subset of the NSD dataset), incorporating the original training data into the tuning process, and diligent training + hyperparameter selection.

I'm excited to apply this same technique to more images models (and maybe even on language models with similar techniques), and this will be important to understand if this approach generalizes (or if I just got lucky (unlucky hope?) with a ResNet-18 model trained on a toy task).

I think this is kind of cool though -- altering model characteristics (in a potentially favorable way in terms of AI safety (though of course, the tuned model is still not "read out the algorithm" interpretable)) by aligning its representations with the human brain.  Could more be in store in terms of Human-AI alignment via alignment of their representations?

## Running this code

Basline training image models on CIFAR-10: ```python /path/to/config.py``` (see ```/configs``` for example configuration files)

Fine-tuning with NSD data: Refactored code coming soon... (in the meantime, checkout tune_nsd.ipynb for a very rough notebook with tuning code)



## Whats next?
Stay tuned to this github repo for a lot more experiments to come.  Notably, probing beyond interpretability and deeper into representational alignment-based behavior tuning.

## A Parting Note
See something wrong? Want to contribute in making the next steps? Have some related work or ideas youd like to chat about or want a collaborator for? Want to chat with a new friend?

Reach out -- I'd love to hear your thoughts.
---
layout: post
title: "About Keeping Your Use Case in Focus"
categories: research
---

Today I have no fancy project and no shiny GitHub repository to show, because today I want to talk about my research.
As some may read in my bio, I am doing my PhD in the field of predictive maintenance (PDM).
This field is directly adjacent to machine learning and fell, like many others, into the grasp of the deep learning hype.
Unfortunately, long series of sensor readings are not as interesting to look at as images or intuitively understood as natural language, so PDM is not as present in the mind of the general ML crowd.

But, this post is not only about my research.
It is about the mindset a ML practitioner has to maintain when it comes to problems outside the commonly represented fields (i.e. CV and NLP).
Maybe *"Be aware of your data and use case."* would be a fitting *tl;dr*, but I think to really get what it means I have to go a bit further in my explaination.
Our road map is pretty simple.
First, we take a short tour of the field of PDM and define the specific task I am researching, remaining useful lifetime (RUL) estimation.
Second, we investigate the problem with this task and how to alleviate it.
Last, we will see how the current literature fails to do so by naively copying techniques from CV and NLP.

## Predictive Maintenance, Prognostics and Remaining Useful Lifetime Estimation

Predictive maintenance (PDM) or sometimes called condition-based maintenance (CBM) is scheduling maintenance for machines based on the automatic prediction of a fault contition.
The aim is to reduce the cost of repair and avoid downtime of the machine.
Traditionally, there are ways to predict the machine status through physical models or expert systems, but we will focus on data-driven approaches (i.e. deep neural networks).

Most times, the data used for PDM are multivariate time series.
The channels of the series constitute of sensor readings during operation (e.g. vibration or temperature) or test readings done at specific time intervals.
Label information is mostly provided as one label for each series.
Naturally, PDM data is imbalanced in favor of data from healthy, normally running machines.
Data from failing machines is much rarer as failures should be the exception, not the norm.

Broadly, we can divide PDM tasks into two categories based on when to predict, relative to the point of failure.
*Diagnostics* aims to predict if the system is running normally or if a fault is present.
This is normally framed as a classification problem with *n+1* classes, one for the normal state and one for each fault.
*Prognostics*, on the other hand, tries to predict when the point of failure will occur in the future.
This family of tasks is much harder as we have to detect an onsetting failure before it even occurs.
Obviously, prognostics is limited to degradation type failures, e.g. wear on a bearing, because it is impossible to predict sudden-onset failures like a goose flying into a jet engine.

Prognostics problems are often framed as regression tasks, where we predict the remaining number of equally long operation cycles until the machine breaks.
This is called remaining useful lifetime (RUL) estimation.
While there are several approaches to do RUL estimation, we will focus on the most direct approach, where we use a DNN to predict the RUL.
For all math fetishists out there, we try to model $$P(RUL|X)$$ with a function $$$f_\Theta(x_t) = rul_t$$ where $$f_\Theta$$ is a neural network parametrized by the weight vector $$\Theta$$, $$x_t$$ is a multivariate time series until time step $$t$$ and $$rul_i$$ the RUL at the same time step.

## Labeling is Hard

As mentioned before, PDM data is highly imbalanced.
The majority of the data is from healthy machines and only a few samples from failing machines are available.
RUL estimation has an additional problem.
The data consists of time series of sensor readings from the last repair ($$t=0$$) to the point of failure ($$t_{max}$$).
The label information here is the remaining lifetime at each step of the time series defined as $$RUL(t) = t_{max} - t$$.
Obviously, we can only calculate the labels if $$t_max$$ is known, i.e. the machine failed.
Considering that failure occurs only after significant time, we can see that we accumulate a lot of unlabeled data before we are able to assign a label.


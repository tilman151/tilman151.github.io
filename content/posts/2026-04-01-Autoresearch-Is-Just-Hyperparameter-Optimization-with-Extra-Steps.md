---
title: "Autoresearch Is Just Hyperparameter Optimization With Extra Steps"
slug: autoresearch-hyperopt
date: 2026-04-01
tags: [ai, llm, deeplearning]
mathjax: false
cover:
  image: "/img/autoresearch_comparison_progress.png"
  hidden: true
---

The last few weeks, the [autoresearch](https://github.com/karpathy/autoresearch) repository by Andrej Karparthy has made some waves.
Everybody seemed to be hyped for LLMs doing deep learning research, while I had a look at the README and thought: "Well, that sounds like hyperparameter optimization with extra steps."
Below you can see the progress plot Karparthy published as part of the repo.
The LLM runs 83 experiments over eight hours and successfully reduces the validation metric by around 0.0282 bits per byte.

{{< figure
  src="/img/autoresearch_org_progress.png"
  link="https://github.com/karpathy/autoresearch/blob/master/progress.png"
  alt="The progress plot from the original autoresearch repository."
  caption="The original progress plot."
>}}

Each of the 15 hyperparameter improvements is labeled.
Even though the LLM has complete freedom over changing the training code, all but the last improvement are simple hyperparameter tweaks any classic optimization algorithm would propose.
The last improvement is changing the seed, which may be admissible in an industry context, but not in an academic one.
My suspicion, that this was a classic case of using LLMs for problems we already have solutions for, was gaining ground.

Before anyone comes at me, I know that autoresearch is just an example of doing research with autonomous LLM agents.
On the other hand, if this example (i.e., hyperparameter optimization) is something we could do before but now with expensive API calls, it is a bad example.
When a person of authority, like Karparthy, gives such an example, it tends to work as an anchor point for other people.
This may limit what others try to do with the idea and perpetuates the bad example.

The question on which the whole argument hinges is: can the LLM do better than classic algorithms?
Maybe the LLM has more insight into the training process or works more flexibly.
To give a better frame of reference to the discussion, I set out to answer this question.

## Claude Code vs. Optuna

Unfortunately, I don't have access to an H100 GPU which is recommended to run autoresearch.
Therefore, I followed Karparthy's instructions on adapting the training script for smaller setups[^1], which primarily meant switching to the `TinyStories-gpt4-clean` dataset.
This dataset is much smaller than `climbmix-400b-shuffle` which was originally used, but I saw no reason why findings wouldn't be generalizable.
The adapted code can be found in [the forked repository](https://github.com/tilman151/autoresearch-hyperopt).
All changes to the repository were made in a few hours using a coding agent.

I chose [optuna](https://optuna.org/) to provide the classic hyperparameter optimization and Claude Code as the LLM agent.
Claude Code received the code adapted for the smaller setup and the original `program.md` file[^2].
In optuna, I selected all hyperparameters defined as module-level constants for optimization.
Additionally, the RoPE base frequency is optimized because it was an improvement made by Karparthy's original run[^3].

Optuna supports several samplers for generating hyperparameter configurations, of which I selected three:

- **Random Search:** a classic among the classics, still sometimes regarded as the optimal choice.
- **Gaussian Process:** uses a Gaussian Process to determine the most promising areas in the search space.
- **CMA-ES:** covariance matrix adaptation evolution strategy, which was new to me ([Paper](https://arxiv.org/pdf/2402.01373)).

All contestants had a time budget of eight hours.
A training run on my setup took slightly longer than 5 minutes[^4] because I did not find the real FLOPS for my GPU which are used to determine each run's compute budget.
The coding agent just hallucinated one with a plausible magnitude.
Because I'm not mad, I also locked down access to the Internet while running Claude Code unattended.

## Results

I know this is what you're here for, so let's look at the new progress plot.

{{< figure
  src="/img/autoresearch_comparison_progress.png"
  link="https://github.com/tilman151/autoresearch-hyperopt/blob/master/progress.png"
  alt="Progress plot comparing optuna's classic hyperparameter optimization algorithms with autoresearch running on Claude Code."
  caption="Progress plot comparing optuna's hyperparameter optimization with autoresearch running on Claude Code. The runs of Random Search are cropped for readability."
>}}

There is no denying it, each of the classic algorithms outperforms Claude Code.
Claude started with a slightly higher baseline, even though the seed was fixed, but the final performance difference is much larger than that.
The Gaussian Process and CMA-ES are nearly tied with Random Search coming in second place.
The two gaps in Gaussian Process' line are due to failing runs.

Optuna was also able to investigate more hyperparameter configurations than Claude as it did not have the overhead of the agent editing the training script.
Random search actually ran over 2.5k different configurations because it stopped the ones that were not promising (pruning).
The other two algorithms don't support early stopping, but in the end, I think the aggressive pruning hurt the performance of Random Search more than it helped.
Below are the final results in table form.

|                             | Random Search | Gaussian Process | CMA-ES | Claude Code |
|:----------------------------|---------------|------------------|:-------|:------------|
| **Best BPB**                | 0.4631        | 0.4587           | 0.4586 | 0.4667      |
| **Improvement**             | 0.0248        | 0.0282           | 0.0293 | 0.0222      |
| **Runs**                    | 2503          | 74               | 76     | 47          |
| **Runs better than Claude** | 1             | 22               | 28     | -           |
| **API Costs**               | -             | -                | -      | $27.21      |

Think about it for a moment.
Gaussian Process and CMA-ES found not only one configuration better than Claude Code, but more than 20 each.
These algorithms did not get lucky, they were simply better equipped for the job.
They did this while being limited to a predefined set of hyperparameters, while Claude Code could and did dynamically change a much larger set.
The final straw for me was the price tag of the Claude Code run.
At more than $27 in API costs, I was really thankful for the weak Dollar to Euro course.

## Conclusion

So at the low cost of about $27, you can get LLM-flavored hyperparameter optimization that is worse than what we already have.
Autoresearch, therefore, remains an interesting but flawed proposition.
LLMs can be insanely useful, but only for problems for which no better solutions are available.

The coding agent didn't even try to change something fundamental about the training, although it had permission to do so.
I think this reveals a fundamental issue: why do we assume an LLM has more insight into training deep neural networks than a human researcher?
There is no first-principle approach to deep learning that I am aware of, so even top-tier researchers can only take educated guesses when changing the training regime.
This is why optimization libraries like optuna exist: we do not know if something works until we ran the training.

I already saw people using autoreseach to optimize their Claude skills or RAG prompts.
These seem more suitable applications, although I think [DSPy](https://dspy.ai/) did it first.
For hyperparameter optimization, a happy middle ground may be asking a coding agent to set up a classic algorithm with the most promising hyperparameters.
This would integrate the flexibility of LLM agents with the specialized capabilities of classic algorithms.
I originally intended to run an experiment like this, too, but I was unwilling to spend any more money on Claude Code.

My only wish going forward is that we stop throwing LLMs at problems we already have better solutions for.

[^1]: an RTX 3060 in my case
[^2]: git tag `for-agent`
[^3]: I'm already cheating, I know
[^4]: 6.4 minutes ± 29 seconds to be precise
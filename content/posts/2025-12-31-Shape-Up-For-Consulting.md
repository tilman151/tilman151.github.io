---
layout: post
title: "Shape Up for Consulting Work"
slug: "shape-up-consulting"
date: 2025-12-31
tags: ["project management", "book review"]
mathjax: false
---

During the holidays I found the time to read a surprisingly wonderful book: Shape Up by Rian Singer.
I came about it in a blog post about alternative engineering practices on [daily.dev](https://daily.dev), which I obviously forgot to bookmark and, therefore, cannot mention here (nevermind, [found it](https://newsletter.manager.dev/p/5-engineering-dogmas-its-time-to)).
For the people who do not know Rian Singer (I was among them before reading the book), he is a product strategist at 37signals, the company that builds Basecamp.
The Shape Up book details the project management process of the same name used at 37signals.
As a longterm admirer of Basecamp, I was immediately intrigued.

In short, Shape Up is an alternative to SCRUM or similar methods for agile project management.
Its goal is to enable teams to build and deliver software in a predictable, flexible, and efficient way.
If you want to know more about Shape Up, just go read the book [for free on the Basecamp website](https://basecamp.com/shapeup).
This post, however, is on what I've learned from it for my consulting practice.
Even though the book itself is focused on organizing product development in a company, I found several parts useful for consulting work.
So without further ado, here are my top-three insights.

## Shaping Work

When preparing a work item, either for yourself or for others, the main question is: how specific do I need to get?
In consulting, you're often brought in as the only person in a project.
Writing down tasks (if you are writing them down at all) results, more often than not, in title-only tickets or a trusty, old `todo.txt` file.
This is fine.
You know your own mind.
The ticket is more of a knot in a handkerchief than anything else.

After a while, another person may get onboarded to the project, and you need to coordinate.
You're senior to the project, knowing it inside and out, and it is up to you to delegate tasks.
A oneliner won't cut it anymore, so you pour all of your thought process into the ticket.
The result is a magnum opus of a ticket: 600 words, graphs, links to papers, the whole shebang.
You basically solved the problem in advance.
Your teammate needs to simply implement it in code.

Tickets like this are problematic in more than one way.
The first one is the time it takes to write it.
You could have implemented it yourself in the same amount of time.
The whole advantage of having another person on the team goes down the drain.
The second part is more devious.
By solving the problem in such detail before handing it over, you rob your teammate of the opportunity to do so themselves.
They are a code monkey executing your vision.
They will never learn from this experience.
They will run into any pitfall you did not forsee.
Don't treat your teammate like a code monkey.
We've got LLMs for that.

Shaping work, as defined by the Shape Up process, is about finding the right balance between the two extremes described above.
Shaped work (as in work that went through shaping) lives in the realm between vague ideas and concrete execution plans.
It has three properties:

> 1. it's rough
> 2. it's solved
> 3. it's bounded

Roughness helps your teammates focus on the right question.
If you've ever thrown together a shiny demo UI for your ML model to demonstrate user interaction, only for the audience to remark that they don't like the button color, you know what I mean.
Work items need to communicate their level of uncertainty.
The question is if the user interaction is the right one, not if the button is blue.
I myself fall into this trap often enough, as I love to build polished demos.

The first point may sound contradictory to the second one.
Either a problem is solved or it is rough.
But solved, in this context, means that a problem was considered thoroughly.
As the ticket creator, I am certain that a solution exists, and, even more important, that the solution is the preferred one.
This is closely related to the [XY problem](https://en.wikipedia.org/wiki/XY_problem) often encountered in consulting.
The client may hire you to do something specific but actually needs something completely different.
As a consultant, a large part of your job is to recognize this pattern.

Bounding a work item may seem easy: you just tell your teammate what to do.
But there is another bound you need to consider: telling them when to stop.
In data-centric work, like machine learning, this is especially devious.
No model works with 100% accuracy (in the loose sense of the word), but chasing this elusive number is still tempting.
In a recent project, the client needed me to build a model predicting settings for one of their processes.
These settings should make the process hit a certain KPI, which was the model input.
The model worked reasonably well on the validation set, but the stakeholders wanted to continue pushing model performance.
It took some time to convince them to build the whole model-user interaction first to see how it turns out.
After a test period in production, it became obvious that the recommended settings worked well for the first run of the process.
Every subsequent run would then underperform and miss the KPI target.
This was because the model only took the current state of the process into account and had no understanding of long-term system stability.
Even if we had spent more time pushing model performance, we would have encountered the same problem at a much later time.
Knowing when to stop was key.

The book gives many more examples on these properties of shaped work.
I found the proposed techniques of _breadboarding_ and _fat marker sketches_ for communicating on the right level of roughness really useful.
Hopefully, I can incorporate them in my future work.

## The Six-Week Cycle

One of the core tenants of Shape Up is the six-week cycle.
Anyone who used SCRUM before will find it familiar.
The length of the cycle is what sets it apart from other agile methods.
The book states that six weeks is enough time to build something meaningful but is short enough to feel the deadline looming immediately.
This resonated with me on a deeper level.

I was part of data teams before that used SCRUM, and I never felt it click.
Data-centric work is often even harder to estimate than conventional programming work.
Maybe the first approach we try to predict something will work out perfectly.
But maybe we will spend a whole month banging our head against the problem.
Sitting down every two weeks just to push the same old tickets into the next sprint felt deeply unsatisfying.
One team I was a member of tried something different by using Kanban.
The cycle-less approach with dynamic reevaluation of priorities seemed to fit the nature of data-centric work much better.
On the other hand, it made work seem an endless slog because there was no natural deadline to work towards.
Six weeks seems to be a good middle ground here.
To be fair, SCRUM never codified the two-week sprint length explicitly (the [official guide](https://scrumguides.org/docs/scrumguide/v2017/2017-Scrum-Guide-US.pdf) says one month or less), but nobody seems to be deviating from this default.

Another thing these longer cycles reduce is planning overhead.
For every two-week sprint, a major portion of time goes into planning, review, retro, and other SCRUM ceremonies.
These time commitments do not scale with the cycle length, so prolonging the cycle reduces their impact on time available.
The project team has even more time on their hands because work is shaped by a different team in parallel.
No more planning and grooming sessions with the whole team where two people talk and the rest either works on something different or zones out completely.

A separate shaping team may sound like a step back from what agile methods brought us: an empowered team of developers taking end-to-end responsibility.
But let's remember what shaped work means.
It is not about taking responsibility away or providing complete solutions in a top-down manner.
Shaping lays the groundwork for project teams to make informed decisions and _take_ responsibility in the first place.
It has to happen _before_ the actual implementation work takes place, so why not parallelize it.
Many practitioners think that in agile projects, everybody needs to be present when decisions are made, even if only a single person (e.g., the project manager) makes the decision.
Shape Up is more honest in this regard, highlighting who is supposed to make which kind of decision at which point in the work item's lifecycle.

Estimation gets easier, too (even though Shape Up does not estimate but assigns appetite).
The inherently hard question of _"How long will this take?"_ is transformed into _"Can we finish it in six weeks or fewer?"_.
No more story points, or T-Shirt sizes, or planning poker.
Just a simple yes-or-no question.
The book talks about _hammering_ the scope of a work item to fit into the slot of a cycle.
This needs to be done because, by default, work items that miss their deadline are not automatically extended.
The book calls this a _circuit breaker_.
If a work item, contrary to our assumption, could not be finished in one cycle, it is rarely on the developers.
More often than not, there are pitfalls and hidden complexities, which were not considered while shaping, that prevent completion.
Continuing without rethinking invites runaway projects that go on forever.

One last delightful detail of the Shape Up process is the Cool Down.
It is a two-week period after each cycle without explicitly scheduled tasks.
Anyone who tried to argue for a sprint of fixing technical debt at least once will appreciate that there is a dedicated time for cleanup, documentation, and bug fixing baked into the agile method itself. 

## Tracking Progress with Hill Charts

Almost everybody dreads this question: "How is the project moving along?"
Communicating progress to managers and stakeholders is a tricky process.
Even inside the team, measuring progress is not straight forward.
But why is that?
The Shape Up book proposes a compelling answer by splitting a project's tasks in _imagined_ and _discovered_.
Imagined tasks are the ones that come up beforehand, e.g., during shaping.
Discovered tasks, on the other hand, only reveal themselves when actual implementation work is done.
If the work item at hand is not completely mundane and/or trivial, there will be tasks waiting to be discovered.
Concluding this thought, the percentage or number of tasks done cannot be a good indicator of progress.
At any point, new tasks could be discovered, and a good progress indicator should not move backward.

I often see this kind of uncertainty when working with Objectives and Key Results (OKRs).
Key Results are the measurable part of the framework, so people simply attach a percentage and call it a day.
I was part of two companies that did it like this and heard stories from many others.
What you will often see is that the percentage stays at zero or one-digit for most of the time and then jumps to almost complete near the end of the OKR period.
Early on, nobody dares to estimate progress due to the uncertainty of undiscovered tasks.
After all tasks are discovered, the team notices that a lot of them are done already and updates the percentage.
What we often ignore is that discovering tasks is progress as well.
It sounds obvious in retrospect, but laying these things down in a systematic manner throughout the book is a strength of Rian Singer.

The Shape Up process tries to encode the uncertainty of discovery in a visualization named _hill chart_, which you can see below.

{{< figure
  src="https://basecamp.com/assets/images/books/shapeup/3.4/hill_concept.png"
  link="https://basecamp.com/shapeup/3.4-chapter-13#work-is-like-a-hill"
  alt="An annotated example of a hill chart. Uphill (left) is the discovery phase. Downhill (right) is the execution phase."
  caption="Copyright ©1999-2025 37signals LLC. Click on the image to go to the chapter with this figure."
>}}

A work item starts at the bottom left and goes uphill first.
Going uphill means that more tasks are discovered than finished.
The work is mainly figuring out the unknown.
After cresting the hill, all tasks are completed by going downhill. 
The team moves their work item along the hill's surface to indicate their progress.
This metaphor acknowledges that there are two phases to completing a work item and gives the team a way to communicate uncertainty.

I can imagine these charts insanely useful for stakeholder communication.
They provide a much more nuanced snapshot of progress in a compact form factor.
The second order information is helpful as well.
As the book points out, managers can asynchronously check the changes to the hill chart.
Intervention from their side is only needed when an item does not progress over the hill.
No need to bother the team otherwise.
My main concern here is if client stakeholders are open to this kind of reporting.
These charts are a bit unusual and would require onboarding clients first.

## Conclusion

I cannot recommend this book enough.
It is written nicely, not too long, and [free online](https://basecamp.com/shapeup).
The examples and anecdotes are relatable for anyone who has ever built software, and the bits of dry humor make for a good chuckle.
I'm looking forward to implementing my findings in the future and hope that clients find them as delightful as I do.
The people at 37signals wrote some other books, too, which I will definitely need to have a look at.
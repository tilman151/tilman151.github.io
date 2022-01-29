---
layout: post
title: "Compose Datasets, Don't Inherit Them"
categories: cleancode
---

In relatively young disciplines, like deep learning, people tend to leave behind old principles.
Sometimes this is a good thing because times have changed and old truths, i.e. over-completeness being a bad thing, have to go.
Other times, such old principles stick around for a reason and still people over-eagerly try to throw them out of the window.
I am no exception in this regard so let me tell you how I "re-learned" the tried and true design pattern of "Composition over Inheritance".

Dataset code is, in my opinion, one of the messiest and hardest parts of any deep learning project.
And with datasets, I mean the part of your program that loads data from disk and puts it into tensors for your model to consume during training.
Data comes in all shapes and forms.
Even if it is intended for the same task, transforming it into a representation suitable for your model is complicated.
Furthermore, there are often as many hyperparameters to be configured in your dataset as in your model architecture.
So the question here is:

> How do I write datasets that are reusable, configurable, and handle different kinds of data without repeating myself all the time?

In computer vision, this is less of a problem.
Most data comes as JPEGs in folders.
Just use `torchvision.datasets.DatasetFolder`, have a nice day, and thank you.
In other disciplines, like NLP, it is a little more complicated.
You get one or multiple CSVs in several dialects, heaps of plain text files with labels encoded in the file name, databases, or even worse.
The things you want to do with this data, in the end, are pretty similar: tokenize it, build a vocabulary, get token IDs for your samples.
I will use a simple text classification task as an example to show you how my dataset code tended to evolve, what went wrong, and how I made it better.
By the way, we will be using PyTorch and its companion package torchtext.

You can find all the code I am going to use at: [https://github.com/tilman151/composing-datasets](https://github.com/tilman151/composing-datasets).
As I mentioned, we will look at how the code changes over time, so I am going to show code snippets at different points in the commit history of the repository.
The relevant commits have a version tag (e.g. v0.1.0) that I will use to refer to them.
Each version is tested and should be usable in its current state if you want to try it out.

## Humble Beginnings

We want to start small and do some simple, preliminary experiments with our text classification architecture.
Therefore, we will use the [Automated Hate Speech Detection and the Problem of Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language/) or hate speech dataset for short.
It consists of tweets that are either labeled hate speech, offensive, or neither.
It is rather small, at about 2.5MB, and comes in a single CSV file.
We can write a short and simple PyTorch dataset to load it and tag it as [*v0.1.0*](https://github.com/tilman151/composing-datasets/tree/v0.1.0).

```python
from torch.utils.data import Dataset

# [...]

class HateSpeechDataset(Dataset):
    DOWNLOAD_URL: str = "..."
    DATA_ROOT: str = "..."
    DATA_FILE: str = "..."
  
    def __init__(self) -> None:
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_FILE)
        self.text, self.labels = self._load_data()

        self.tokenizer = torchtext.data.get_tokenizer("basic_english")
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

   # [...]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.text)
```

The dataset creates a folder for the data and downloads the CSV file if it is not already there.
As the amount of data is relatively small, we can load it into memory all at once.
We use the simple `basic_english` tokenizer from the `torchtext` package and build a vocabulary from all tokens and a default out-of-vocabulary token.
At last, the token ids for each tweet and the label are stored as tensors in two lists.
The `__get_item__` function simply indexes these two lists.
The advantage of this dataset is that retrieving samples directly from memory is fast but comes at the cost of a longer initialization time.

Instantiating the dataset is as simple as it gets because there is nothing to configure.
But, as I said, data representation matters for deep learning, so we may want to experiment with different ones.
Let's say, we want to try out another tokenizer, e.g. one from the revtok package.
Good for us that the `get_tokenizer` function from `torchtext` already provides a way to retrieve this tokenizer by passing `"revtok"`.
We just add a string argument named `tokenizer` to the constructor and pass it to the function.
Making this argument optional preserves  the original feature of calling it without arguments to get the `basic_english` tokenizer.
This is version [`0.2.0`](https://github.com/tilman151/composing-datasets/tree/v0.2.0) of our dataset.

```python
class HateSpeechDataset(Dataset):
    DOWNLOAD_URL: str = "..."
    DATA_ROOT: str = "..."
    DATA_FILE: str = "..."

    def __init__(self, tokenizer: Optional[str] = None) -> None:
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_FILE)
        self.text, self.labels = self._load_data()

        self.tokenizer = self._get_tokenizer(tokenizer)
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

    # [...]

    def _get_tokenizer(self, tokenizer: Optional[str]) -> Callable:
        if tokenizer is None:
            tokenizer = torchtext.data.get_tokenizer("basic_english")
        else:
            tokenizer = torchtext.data.get_tokenizer(tokenizer)

        return tokenizer

    # [...]
```

Through `get_tokenizer` we have access to several tokenizers, like `spacy` or `moses`, which makes it quite flexible.
All in all, not a big code change.

## The More, the Merrier

Ok, now we fooled around enough with our tiny hate speech dataset and want to go a little bigger.
What about the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) or Imdb dataset for short?
We get a user's movie review and have to predict the number of stars this user gave the movie.
Each review comes as a plain text file with the star rating encoded in the file's name.

Basically, we want to do exactly the same data processing as with the hate speech dataset but have to load a different data format.
Our *Intro to CS* knowledge tells us that this is a job for inheritance.
Put the shared functionality in a base class and do the specialized data reading in a child class.
This brings us to version [0.3.0](https://github.com/tilman151/composing-datasets/tree/v0.3.0) of our code.

```python
class TextClassificationDataset(Dataset, metaclass=ABCMeta):
    DOWNLOAD_URL: str
    DATA_ROOT: str
    DATA_FILE: str

    # [...]
    
    @abstractmethod
    def _load_data(self) -> Tuple[List[str], List[int]]:
        pass

    
    # [...]

class HateSpeechDataset(TextClassificationDataset):
    DOWNLOAD_URL: str = "..."
    DATA_ROOT: str = "..."
    DATA_FILE: str = "..."

    def _load_data(self) -> Tuple[List[str], List[int]]:
        # [...]


class ImdbDataset(TextClassificationDataset):
    DOWNLOAD_URL: str = "..."
    DATA_ROOT: str = "..."
    DATA_FILE: str = "..."

    # [...]

    def __init__(self, split: str, tokenizer: Optional[str] = None) -> None:
        if split not in ["train", "test"]:
            raise ValueError("Unknown split supplied. Use either 'train' or 'test'.")
        self.split = split
        super(ImdbDataset, self).__init__(tokenizer)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        # [...]
```

Our base class `TextClassificationDataset` handles everything besides reading the data from disk.
Instead, it has an abstract `_load_data` function that has to be implemented by its children.
This makes adding new datasets really easy.
Just provide the class variables (`DOWNLOAD_URL`, etc.) and implement `_load_data`.
For the Imdb dataset, we have to provide a separate constructor, too, because it comes with a pre-defined train-test split.
We have to pass the desired split as an argument so that `_load_data` knows which one to load.

With the automatic refactoring tools of modern IDEs, this is, again, not a big change but enables us to reuse our code for new data with ease.

## Diverging Paths

We did some experiments on the Imdb data and came to the conclusion that we need more control over the tokenizer.
For example, the `revtok` tokenizer has options that control the capitalization of tokens and if splits should be made on punctuation.
We want to try these new configurations on the smaller hate speech dataset but then use it on the Imdb data with minimal effort.

An easy solution would be adding a dictionary argument to the dataset constructor which contains the keyword arguments to configure the tokenizer.
The code below is version [0.3.1-a](https://github.com/tilman151/composing-datasets/tree/v0.3.1-a) of our code and showcases this approach for the `revtok` tokenizer.

```python
class TextClassificationDataset(Dataset, metaclass=ABCMeta):
    DOWNLOAD_URL: str
    DATA_ROOT: str
    DATA_FILE: str

    def __init__(self, tokenizer: Optional[str] = None, **kwargs) -> None:
        os.makedirs(self.DATA_ROOT, exist_ok=True)
        if not os.path.exists(self.DATA_FILE):
            _download_data(self.DOWNLOAD_URL, self.DATA_ROOT)
        self.text, self.labels = self._load_data()

        self.tokenizer = self._get_tokenizer(tokenizer, kwargs)
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

    # [...]

    def _get_tokenizer(
        self, tokenizer: Optional[str], kwargs: Dict[Any, Any]
    ) -> Callable:
        if tokenizer is None:
            tokenizer = torchtext.data.get_tokenizer("basic_english")
        elif not kwargs:
            tokenizer = torchtext.data.get_tokenizer(tokenizer)
        elif tokenizer == "revtok":
            tokenizer = _RevtokTokenizer(kwargs)
        else:
            raise ValueError(f"Unknown tokenizer '{tokenizer}'.")

        return tokenizer
        
    # [...]


class _RevtokTokenizer:
    def __init__(self, kwargs: Dict[Any, Any]) -> None:
        self.kwargs = kwargs

    def __call__(self, x: str) -> List[str]:
        return revtok.tokenize(x, **self.kwargs)
```

We needed to introduce a wrapper class for the `revtok` tokenizer because it is just a function.
A lambda function may have been sufficient but this way it is easier to unit test.
The `_get_tokenizer` function is now a little more complicated as we have to check for `revtok` and `kwargs` specifically, and cannot solely rely on the `torchtext` function anymore.
We would need to change this function for each new tokenizer we want to configure via `kwargs`.
Of course, we need to change the constructor of the Imdb dataset, as well, for passing `kwargs` to the super class' constructor.

This approach is simple but has some drawbacks.
First, as mentioned before, you have to change `_get_tokenizer` for each new tokenizer.
Second, you cannot use auto-complete or other IDE functions to look up the arguments for your chosen tokenizer.
Third, if we want to add more configurable elements, the number of constructor arguments will get large quickly.
Furthermore, the names of these arguments get longer, too, as we would have to specify that `kwargs` is in fact `tokenizer_kwargs`.
All of this makes this version a bit clunky to use.
Could more inheritance solve these problems?
We could use specialized classes with fitting constructors for each tokenizer and pass arguments without resorting to dictionaries.
Inheritance helped once, why not twice?

We can make `_get_tokenizer` abstract and create a child that gives us the `torchtext`-based implementation and another one that implements a configurable `revtok` tokenizer specifically.
For both, hate speech and Imdb, we now have two classes.
One that descends from the base class plus the `torchtext` implementation and one descending from the base class plus the `revtok` implementation.
This version is tagged [v0.3.1-b](https://github.com/tilman151/composing-datasets/tree/v0.3.1-b).

Even though this solves the second problem of 0.3.1-a, it creates several others.
We introduce even more classes.
So many that I even refrain from showing them here.
For new data and new tokenizers, we need not only a new class for both but additional classes for each resulting data-tokenizer combination.
A combinatorial explosion of classes is imminent.
Thoroughly testing all of these combinations is an exercise in futility.
Multiple inheritance has its own problems, as well.
The inheritance order in Python is from right to left.
This means that the class implementing `_load_data` has to be left of the class defining the abstract `_load_data` to override it.
It won't work the other way around.

This problem of an ever expanding-amount of subclasses is a known problem when using inheritance extensively.
It may be fine while inheritance follows only one path of specialization but fails when multiple paths are available.
In our example, one path is each of our datasets (hate speech and Imdb) and the second one is the tokenizer to use.
If we were to add a third path, like configuring pre-trained embeddings, the number of necessary subclasses would increase even more.

## Composing a Solution

Composition is a technique where an object is constructed from other objects in a building block fashion.
Each building block fulfills a specific functionality but capsules its inner workings away from the outer object.
In our case, we have two different types of building blocks: the ones that read the text and the labels from disk, and the tokenizer.
Composing a dataset would be as easy as passing the constructor an object that reads the data of our choice and a tokenizer object.
Both are configured outside the dataset object which gives us great flexibility.
Version [0.4.0](https://github.com/tilman151/composing-datasets/tree/v0.4.0) of our code adheres to these principles.

```python
class TextClassificationDataset(Dataset):
    def __init__(
        self, dataset: Dataset, tokenizer: Union[str, Callable, None] = None
    ) -> None:
        self.dataset = dataset
        self.text, self.labels = tuple(zip(*self.dataset))

        self.tokenizer = self._get_tokenizer(tokenizer)
        self.tokens = [self.tokenizer(text) for text in self.text]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokens)
        self.vocab.set_default_index(len(self.vocab))
        self.token_ids = [self._tokens_to_tensor(tokens) for tokens in self.tokens]
        self.labels = [torch.tensor(label, dtype=torch.long) for label in self.labels]

    def _get_tokenizer(self, tokenizer: Optional[str]) -> Callable:
        if tokenizer is None:
            tokenizer = torchtext.data.get_tokenizer("basic_english")
        else:
            tokenizer = torchtext.data.get_tokenizer(tokenizer)

        return tokenizer

    # [...]


class HateSpeechDataset(Dataset, DownloadDataMixin):
    
    # [...]

    def __init__(self):
        self._download_data()
        self.text, self.labels = self._load_data()

    # [...]

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.text[index], self.labels[index]


class ImdbDataset(Dataset, DownloadDataMixin):
    
    # [...]

    def __init__(self, split: str) -> None:
        if split not in ["train", "test"]:
            raise ValueError("Unknown split supplied. Use either 'train' or 'test'.")
        self.split = split

        self._download_data()
        self.text, self.labels = self._load_data()

    # [...]

    def __getitem__(self, index: int) -> Tuple[str, int]:
        return self.text[index], self.labels[index]
```

The classes reading the data from disk adhere to the `torch` dataset interface and return a text string and its integer label.
Any object that implements this interface can be passed to our `TextClassificationDataset`.
We can reuse the `_get_tokenizer` function from version 0.3.0 because of the versatile design of `get_tokenizer` from `torchtext`.
It can receive a callable object and simply pass it through.
This way, we can still use `tokenizer` as a simple string argument or pass a fully configured tokenizer object.
To instantiate a hate speech dataset with a custom revtok tokenizer is as simple as:

```python
TextClassificationDataset(HateSpeechDataset(),
                          lambda x: revtok.tokenize(x, split_punctuation=True))
```

Adding new data this way is as easy as implementing the `torch` dataset interface.
Most tokenizers can be used out of the box, as they follow the same interface of just being a callable.

## Composing Even Further

We have seen how composing our dataset makes our code more flexible.
Extending its functionality is simple, as well, by composing new datasets from `TextClassificationDataset`.
Need a train-val split for the hate speech data?

```python
from torch.utils.data import random_split

hate_speech = TextClassificationDataset(HateSpeechDataset())
num_samples = len(hate_speech)
train, val = random_split(hate_speech, [num_samples // 2] * 2)
```

The `random_split` function internally uses the `Subset` class which takes our base dataset and returns only a specified set of samples.
But now the vocabulary still contains tokens that are only in the validation data.
We may not want that.
No problem at all for our composed dataset.

```python
hate_speech = HateSpeechDataset()
num_samples = len(hate_speech)
train, val = random_split(hate_speech, [num_samples // 2] * 2)
train = TextClassificationDataset(train)
val = TextClassificationDataset(val)
```

This way the data is split before the vocabulary is constructed so that we avoid data leakage.
Now, let's imagine we want to use a pre-trained embedding layer.
Do we need to change our dataset class?
No, we don't.
We simply compose a new dataset.

```python
class PreTrainedEmbeddingDataset(Dataset):
    def __init__(self, dataset: Dataset, embeddings: torch.nn.Embedding) -> None:
        self.dataset = dataset
        self.embeddings = embeddings

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids, label = self.dataset(index)
        embedded_tokens = self.embeddings(token_ids)
        
        return embedded_tokens, label

    def __len__(self) -> int:
        return len(self.dataset)

```

Because all of our classes follow the same dataset interface, we can just keep stacking them.
This keeps the base dataset class short and adds functionality as needed.

Another advantage of composing datasets comes with powerful configuration frameworks like [hydra](https://hydra.cc/).
As hydra uses composition to structure config files, it works best with a codebase that uses composition, too.
But this is a topic for a later article.

## In Conclusion

This concludes our journey through the commit history of this project.
I hope you agree with me that version 0.4.0 is the superior version of our codebase.
It is relatively short, it is flexible, and it is easily extended.
By the way, it was easiest to test, as well.
You can check out the `tests` folder to see for yourself.

Composition over inheritance really seems to be not only a phrase but a tried and true principle of software design.
Even for deep learning.
As always, these design patterns have to be taken with a grain of salt.
Is composition better for each problem?
Probably not.
If your problem is of limited scope, inheritance may be the quicker solution.
It really depends.
Just keep in mind that composition is in your toolbox so that you don't repeat yourself.
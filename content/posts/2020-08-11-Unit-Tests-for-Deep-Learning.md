---
layout: post
title: "How to Trust Your Deep Learning Code"
slug: deep-learning-unit-tests
date: 2020-08-01
tags: [cleancode]
aliases: 
    - /cleancode/2020/08/11/Unit-Tests-for-Deep-Learning.html
---

Deep learning is a discipline where the correctness of code is hard to assess.
Random initialization, huge datasets and limited interpretability of weights mean that finding the exact issue of why your model is not training, is trial-and-error most times.
In classical software development, automated unit tests are the bread and butter for determining if your code does what it is supposed to do.
It helps the developer to trust their code and be confident when introducing changes.
A breaking change would be detected by the unit tests.

If one can go by the state of many research repositories on GitHub, practitioners of deep learning are not yet fond of this method.
Are practitioners okay with not knowing if their code works correctly?
Often, the problem is that the expected behavior of each component of a learning system is not easily defined because of the three reasons above.
Nevertheless, I believe that practitioners and researchers should rethink their aversion to unit tests as it can help smooth the research process.
You just have to learn how to trust your code.

Obviously, I am not the first and, hopefully, not the last to talk about unit tests for deep learning.
If you are interested in the topic, you can have a look here:

* [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy
* [How to Unit Test Deep Learning](https://theaisummer.com/unit-test-deep-learning/) by Sergios Karagiannakos
* Chapter 9 (Unit Tests) of Clean Code by Robert C. Martin

This article was inspired by the ones above and probably many others I cannot recall at the moment.
To add something to the discussion, we will focus on how to write **reusable unit tests**, so that you *"Don't repeat yourself"*.

Our example will test the components of a system written in PyTorch that trains a variational autoencoder (VAE) on MNIST (creative, I know).
You can find all of the code from this article at [github.com/tilman151/unittest_dl](https://www.github.com/tilman151/unittest_dl).

## What are unit tests anyway?

If you are familiar with unit tests, you can skip this section.
For everyone else, we will see what a unit test in Python looks like.
For simplicity's sake, we will use the built-in package `unittest` and not one of the fancier ones.

The aim of a unit test, in general, is to see if your code behaves correctly.
Often (and I was guilty of this for a long time, too), you will see something like this at the end of a file:
    
```python
if __name__ == 'main':
    net = Network()
    x = torch.randn(4, 1, 32, 32)
    y = net(x)
    print(y.shape)
```
        
If the file is executed directly, the code snipped will build a network, do a forward pass and print out the shape of the output.
With that, we can see if the forward pass throws an error and if the shape of the output seems plausible.
If you have distributed your code in different files, you would have to run each of them by hand and review the things printed to the console.
Even worse, this snippet is sometimes deleted after running it and rewritten when something changes (not that I ever did that \*cough\*).

In principle, this is already a rudimentary unit test.
All we have to do is formalizing it a bit to make it run automatically with ease.
This would look like this:

```python
import unittest

class MyFirstTest(unittest.TestCase):
    def test_shape(self):
        net = Network()
        x = torch.randn(4, 1, 32, 32)
        y = net(x)
        self.assertEqual(torch.Size((10,)), y.shape)
```

The main component of the `unittest` package is the class `TestCase`.
A single unit test is a member function of a child class of `TestCase`.
In our case, the package would automatically detect the class `MyFirstTest` and run the function `test_shape`.
The test succeeds if the condition of the `assertEqual` call is met.
Otherwise, or if it crashes, the test fails.

If you need more information on how the package works, have a look at the official documentation [here](TODO).
In practice, I would always recommend the use of an IDE with test running integration, like PyCharm.
It lets you run the test you need at the press of a button.

## What should I test?

Now that we have an understanding of how unit tests work, the next question is what exactly we should test.
Below you can see the structure of the code of our example:

    |- src
       |- dataset.py
       |- model.py
       |- trainer.py
       |- run.py
  
The first three of the files contain what the name states, while the last one creates all components of training and starts it.
We will test the functionalities in each file but `run.py`, as it is just the entry point of our program.

### Dataset

The dataset we are using for our example is the `torchvision` MNIST class.
Because of this, we can assume that the basic functionality like loading the images and train/test splitting is working as intended.
Nevertheless, the MNIST class provides ample opportunity for configuration, so we should test if we configured everything correctly.
The file `dataset.py` contains a class called `MyMNIST` with two member variables.
The member `train_data` holds an instance of the `torchvision` MNIST class configured to load the training split of the data, while the instance in `test_data` loads the test split.
Both of them pad each image with 2 pixels on each side and normalize the pixel values between [-1, 1].
Additionally, `train_data` applies a random rotation to each image for data augmentation.

#### The shape of your data

To stick with the code snippet from above, we will first test if our dataset outputs the shape we intended.
Our padding of the images means, that they should now have a size of 32x32 pixels.
Our test would look like this:

```python
def test_shape(self):
    dataset = MyMNIST()
    sample, _ = dataset.train_data[0]
    self.assertEqual(torch.Shape((1, 32, 32)), sample.shape)
```

Now we can be sure that our padding does what we want.
This may seem trivial and some of you may think me pedantic for testing this, but I cannot count how many times I got a shape error because I confused how the padding function worked.
Trivial tests like this are quick to write and they can spare you a lot of headaches later on.

#### The scale of your data

The next thing we configured, was the scaling of our data.
In our case, this is very simple.
We want to make sure that the pixel values of each image lie between [-1, 1].
In contrast to the previous test, we will run the test for all images in our dataset.
This way, we can be certain that our assumptions on how to scale the data are valid for the whole dataset.

```python
def test_scaling(self):
    dataset = MyMNIST()
    for sample, _ in dataset.train_data:
        self.assertGreaterEqual(1, sample.max())
        self.assertLessEqual(-1, sample.min())
        self.assertTrue(torch.any(sample < 0))
        self.assertTrue(torch.any(sample > 0))
```
            
As you can see, we are not only testing if the maximum and minimum values of each image are within range.
We are also testing that we did not accidentally scale the values to [0, 1], by asserting that there are values above and below zero.
This test only works because we can assume that each image in MNIST covers the whole range of values.
In the case of more complicated data like natural images, we would need a more sophisticated test condition.
If you are basing your scaling on the statistics of your data, it is also a good idea to test that you only used the training split for calculating these statistics.

#### The augmentation of your data

Augmenting your training data helps the performance of your model tremendously, especially if you have a limited amount of data.
On the other hand, we would not augment our testing data, as we want to keep the evaluation of our models deterministic.
This means in turn, that we should test if our training data is augmented and our testing data is not.
The perceptive reader will notice something important at this point.
Up until now, we only covered the training data with our tests.
This is a thing to emphasize:

> Always run your tests on training and testing data

Just because your code works on one split of the data, it is not guaranteed that there isn't some undetected bug lurking with the other one.
For data augmentation, we even want to assert different behavior of our code for each split.

An easy test for our augmentation problem now would be loading a sample two times and checking if both versions are equal or not.
The trivial solution would be to write a test function for each of our splits:

```python
def test_augmentation_active_train_data(self):
    dataset = MyMNIST()
    are_same = []
    for i in range(len(dataset.train_data)):
        sample_1, _ = dataset.train_data[i]
        sample_2, _ = dataset.train_data[i]
        are_same.append(0 == torch.sum(sample_1 - sample_2))

    self.assertTrue(not all(are_same))

def test_augmentation_inactive_test_data(self):
    dataset = MyMNIST()
    are_same = []
    for i in range(len(dataset.test_data)):
        sample_1, _ = dataset.test_data[i]
        sample_2, _ = dataset.test_data[i]
        are_same.append(0 == torch.sum(sample_1 - sample_2))

    self.assertTrue(all(are_same))
```
        
These functions test what we wanted to test, but, as you can see, they are nearly duplicates of each other.
This has two major drawbacks.
First, if something has to be changed in the test, we have to remember to change it in both functions.
Second, if we would like to add another split, e.g. a validation split, we would have to copy the test a third time.
To solve this problem, we should extract the test functionality into a separate function that is then called two times by the real test functions.
The refactored tests would look like this:

```python
def test_augmentation(self):
    dataset = MyMNIST()
    self._check_augmentation(dataset.train_data, active=True)
    self._check_augmentation(dataset.test_data, active=False)

def _check_augmentation(self, data, active):
    are_same = []
    for i in range(len(data)):
        sample_1, _ = data[i]
        sample_2, _ = data[i]
        are_same.append(0 == torch.sum(sample_1 - sample_2))

    if active:
        self.assertTrue(not all(are_same))
    else:
        self.assertTrue(all(are_same))
```

The `_check_augmentation` function asserts if augmentation is active or not for the given dataset and effectively removes the duplication in our code.
The function itself will not be run automatically by the `unittest` package, as it does not start with `test_`.
Because our test functions are now really short, we merged them into one combined function.
They test the single concept of how augmentation works and should, therefore, belong to the same test function.
But, with this combination, we introduced another problem.
It is now hard to see directly for which of the splits the test fails if it does.
The package would only tell us the name of the combined function.
Enter, the `subTest` function.
The `TestCase` class has a member function `subTest` that makes it possible to mark different test components in a testing function.
This way, the package can show us exactly what part of the test failed.
The final function would look like this:

```python
def test_augmentation(self):
    dataset = MyMNIST()
    with self.subTest(split='train'):
        self._check_augmentation(dataset.train_data, active=True)
    with self.subTest(split='test'):
        self._check_augmentation(dataset.test_data, active=False)
```

Now we have a duplication free, precisely pinpointing, reusable test function.
The core principle we have used to get there, are transferable to all other unit tests we have written in the sections before.
You can see the resulting tests in the accompanying repository.

#### The loading of your data

The last type of unit test for datasets is not completely relevant for our example as we are using a built-in dataset.
We will include it anyway because it covers an important part of our learning system. 
Normally you will use your dataset in a dataloader class that handles batching and can parallelize the loading process.
Testing if your dataset works with the dataloader in both single and multiprocess mode is, therefore, a good idea.
Taking into account what we have learned with the augmentation tests, the testing functions look like this:

```python
def test_single_process_dataloader(self):
    dataset = MyMNIST()
    with self.subTest(split='train'):
        self._check_dataloader(dataset.train_data, num_workers=0)
    with self.subTest(split='test'):
        self._check_dataloader(dataset.test_data, num_workers=0)

def test_multi_process_dataloader(self):
    dataset = MyMNIST()
    with self.subTest(split='train'):
        self._check_dataloader(dataset.train_data, num_workers=2)
    with self.subTest(split='test'):
        self._check_dataloader(dataset.test_data, num_workers=2)

def _check_dataloader(self, data, num_workers):
    loader = DataLoader(data, batch_size=4, num_workers=num_workers)
    for _ in loader:
        pass
```

The `_check_dataloader` function does not test anything on the loaded data.
We simply want to check that the loading process throws no error.
Theoretically, you could check things like correct batch size or padding for sequence data of varying lengths, too.
As we are using the most basic of configurations for the dataloader, we can omit these checks.

Again, this test may seem trivial and unnecessary, but let me give you an example where this simple check saved me.
The project required to load our sequence data from pandas dataframes and construct samples from a sliding window over these dataframes.
Our dataset was too large to fit into memory so we had to load the dataframes on demand and cut out the requested sequence from it.
To increase loading speed, we decided to cache a number of dataframes with a [LRU cache](https://docs.python.org/3.7/library/functools.html#functools.lru_cache).
It worked as intended in our early single process experiments, so we decided to include it in the codebase.
Turns out that this cache did not play nicely with multiprocessing, but our unit test caught that problem well in advance.
We deactivated the cache when using multiprocessing and avoided unpleasant surprises later.

#### Last remarks

Some of you may have seen another duplicate pattern in our unit tests now.
Each of the tests is run once for training data and one for testing data, resulting in the same four lines of code:

```python
with self.subTest(split='train'):
    self._check_something(dataset.train_data)
with self.subTest(split='test'):
    self._check_dataloader(dataset.test_data)
```

It would very well be justified to remove this duplication, too.
Unfortunately, this would involve creating a higher-order function that takes the function `_check_something` as an argument.
Sometimes, e.g. for the augmentation tests, we would need to pass additional parameters to the `_check_something` function, too.
In the end, the programming constructs needed would introduce much more complexity and obscure the concepts that are tested.
A general rule is, to make your test code only as complicated as needed for readability and reusability.

### Model

The model is arguably the core component of your learning system and oftentimes needs to be completely configurable.
This means, that there are a lot of things to test, too.
Fortunately, the API for neural network models in PyTorch is really concise and most practitioners adhere to it pretty closely.
This makes writing reusable unit tests for models fairly easy.

Our model is a simple VAE consisting of a fully-connected encoder and decoder (if you are unfamiliar with VAEs, look [here](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)).
The forward function takes an input image, encodes it, performs the reparametrization trick, and then decodes the latent code back to an image.
Although relatively simple, this forward pass can demonstrate several aspects worthy of unit testing.

### The output shape of your model

The very first piece of code we saw at the beginning of the article is a test nearly everybody does.
We also already know how this test looks like written as a unit test.
The only thing we have to do is to add the correct shape to test against.
For an autoencoder this is simply the same shape as the input:

```python
@torch.nograd()
def test_shape(self):
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
    inputs = torch.randn(4, 1, 32, 32)
    outputs = net(x)
    self.assertEqual(inputs.shape, outputs.shape)
```
            
Again, simple enough but helps to find some of the most annoying bugs.
An example would be forgetting to add the channel dimension when reshaping your model output back from its flat representation.

The last addition to our test is the `torch.nograd` decorator.
It tells PyTorch that gradients do not need to be recorded for this function and gives us a small speedup.
It may not be much for each test, but you never know how many you have to write.
Again, this is another piece of quotable unit test wisdom:

> Make your tests fast. Otherwise, no one will want to run them.

Unit tests should be run very frequently during development.
If your tests take a long time to run, you will be tempted to skip them.

### The moving of your model

Training deep neural networks on CPUs is incredibly slow most of the time.
This is why we use GPUs to accelerate it.
For this, all of our model parameters must reside on the GPU.
We should, therefore, assert that our model can be moved between the devices (CPU and multiple GPUs) correctly.

We can illustrate the problem on our example VAE with a common mistake.
Here you can see the `bottleneck` function that performs the reparametrization trick:

```python
def bottleneck(self, mu, log_sigma):
    noise = torch.randn(mu.shape)
    latent_code = log_sigma.exp() * noise + mu

    return latent_code
```

It takes the parameters of our latent prior, samples a noise tensor from a standard Gaussian, and uses the parameters to transform it.
This will run without a problem on CPU but fails when the model is moved to a GPU.
The bug is that the noise tensor is created in CPU memory as it is default and not moved to the device the model resides on.
A simple bug with a simple solution.
We just replace the offending line of code with `noise = torch.randn_like(mu)`.
This creates a noise tensor of the same shape and on the same device as the tensor `mu`.

The test that helps us catching such bugs early is straight forward:

```python
@torch.no_grad()
@unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
def test_device_moving(self):
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
    net_on_gpu = net.to('cuda:0')
    net_back_on_cpu = net_on_gpu.cpu()
    
    inputs = torch.randn(4, 1, 32, 32)

    torch.manual_seed(42)
    outputs_cpu = net(inputs)
    torch.manual_seed(42)
    outputs_gpu = net_on_gpu(inputs.to('cuda:0'))
    torch.manual_seed(42)
    outputs_back_on_cpu = net_back_on_cpu(inputs)

    self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))
    self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu))
```

We move our network from the CPU to a CPU and then back again just to make sure.
We now have three copies of our network (moving networks copies them) and make a forward pass with the same input tensor.
If the network was moved correctly, the forward pass should run without throwing an error and yield the same output each time.

To run this test we obviously need a GPU, but maybe we want to do some quick testing on a laptop.
The `unittest.skipUnless` decorator lets us skip the test if PyTorch does not detect a GPU.
This avoids cluttering the test results with a failed test.

You can also see that we fixed the random seed of PyTorch before each pass.
We have to do so because VAEs are non-deterministic and we would get different results otherwise.
This illustrates another important concept of unit testing deep learning code:

> Control the randomness in your tests.

How do you test a rare boundary condition of your model if you cannot make sure your model reaches it?
How do you make sure that the outputs of your model are deterministic?
How do you know if a failing test is only due to a random fluke or really due to a bug you introduced?
By setting the seed of your deep learning framework manually, you remove randomness from the equation.
Additionally, you should set CuDNN to deterministic mode, too.
This affects mainly convolutions but is a good idea anyway.
You can read all about making PyTorch deterministic in the [docs](https://pytorch.org/docs/stable/notes/randomness.html).

Be careful to fix the seeds of all frameworks you are using.
Numpy and the built-in Python random number generators have their own seeds and have to be set separately.
It is useful to have a utility function like this:

```python
def make_deterministic(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Numpy
    np.random.seed(seed)
    
    # Built-in Python
    random.seed(seed)
```

*Update: To truly fix all randomness, you have to set the [PYTHONHASHSEED](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED), too. This affects the hashing in dictionaries and sets. Thanks to /u/bpooqd on Reddit for the feedback.* 

### The sample independence of your model

In 99.99% of cases, you want to train your model with stochastic gradient descent in one form or another.
You feed your model a (mini-)batch of samples and calculate the mean loss over them.
Batching your training samples assumes that your model can process each sample as if you would have fed it individually.
In other words, the samples in your batch do not influence each other when processed by your model.
This assumption is a brittle one and can break with one misplaced reshape or aggregation over a wrong tensor dimension.

The following test checks sample independence by performing a forward and backward pass with respect to the inputs.
Before averaging the loss over the batch we multiply one loss with zero.
This will result in a gradient of zero given that our model upholds sample independence.
The only thing we have to assert is if only the masked samples gradient is zero:

```python
def test_batch_independence(self):
    inputs = torch.randn(4, 1, 32, 32)
    inputs.requires_grad = True
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)

    # Compute forward pass in eval mode to deactivate batch norm
    net.eval()
    outputs = net(inputs)
    net.train()

    # Mask loss for certain samples in batch
    batch_size = inputs[0].shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backward pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(inputs.grad):
        if i == mask_idx:
            self.assertTrue(torch.all(grad == 0).item())
        else:
            self.assertTrue(not torch.all(grad == 0))
```

If you read the code snipped precisely, you noticed that we set the model into evaluation mode.
This is because batch normalization violates our assumption above.
The running mean and standard deviation cross-contaminates the samples in our batch, which is why we deactivate their update through evaluation mode.
We can do this because our model behaves the same in training and evaluation mode.
If your model doesn't, you would have to find another way to deactivate it for testing.
An option would be to temporarily replace it with instance normalization.

The test function above is pretty general and can be copied as-is.
The exception is if your model takes more than one input.
Additional code for handling this would be necessary.

### The parameter updates of your model

The next test is concerned with gradients, too.
When your network architectures get more complex, e.g. Inception, it is easy to build dead sub-graphs.
A dead sub-graph is a part of your network containing learnable parameters that is either not used in the forward pass, the backward pass, or both.
This is as easy as building a network layer in the constructor and forgetting to apply it in the `forward` function.

Finding these dead sub-graphs can be done by running an optimization step and checking the gradients of your network parameters:

```python
def test_all_parameters_updated(self):
    net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    outputs = net(torch.randn(4, 1, 32, 32))
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param_name, param in self.net.named_parameters():
        if param.requires_grad:
            with self.subTest(name=param_name):
                self.assertIsNotNone(param.grad)
                self.assertNotEqual(0., torch.sum(param.grad ** 2))
```

All parameters of your model returned by the `parameters` function should have a gradient tensor after the optimization step.
Furthermore, it should not be zero for the loss we are using.
The test assumes that all parameters in your model require gradients.
Even parameters that are not supposed to be updated are accounted for by checking the `requires_grad` flag first.
If any parameter fails the test, the name of the sub-test will give you a hint where to look.

### Improving reusability

Now that we have written out all the tests of our model, we can analyze them as a whole.
We will notice that the tests have two things in common.
All tests begin by creating a model and defining an example input batch.
This level of redundancy has, as always, the potential for typos and inconsistencies.
Additionally, you do not want to update each test separately when changing the constructor of your model.

Fortunately, `unittest` gives us an easy solution to this problem, the `setUp` function.
This function is called before executing each test function in a `TestCase` and is normally empty.
By defining the model and inputs as member variables of the `TestCase` in `setUp` we can initialize the components of our tests all in one place.

```python
class TestVAE(unittest.TestCase):
    def setUp(self):
        self.net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
        self.test_input = torch.random(4, 1, 32, 32)

    ... # Test functions
```

Now we replace each occurrence of `net` and `inputs` with the respective member variable and we are done.
If you want to go a step further and use the same model instance for all tests, you can use `setUpClass`.
This function is called once when constructing the `TestCase`.
This is useful if construction is slow and you don't want to do it multiple times.

At this point, we have a neat system for testing our VAE model.
We can add tests easily and be sure to test the same version of our model each time.
But what happens if you want to introduce a new kind of VAE, one with convolutional layers.
It will run on the same data and should behave the same, too, so the same tests would apply.

Just copying the whole `TestCase` is obviously not the preferred solution, but by using `setUp` we are already on the right track.
We move all our test functions to a base class that leaves `setUp` as an abstract function.

```python
class AbstractTestVAE(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError

    ... # Test functions
```

Your IDE will nag that the class does not have the member variables `net` and `test_inputs` but Python does not care.
As long a child class adds them it will work.
For each model we want to test, we create a child class of this abstract class where we implement `setUp`.
Creating `TestCases` for multiple models or multiple configurations of the same model is then as easy as:

```python
class TestCNNVAE(AbstractTestVAE):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = model.CNNVAE(input_shape=(1, 32, 32), bottleneck_dim=16)

class TestMLPVAE(AbstractTestVAE):
    def setUp(self):
        self.test_inputs = torch.randn(4, 1, 32, 32)
        self.net = model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16)
```

Neat.
There is only one problem left.
The `unittest` package discovers and runs all children of `unittest.TestCase`.
As this includes our abstract base class that cannot be instantiated, we will always have a failed test in our suite.

The solution is presented by a popular design pattern.
By removing `TestCase` as the parent class of `AbstractTestVAE` it is not discovered anymore.
Instead, we let our concrete tests have two parents, `TestCase` and `AbstractTestVAE`.
The relationship between the abstract and concrete class is not anymore one of parent and child.
Instead, the concrete class uses shared functionality provided by the abstract class.
This pattern is called a **MixIn**.

```python
class AbstractTestVAE:
    ...

class TestCNNVAE(unittest.TestCase, AbstractTestVAE):
    ...

class TestMLPVAE(unittest.TestCase, AbstractTestVAE):
    ...
```

The order of the parent classes is important, as the method lookup is done from left to right.
This means that `TestCase` would override shared methods of `AbstractTestVAE`.
In our case, this is not a problem, but good to know anyway.

## Trainer

The last part of our learning system is the trainer class.
It brings all of your components (dataset, optimizer & model) together and uses them to train the model.
Additionally, it implements an evaluation routine that outputs the mean loss over the test data, too.
While training, all losses and metrics are written to a TensorBoard event file for visualization.

For this part, it is the hardest to write reusable tests, as it allows for the most freedom of implementation.
Some practitioners only use a plain code in a script file for training, some have it wrapped in a function and some others try to keep the more object-oriented style.
I will not judge which way you prefer.
The only thing I will say is, that a neatly encapsulated trainer class makes unit testing the most comfortable in my experience.

Nevertheless, we will find that some of the principles we learned earlier will be valid here, too.

### The loss of your trainer

Most of the time you will have the comfort of just picking a pre-implemented loss function from the `torch.nn` module.
But then again, your particular choice of loss function might not be implemented.
This may either be the case because the implementation is relatively straight forward, the function is too niche or too new.
In any case, if you implemented it yourself, you should test it, too.

Our example uses the Kulback-Leibler (KL) divergence as a part of the overall loss function, which is not present in PyTorch.
Our implementation looks like this:

```python
def _kl_divergence(log_sigma, mu):
    return 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma)
```

The function takes the logarithm of the standard deviation and the mean of a multivariate Gaussian and calculates the KL divergence to a standard Gaussian in closed form.

One method of checking this loss would be doing the calculation by hand and hard code them for comparison.
A better way would be to find a reference implementation in another package and check your code against its output.
Fortunately, the `scipy` package has an implementation of discrete KL divergence we can use:

```python
@torch.no_grad()
def test_kl_divergence(self):
    mu = np.random.randn(10) * 0.25  # means around 0.
    sigma = np.random.randn(10) * 0.1 + 1.  # stds around 1.
    standard_normal_samples = np.random.randn(100000, 10)
    transformed_normal_sample = standard_normal_samples * sigma + mu

    bins = 1000
    bin_range = [-2, 2]
    expected_kl_div = 0
    for i in range(10):
        standard_normal_dist, _ = np.histogram(standard_normal_samples[:, i], bins, bin_range)
        transformed_normal_dist, _ = np.histogram(transformed_normal_sample[:, i], bins, bin_range)
        expected_kl_div += scipy.stats.entropy(transformed_normal_dist, standard_normal_dist)

    actual_kl_div = self.vae_trainer._kl_divergence(torch.tensor(sigma).log(), torch.tensor(mu))

    self.assertAlmostEqual(expected_kl_div, actual_kl_div.numpy(), delta=0.05)
```

We first draw a big enough sample from the standard Gaussian and one from a Gaussian of different mean and standard deviation.
Then we use the `np.histogram` function to get a discrete approximation of the underlying PDFs.
Using these, we can use the `scipy.stats.entropy` function to get a KL divergence to compare against.
We use a relatively large `delta` for comparison, as `scipy.stats.entropy` is only an approximation.

You will have probably noticed, that we do not create a `Trainer` object, but use a member of the `TestCase`.
We simply used the same trick here as with the model tests and created it in the `setUp` function.
There we also fixed the seeds of both PyTorch and NumPy.
As we do not need any gradients here, we decorated the function with `@torch.no_grad`, too.

### The logging of your trainer

We use TensorBoard to log the losses and metrics of our training process.
For this, we want to make sure that all logs are written as expected.
One way to do this is to open up the event file after training and look for the right events.
Again, a valid option, but we will do it another way to have a look at an interesting functionality of the `unittest` package: the `mock`.

A `mock` lets you replace or wrap a function or an object with one that monitors how it is called.
We will replace the `add_scalar` function of our summary writer and make sure that all losses and metrics we care about were logged this way.

```python
def test_logging(self):
    with mock.patch.object(self.vae_trainer.summary, 'add_scalar') as add_scalar_mock:
        self.vae_trainer.train(1)

    expected_calls = [mock.call('train/recon_loss', mock.ANY, 0),
                      mock.call('train/kl_div_loss', mock.ANY, 0),
                      mock.call('train/loss', mock.ANY, 0),
                      mock.call('test/loss', mock.ANY, 0)]
    add_scalar_mock.assert_has_calls(expected_calls)
```

The `assert_has_calls` function matches a list of expected calls with the ones actually recorded.
The use of `mock.ANY` indicates that we do not care about the value of the scalar logged, as we don't know it anyway.

As we do not need to do an epoch over the whole dataset, we configured the training data in `setUp` to have only one batch.
This way, we can speed up our test significantly.

### The fitting of your trainer

The last question is the hardest to answer, too.
Does my training converge in the end?
To answer it conclusively, we would need to do a full training run with all our data and rate it.

As this is extremely time consuming, we will make do with a quicker method.
We will see if our training can overfit the model to a single batch of data.
The test function is quite simple:

```python
def test_overfit_on_one_batch(self):
    self.vae_trainer.train(500)
    self.assertGreaterEqual(30, self.vae_trainer.eval())
```

As already stated in the previous section, the `setUp` function creates a trainer with a dataset containing only one batch.
Additionally, we use the training data as test data, too.
This way we can get the loss for our training batch from the `eval` function and compare it to our expected loss.

For a classification problem, we would expect the loss to be zero when we overfitted completely.
The problem with a `VAE` is, that it is a non-deterministic generative model and a loss of zero is not realistic.
This is why we use an expected loss of 30, which equals a per-pixel error of 0.04.

This is by far the longest-running test, as it trains for 500 epochs.
In the end, it takes about 1.5 minutes on my laptop, which is still reasonable.
To speed it up further without dropping support for machines without a GPU, we can simply add this line to the `setUp`:

```python
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
```

This way we can take advantage of the GPU if we have one and train on the CPU if not.

### Last remarks

As we are logging, you may notice that the unit tests for the trainer tend to clutter up your folders with event files.
To avoid that, we use the `tempfile` package to create a temporary logging directory for the trainer to use.
After the test, we only have to delete it and its content again.
For this, we use the twin function of `setUp`, `tearDown`.
This function is called after each test function and cleaning up is made as simple as:

```python
def tearDown(self):
    shutil.rmtree(self.log_dir)
```

## Conclusion

We arrived at the end of the article.
Let's assess what we gained from the whole ordeal.

The test suite we wrote for our little example consists of 58 unit tests that take about 3.5 minutes to run completely (on my laptop from 2012).
For these 58 tests, we wrote only 20 functions.
All tests can run deterministically and independently from each other.
We can run additional tests if a GPU is present.
Most tests, i.e. the dataset and model tests, can be easily reused in other projects.
We were able to do all that by using:
* sub-tests to run one test for multiple configurations of our dataset
* the `setUp` and `tearDown` functions to consistently initialize and clean up our tests
* abstract test classes to test different implementations of our VAE
* the `torch.no_grad` decorator to disable gradient calculation when possible
* the `mock` module to check if functions are called correctly

In the end, I hope that I was able to convince at least someone to use unit tests for their deep learning project.
The companion repository of this article can be a starting point.
If you think I missed your favorite test or you can think of a way to make the tests even more reusable, feel free to contact me.
I'll try to update the post accordingly.

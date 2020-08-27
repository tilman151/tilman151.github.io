---
layout: post
title: "Make DL4J Readable Again"
categories: kotlin
---
    
A while ago, I stumbled upon an [article](https://blog.jetbrains.com/kotlin/2019/12/making-kotlin-ready-for-data-science/) about the language Kotlin and how to use it for Data Science.
I found it interesting, as some of Pythons quirks were starting to bother me and I wanted to try something new.
A day later, I had completed the Kotlin tutorials using [Kotlin Koans](https://www.jetbrains.com/help/education/learner-start-guide.html?section=Kotlin%20Koans&_ga=2.101592385.1724296010.1598524435-160366776.1590830721) in IntelliJIdea (which is an excelent way to get started with Kotlin).
Hungry to test out my new language skills, I looked around for a project idea.
As I am a deep learning engineer, naturally I had a look at what DL frameworks Kotlin has to offer and arrived at DL4J.
This is a Java framework, but as Kotlin is interoperable with Java, it can be used anyway.
I had a look at some examples of how to build a network and found this ([Source](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/kotlin/org/deeplearning4j/quickstartexamples/feedforward/mnist/MLPMnistTwoLayerExample.kt)):

```kotlin
val conf = NeuralNetConfiguration.Builder()
    .seed(rngSeed.toLong()) //include a random seed for reproducibility
     // use stochastic gradient descent as an optimization algorithm
    
    .activation(Activation.RELU)
    .weightInit(WeightInit.XAVIER)
    .updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
    .l2(rate * 0.005) // regularize learning model
    .list()
    .layer(DenseLayer.Builder() //create the first input layer.
        .nIn(numRows * numColumns)
        .nOut(500)
        .build())
    .layer(DenseLayer.Builder() //create the second input layer
        .nIn(500)
        .nOut(100)
        .build())
    .layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
    .build()

val model = MultiLayerNetwork(conf)
model.init()
```

Coming from Python and PyTorch, I just thought: "Damn, that's garbage!".
Maybe that's just my bias because I think Java code is ugly as hell, but Kotlin promised to reduce the verbosity that makes Java so hard to read, so I did not expect this.
At this point, my project began to take form.
What if I could use the nice Kotlin techniques from the tutorial to make network declarations in DL4J more readable.
I arrived at this:

```kotlin
val conf = sequential {
    seed(rngSeed.toLong()) //include a random seed for reproducibility
    // use stochastic gradient descent as an optimization algorithm
    
    activation(Activation.RELU)
    weightInit(WeightInit.XAVIER)
    updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
    l2(rate * 0.005) // regularize learning model
    layers {
       dense {
           nIn(numRows * numColumns)
           nOut(500)
       }
       dense {
           nIn(500)
           nOut(100)
       }
       output {
           lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
           activation(Activation.SOFTMAX)
           nIn(100)
           nOut(outputNum)
       }
    }
}

val model = MultiLayerNetwork(conf)
model.init()
```

This code snippet defines exactly the same network as the one before but omits all the syntactic clutter.
No more dots before the function calls because we don't have to hide a gigantic one-liner anymore.
No more calling `layer` each time when adding a new layer.
No more creating a `Builder` object for each layer.
No more calling `build` after each layer configuration.
In my opinion, this is a definite improvement over the Java version and much more readable.

How did I do this?
With the Domain Specific Language (DSL) feature of Kotlin and much less code than I expected.
The result is a small library named *klay* (Kotlin LAYers) that can be used to define neural networks in DL4J.

So without further ado, let's dive into what exactly DL4J does and how *klay* makes it easier.
You can find all the code shown here at [github.com/tilman151/klay](https://www.github.com/tilman151/klay).
    
## How DL4J Defines Neural Networks

The API of DL4J reminded me a lot of keras.
It follows a *define-and-run* scheme which means that you first build a computation graph and then run it with your inputs.
Coming from PyTorch, that uses a *define-by-run* scheme, this was a something I had to adjust to again.

Everything starts with the `NeuralNetConfiguration` class.
Instances of this class, hold all the information we need to build the computation graph of our network.
Creating a new `NeuralNetConfiguration` follows a builder pattern.
We first create a `NeuralNetConfiguration.Builder` that provides member functions to set the properties of our configuration.
Each of these functions, e.g. `updater` to set the weight updating algorithm, returns the `Builder` instance.
This makes it easy to chain calls.
When we are done, we call the `build` function to receive our configuration object:

```kotlin
val conf = NeuralNetConfiguration.Builder()
    .seed(rngSeed.toLong())
    .activation(Activation.RELU)
    .weightInit(WeightInit.XAVIER)
    .updater(Nesterovs(rate, 0.98))
    .build()
```

By calling a function like `activation`, we set the default value for all layers of the network.
The example above uses *ReLU* activation and *Xavier* initialization for all layers if not specified otherwise in the layer itself.

To add layers to the network, we call the `list` function of the `Builder` object.
This gives us a `ListBuilder` where we can add a layer by passing a layer configuration to its `layer` function:

```kotlin
val conf = NeuralNetConfiguration.Builder()
    .list()
    .layer(DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(500)
        .build())
    .build()
```

Layer configurations follow the same pattern as the overall network.
We create a `Builder` for the desired layer, call its configuration functions, and then `build`.

The last step is building a computation graph from our configuration.
This can be done by simply instantiating a `MultiLayerNetwork` object:

```kotlin
val model = MultiLayerNetwork(conf)
model.init()
```    

We can train our network, by feeding batches from a `DataSetIterator`, e.g. MNIST:

```kotlin
val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
model.fit(mnistTrain, numEpochs)
```

Now that we know how DL4J builds network, let's have a look at what Kotlin brings to the table.

* Kotlin DSL features
    * Extension functions
    * Higher order functions
    * Syntactic sugar
    * HTML example
    
## Domain Specific Languages in Kotlin
    
* Klay for defining networks
    * simple extension functions for NeuralNetConfiguration
    * abstraction to generic function (hopefully thanks to stackoverflow)
    * why is it better than JSON
    
    
    
* How to use it?
    * TODO
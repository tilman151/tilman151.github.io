---
layout: post
title: "Make DL4J Readable Again"
categories: kotlin
---
    
A while ago, I stumbled upon an [article](https://blog.jetbrains.com/kotlin/2019/12/making-kotlin-ready-for-data-science/) about the language Kotlin and how to use it for Data Science.
I found it interesting, as some of Python's quirks were starting to bother me and I wanted to try something new.
A day later, I had completed the Kotlin tutorials using [Kotlin Koans](https://www.jetbrains.com/help/education/learner-start-guide.html?section=Kotlin%20Koans&_ga=2.101592385.1724296010.1598524435-160366776.1590830721) in IntelliJ IDEA (which is an excellent way to get started with Kotlin).
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
With the Domain-Specific Language (DSL) feature of Kotlin and much less code than I expected.
The result is a small library named *Klay* (Kotlin LAYers) that can be used to define neural networks in DL4J.

So without further ado, let's dive into what exactly DL4J does and how *Klay* makes it easier.
You can find all the code shown here at [github.com/tilman151/klay](https://www.github.com/tilman151/klay).
    
## How DL4J Defines Neural Networks

The API of DL4J reminded me of Keras a lot.
It follows a *define-and-run* scheme which means that you first build a computation graph and then run it with your inputs.
Coming from PyTorch, which uses a *define-by-run* scheme, this was something I had to adjust to again.

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

Now that we know how DL4J builds networks, let's have a look at what Kotlin brings to the table.
    
## Domain-Specific Languages in Kotlin

Kotlin brings a bunch of nice features with it and describing them all would break the scope of this article.
Therefore, we will focus on the features that make defining Domain-Specific Languages (DSLs) in Kotlin so easy.
DSL is quite a buzzword (memorize it if you want to impress your superiors), so to be clear, I am referring to the definition on the Kotlin [website](https://kotlinlang.org/docs/reference/type-safe-builders.html):

> Type-safe builders allow creating Kotlin-based domain-specific languages (DSLs) suitable for building complex hierarchical data structures in a semi-declarative way. Some of the example use cases for the builders are:
>  
> * Generating markup with Kotlin code, such as HTML or XML
> * Programmatically laying out UI components: Anko
> * Configuring routes for a web server: Ktor.

Using this definition, DL4J, in a way, already has a DSL for defining network structures, albeit an ugly one.
Thus, we only need to wrap the existing language into a readable one.
Because Kotlin is a JVM language and interoperable with Java, I will use Java instead of Python as a reference point in the following paragraphs.

*Skip this part if you know everything about higher-order functions, Lambda expressions, and extension functions.*

### Extension Functions

The first concept we need is extension functions.
In Java, all of a class' member functions are defined inside it.
If we want to add a member function, we would need to create a child class:

```java
public class Base {
    protected int bar;

    public void foo(int bar) {
        this.bar = bar;
    }
}

// Adding a getter for bar

public class Child extends Base {
    public int getBar() {
        return super.bar;
    }
}
```

In Kotlin, we can instead use an extension function like this:

```kotlin
fun Base.getBar() {
    return this.bar
}
```

The `this` keyword refers to the instance of the `Base` class we called the function on.
We can write code exactly as if the extension function is a normal member of the class.
Therefore, we can omit the `this` keyword, too:

```kotlin
fun Base.getBar() {
    return bar
}
```

This way we can add functions and even properties to Java and Kotlin classes without inheriting or modifying them.
Overloading functions is possible, too.
The approach has an important advantage over sub-classing: we don't have to substitute our usage of the class `Base`.
All code that is working with `Base` at the moment will continue to do so and no casting is involved if we want to call our new function.
We only need to import the extension function beforehand.

### Higher-Order Functions and Lambdas

Higher-order functions are functions that take other functions as arguments.
A Java example is the `forEach` function that applies a function to each element of an `Iterable`.

```java
public class Arithmetic {
    public static inc(int n) {
        return ++n;
    }
}

// ...

ArrayList<int> numbers = new ArrayList<>(List.of(1, 2, 3)); 
ArrayList<int> incrementedNumbers = numbers.forEach(Arithmetic::inc);
```

We pass a reference of the static method `inc` in the class `Arithmetic` to `forEach` and receive an `ArrayList` of incremented numbers.
Now, this is a lot of code for defining the function `forEach`.
Fortunately we have Lambda expression at our hands to make things easier for us:

```java
ArrayList<int> incrementedNumbers = numbers.forEach({(n) -> ++n});
```

We simply pass an anonymous function in the form of a Lambda expression to `forEach` and don't have to bother with defining it elsewhere.

In Kotlin the process of using higher-order functions and Lambda expression is a little more streamlined.
The `forEach` equivalent is called `map`, so our example looks like this:

```kotlin
val numbers = listOf(1, 2, 3)
val increasedNumbers = numbers.map({n -> ++n})
```

In fact, Kotlin even lets us omit the parenthesis if the last argument of a function is a Lambda expression:

```kotlin
val increasedNumbers = numbers.map {n -> ++n}
```

This way we got rid of all the syntactic clutter and receive code that is much more readable.

But, this is not where it ends.
Even extension functions from the previous section can be Lambda expressions.
This way, we can call members of an object inside the Lambda Expression with the `this` statement:

```kotlin
data class Person(val name: String)
val persons = listOf(Person("Foo"), Person("Bar"))

val getName: Person.() -> String = {this.name}
val name = persons.map(getName)
```

We first assigned the Lambda to a variable to declare the function type.
Let's have a closer look.
A Kotlin function type follows the scheme `argument types -> output type`.
`Person.()`  means that the Lambda takes an instance of the class `Person` which is called the *receiver*.
The function returns a `String` which is signalized by the right-hand side of the arrow.

Anonymous extension functions are especially helpful for initializing objects with the higher-order function `apply`:

```kotlin
data class Person(val name: String, var age: Int = 0, var city: String = "")

val p = Person("Foo")
p.apply {
    age = 25
    city = "Bar"
}
```

The return type of the Lambda, that is passed to `apply`, is `Unit` which means it returns nothing (similar to Java's `void`).
Alternatively, we could use the `run` function, which assumes that the Lambda returns a result:

```kotlin
val p = Person("Foo")
print(p.run {age = 25})
```

With all that theory in our mind, let us see how all this leads to our neural network DSL.
    
## Klay for Defining Neural Networks

Our little "library", *Klay*, makes heavy use of higher-order functions, extension functions, and Lambdas with receivers.
It is not much different from the example in the [official Kotlin docs](https://kotlinlang.org/docs/reference/type-safe-builders.html) that builds HTML.
Let's have a look again at our DL4J example:

```kotlin
val conf = NeuralNetConfiguration.Builder()
    .seed(rngSeed.toLong())
    .activation(Activation.RELU)
    .weightInit(WeightInit.XAVIER)
    .updater(Nesterovs(rate, 0.98))
    .l2(rate * 0.005)
    .list()
    .layer(DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(500)
        .build())
    .layer(DenseLayer.Builder()
        .nIn(500)
        .nOut(100)
        .build())
    .layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
    .build()
```

As you probably remember, the first step of building a neural network in DL4J is creating a `NeuralNetConfiguration.Builder`.
Using our knowledge about Lambdas with receivers, we can write the following function:

```kotlin
fun sequential(init: NeuralNetConfiguration.Builder.() -> NeuralNetConfiguration.ListBuilder): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().run(init).build()
}
```

This function takes a Lambda, `init`, with a `NeuralNetConfiguration.Builder` receiver.
The receiver object is created inside the `sequential` function.
We call the higher-order function `run` on our receiver object and get a `NeuralNetConfiguration.ListBuilder` object which we then build into a `MultiLayerConfiguration`.
Using this function would look like this:

```kotlin
val conf = sequential( {
    this.seed(rngSeed.toLong())
    this.activation(Activation.RELU)
    this.weightInit(WeightInit.XAVIER)
    this.updater(Nesterovs(rate, 0.98))
    this.l2(rate * 0.005)
    this.list()
    .layer(DenseLayer.Builder()
        .nIn(numRows * numColumns)
        .nOut(500)
        .build())
    .layer(DenseLayer.Builder()
        .nIn(500)
        .nOut(100)
        .build())
    .layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
} )
```

Inside the `init` Lambda, we have access to all member functions of the `Builder` to configure defaults.
Calling the `list` function, we can add layers the conventional way.
`list` and each call of `layer` return a `NeuralNetConfiguration.ListBuilder` object.
As `layer` is the last function call in the Lambda expression, its resulting `Builder` is returned to `sequential` to be built there.

Next, we want to get rid of the call to `list`.
We will define an extension function of `NeuralNetConfiguration.Builder` like this:

```kotlin
fun NeuralNetConfiguration.Builder.layers(init: NeuralNetConfiguration.ListBuilder.() -> Unit): NeuralNetConfiguration.ListBuilder {
    return this.list().apply(init)
}
```

Inside the function, we call `list` and use the higher-order function `apply` to execute our Lambda expression `init` on it.
This simplifies our example like this:

```kotlin
val conf = sequential( {
    this.seed(rngSeed.toLong())
    this.activation(Activation.RELU)
    this.weightInit(WeightInit.XAVIER)
    this.updater(Nesterovs(rate, 0.98))
    this.l2(rate * 0.005)
    this.layers( {
        layer(DenseLayer.Builder()
            .nIn(numRows * numColumns)
            .nOut(500)
            .build())
        layer(DenseLayer.Builder()
            .nIn(500)
            .nOut(100)
            .build())
        layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(100)
            .nOut(outputNum)
            .build())
    } )
} )
```

`apply` returns the `ListBuilder` created by `list`.
Therefore, our function can be used as a drop-in replacement.

The last offending code is the call of the `layer` function for adding a single layer to the network.
We can simply outsource the call, and the creation of the layer's `Builder` to an extension function of the `ListBuilder`.
For the `DenseLayer` and `OutputLayer`, the functions looks like this:

```kotlin
fun NeuralNetConfiguration.ListBuilder.dense(init: DenseLayer.Builder.() -> Unit) {
    this.layer(DenseLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.output(init: OutputLayer.Builder.() -> Unit) {
    this.layer(OutputLayer.Builder().apply(init).build())
}
```

The Lambda expression with the layer's `Builder` as the receiver lets us again conveniently configure the layer.
Our example has is now completely transformed:

```kotlin
val conf = sequential( {
    this.seed(rngSeed.toLong())
    this.activation(Activation.RELU)
    this.weightInit(WeightInit.XAVIER)
    this.updater(Nesterovs(rate, 0.98))
    this.l2(rate * 0.005)
    this.layers( {
       this.dense( {
           this.nIn(numRows * numColumns)
           this.nOut(500)
       } )
       this.dense( {
           this.nIn(500)
           this.nOut(100)
       } )
       this.output( {
           this.lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
           this.activation(Activation.SOFTMAX)
           this.nIn(100)
           this.nOut(outputNum)
       } )
    } )
} )
```
    
But wait, this isn't even its final form.
Now we have to apply all of Kotlin's syntactic sugar, i.e. removing `this` and the parenthesis.
Et voila:

```kotlin
val conf = sequential {
    seed(rngSeed.toLong())
    activation(Activation.RELU)
    weightInit(WeightInit.XAVIER)
    updater(Nesterovs(rate, 0.98))
    l2(rate * 0.005)
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
```

Another point where *Klay* shines is procedurally generating network layer declarations.
A common example would be to add several dense layers with an increasing number of units to our network with a loop.
In standard DL4J it would look like this:

```kotlin
val units = listOf(100, 200, 300, 400)

val unfinished = NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .updater(Nesterovs(rate, 0.98))
    .list()
    .layer(DenseLayer.Builder()
             .nIn(numRows * numColumns)
             .nOut(units[0])
             .build())

for (u in units.zipWithNext()) {
    unfinished.layer(DenseLayer.Builder()
        .nIn(u.first)
        .nOut(u.second)
        .build())
}

val conf = unfinished.layer(OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(100)
        .nOut(outputNum)
        .build())
    .build()
```

As we can see, we have to break our declaration flow to insert the loop.
This makes the code much uglier than before.
Let's see the *Klay* declaration on the other hand:

```kotlin
val units = listOf(100, 200, 300, 400)
val config = sequential {
             activation(Activation.RELU)
             updater(Nesterovs(rate, 0.98))
             layers {
                 dense {
                     nIn(numRows * numColumns)
                     nOut(units[0])
                 }
                 for (u in units.zipWithNext()) {
                     dense {
                         nIn(u.first)
                         nOut(u.second)
                     }
                 }
                 output {
                     lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                     activation(Activation.SOFTMAX)
                     nIn(units.last())
                     nOut(outputNum)
                 }
             }
         }
```

The loop integrates nicely with the rest of the declaration, and we do not break the flow.
The point is, this is not some gimmick I added in the background.
This is out of the box functionality in Kotlin.
We can use the full power of the programming language while staying true to our DSL.

### Is *Klay* ready to use?

Yes, it is!
Even though it took so few lines of code that it does not really warrant calling it a library, you can find it [here](https://www.github.com/tilman151/klay).
All code is provided as-is, yadda, yadda, yadda.

Currently, the library supports all operations needed to recreate the quickstart examples of DL4J.
They are included in the project repository.
Converting them from Java to Kotlin was, fortunately, extremely easy thanks to IntelliJ IDEA's automatic conversion function.
If you are missing something and want to help out, feel free to send me a pull request.

### Conclusion

I liked working with Kotlin for a change and maybe I will expand *Klay*'s coverage of DL4J later on.
On the other hand, I noticed that I am not as fluent in Kotlin as in Python which let me struggle a bit with this project.

If you are skilled in Java or Kotlin and know your way around generic functions, you may want to check out my question on StackOverflow related to this article
I was not able to make the layer building functions generic and would appreciate some input.
You would really help me out there.

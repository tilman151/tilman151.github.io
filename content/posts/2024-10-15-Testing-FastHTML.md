---
layout: post
title: "Testing FastHTML Dashboards"
slug: testing-fasthtml
date: 2024-10-15
tags: ["fasthtml", "testing", "cleancode"]
mathjax: false
---

Building dashboards to visualize data or the results of experiments is the bread and butter of data people (read: data scientist, engineers, analysts, etc.).
Often, these dashboards are hacked together in record time to meet a presentation deadline.
Now imagine this: you built a dashboard for showcasing your latest model to your team.
Instead of your go-to tool, [Streamlit](https://streamlit.io/), you decided to try out [FastHTML](https://fastht.ml), a shiny new framework that promises better handling and scalability if your dashboard ever needs to go bigger.
Your team lead is so impressed with your model that they want to show it to the whole company.
That is your chance to shine!
With FastHTML, you don't have to worry about scaling to a bigger audience.
But wait: are you sure your dashboard is really working as expected?
How can you be certain nothing fails if the CEO happens to use it?
Normally, you would go for automated testing, but after scouring the FastHTML documentation on how to do it, you found nothing.

Thank goodness you found this post!
As I was in a similar situation and came up empty-handed, I needed to figure it out myself.
This post is intended for data people who want to test their FastHTML dashboards.
If you're a web developer, you might find this post too basic, but feel free to give feedback on how to improve the testing process described here.
But first, we need to go over some basics of automated testing.

## Types of Automated Tests

There are several types of automated tests and even more definitions to each.
We will focus on three types here: unit tests, integration tests, and end-to-end tests.
There are many a StackOverflow post discussing where to draw the line between these types.
I was always of the opinion that one should be pragmatic about it, as long as the following is true:

> Unit tests are extremely fast, integration tests are reasonably fast, and end-to-end tests don't take forever to run.

Each test of any type needs to be independent of the others, so you can run them in any order.
Furthermore, your test suite should have a lot of unit tests, fewer integration tests, and even fewer end-to-end tests.
We will explore why in the following sections.

### Unit Tests

The term unit test is often synonymous with automatic test.
To be precise, unit tests check the functionality of the smallest code part, usually a function or a method.
They need next to no setup, don't interact with code that is not under test, and run lightning-fast.
Try to check all edge cases, so you don't have to do it with the other test types.
If you develop test-driven, these tests would be written in advance to define the functionality of your code.
But, as only a few people have the discipline to do so, unit tests are often written after the fact, too.
Because they're so fast, you can, in theory, run them every time you save your code.
Unit tests are the most numerous in your suite, because the smallest code parts are the most numerous in your codebase.

### Integration Tests

Integrations tests check if the parts of your code integrate well with each other.
In our case, this could be the interaction between the dashboard endpoints and the database code.
They need some setup to ensure that the parts under test are connected as they would be in production.
Integration tests are a little slower than unit tests, but still fast enough to run them when you push to remote.
They're fewer than unit tests, because there should be fewer combinations in your codebase than individual code parts.
How to make sure this is the case is a topic for another post.

### End-to-End Tests

End-to-end tests check if a user interaction can be successfully executed.
They're often mapped to user stories like "*As a user, I want to log in to the dashboard and expect to see the start page afterward.*"
These tests are the slowest of the bunch, because they need to set up the whole application before running.
End-to-end tests are the fewest in your suite, because they're the most complex to write and maintain.
They're also the most brittle, tending to break over minor changes like moving or renaming a button.

## Our Example Dashboard

We will explore our test strategy with a [simple example dashboard](https://github.com/tilman151/testing-fasthtml).
It shows a text input field and a button labeled "*Ask.*"
When the user enters a question in the input field and clicks the button, the dashboard disables the input field and displays the LLM-generated answer below with a button to reset the dashboard.
The dashboard also features a button that displays the last ten questions asked which are stored in a Sqlite database.
This is as simple as it gets, but still provides ample opportunity for testing.
And of course, we make our lives more complicated than necessary by including LLM functionality.
It is 2024, after all.

We use [uv](https://github.com/astral-sh/uv) for managing dependencies, [SQLAlchemy](https://www.sqlalchemy.org/) for interacting with the database, [pytest](https://docs.pytest.org/en/stable/) as our testing framework, and [Poe the Poet](https://poethepoet.natn.io/) as our task runner.
For instructions on how to set up the project, refer to the [README](https://github.com/tilman151/testing-fasthtml/blob/main/README.md).

### Unit Tests for the Database

First, we want to test the code that reads and writes the questions from and to the database.
I already hear the first people scoffing, "*But that's not a unit test, that's an integration test!*"
As I said, let us be pragmatic about it.
Testing database interaction code without a database is nearly useless.
Additionally, we're using Sqlite, which is an embedded database and part of the Python standard library.
You would not consider testing code that uses NumPy an integration test because it includes a C library, would you?

With that out of the way, there is some setup we need to do.
We need an empty database in working order before each test.
Remember, automated tests need to be independent, so the remains of the last test need to be removed from the database.
We use the following [fixtures](https://docs.pytest.org/en/7.1.x/explanation/fixtures.html) for this:

```python
@pytest.fixture
def clean_database(tmp_database):
    yield tmp_database

    engine = sa.create_engine(tmp_database)
    with engine.connect() as connection:
        database.metadata.drop_all(connection)
        connection.commit()
    _init_database(engine)
    engine.dispose()
    
    
@pytest.fixture(scope="module")
def tmp_database(tmp_database_path):
    engine = sa.create_engine(tmp_database_path)
    _init_database(engine)
    engine.dispose()

    return tmp_database_path


@pytest.fixture(scope="module")
def tmp_database_path(tmp_path_factory, request):
    path = tmp_path_factory.mktemp(request.module.__name__ + "_data", numbered=True)
    path = path / "inventory.db"

    return f"sqlite:///{path}"
```

Let us start from the bottom:

1. The `tmp_database_path` fixture creates a connection string for our database, pointing to a temporary directory.
1. The `tmp_database` fixture takes the connection string, creates the database, and initializes it with the `_init_database` function, which is omitted for brevity.
1. The `clean_database` fixture takes the connection string to the initialized database, yields it to the test function and cleans up afterward by reinitializing the database.

But why so complicated?
Why not do it in one fixture?
The answer is saving time.
If we create a new database for each test, we would waste some time, slowing down our unit tests.
Remember, they need to be extremely fast.
By scoping `tmp_database_path` and `tmp_database` to module level, pytest runs these fixtures only once per test module.
This way, the database is only set up once and `clean_database` merely reinitializes it.

The last piece of setup concerns the way we connect to the database itself.
The `database` module of the dashboard maintains a global instance of an SQLAlchemy engine that controls the database connections.
This way, we can spin up connections to the database quickly without needing to establish the engine first.
Unfortunately, this also introduces a global state into our app.
Global state is always complicated for automated testing, as it breaks test independence.
The following fixture solves this problem for us:

```python
@pytest.fixture
def database_module():
    from app import database

    importlib.reload(database)

    return database
```

This fixture supplies a freshly imported database module for each of our tests, effectively resetting the global state.

Now for the unit tests themselves.
We will exemplify the process on the `append_to_history` function.
Here is the test code:

```python
def test_append_to_history(database_module, clean_database):
    database_module.connect(clean_database)
    database_module.append_to_history("question0", "answer0")
    database_module.append_to_history("question1", "answer1")

    with database_module._get_connection() as conn:
        result = conn.execute(database_module.history.select()).fetchall()

    assert len(result) == 2
    for i, row in enumerate(result):
        assert row.question == f"question{i}"
        assert row.answer == f"answer{i}"
        assert isinstance(row.created_at, datetime.datetime)
    assert result[0].created_at <= result[1].created_at
```

The function under test takes a question and the corresponding answer and inserts it into the history table.
The database is expected to supply the current timestamp for the `created_at` column.
With our test function, we first connect to the database, insert two lines of data, and retrieve them again.
We check if the table really contains only these two entries and if the `created_at` column was filled correctly.
Two fundamental unit testing practices can be observed here:

1. **Fail for the right reason:** unit tests are not only intended to check correctness, but also to hint in the direction of a bug. Whether there is only one row in the table, the rows are swapped or the `created_at` column is unfilled, each broken assumption fails on the appropriate `assert`. This makes debugging much easier.
1. **Test only one unit of code:** even though, there is a function to retrieve questions from the database in our module, we don't use it here. If we use it, we would introduce another piece of code under test, making this an integration test.

The `test_database` module contains more tests than this one.
You can find the complete module [here](https://github.com/tilman151/testing-fasthtml/blob/main/tests/test_database.py).
The tests can be run in the activated virtual environment with the following command:

```bash
poe unit
```

### Integration Tests for the Routes

Next, we will test if sending requests to the routes of our dashboard works as expected.
We will use the `starlette.testclient.TestClient` for this.
It allows us to simulate HTTP requests to our dashboard without spinning up a server.
The following fixture sets up the test client:

```python
@pytest.fixture
def client(clean_database, monkeypatch):
    monkeypatch.setenv("APP_DATABASE_URL", clean_database)

    from app.main import app

    client = TestClient(app)
    client.headers["HX-Request"] = "true"
    database.connect(clean_database)

    yield client

    database.disconnect()
```

The `monkeypatch` fixture is used to temporarily set the database URL environment variable.
This way, we can connect the client to the database we set up in the `clean_database` fixture.
As you can see, we're using our `database.connect` and `database.disconnect` functions.
This is possible because integration tests are only executed once all unit tests have passed.
We can, therefore, assume that these functions are working as intended.
This demonstrates the importance of the test hierarchy, as we can now use our high-level functions in the integration tests, making them much more concise.
The `client` fixture also sets the `HX-Request` header to `true`.
This tells FastHTML that the request is coming from HTMX and the result should be an HTML fragment.
Otherwise, FastHTML would wrap the result in additional HTML tags, which would complicate the checks.

Let us first look at a test for the `/ask` route.
It expects a POST request with a JSON body containing a question.
The returned fragment is intended to be swapped with the question input field and submit button.

```python
def test_ask_mocked_answer(client, mocker):
    mock_generate = mocker.patch("ollama.generate")
    mock_generate.return_value = {"response": "answer0"}

    response = client.post("/ask", data={"question": "question0"})

    assert response.status_code == 200
    html = htmlmin.minify(response.text, remove_empty_space=True)
    assert '<input name="question" value="question0" disabled>' in html
    assert '<input name="answer" value="answer0" disabled>' in html
```

The most important part of this test is the `mocker` fixture.
We use it to replace the `ollama.generate` function with a dummy function that always returns the same answer.
Setting up the LLM is far too costly for integration tests, which is why we mock it.
Additionally, the response of an LLM is not deterministic, making it hard to test.

The test now sends a POST request to the `/ask` route with the question.
The response is then checked for the appropriate status code and the expected HTML fragment.
We use the `htmlmin` package to remove unnecessary whitespaces and linebreaks from the response text.
This makes the checks more robust against minor changes in the HTML structure.

Next, we will look at a test for the `/history` route:

```python
def test_history_empty(client):
    response = client.get("/history")

    assert response.status_code == 200
    html = htmlmin.minify(response.text, remove_empty_space=True)
    assert re.search("<table>.+?</table>", html)
    assert "<thead><tr><th>Question</th><th>Asked at</th></tr></thead>" in html
    assert "<tbody hx-target=#question-form></tbody>" in html
```

This test sends a GET request to retrieve the table of the last ten questions.
In this case, there are no questions in the database, so the table should be empty.
We check if the returned fragment is still a table with the correct headers and an empty body.
To do so, we use a regex to check for a non-empty table tag.

You can find the `test_app` module with all integration tests [here](https://github.com/tilman151/testing-fasthtml/blob/main/tests/test_app.py).
They are marked with the `integration` marker, so you can run them with the following command:

```bash
poe integration
```

Overall, I am still unsatisfied with the integration tests.
Using `in` operators and regexes to check the HTML fragments is not very elegant.
The alternative of checking against the whole expected HTML fails to properly convey the intent of the test.
If I ever come up with a better solution, I will update the post and the repository.

### End-to-End Tests for the Dashboard

End-to-end tests run the same way a user would experience the dashboard: through the browser.
Therefore, we need a way to automate browser interactions, which is the Python API of [Playwright](https://playwright.dev/python/) in our case.
As the browser needs to communicate with the server, we need to set it up first.
The following fixture will do so:

```python
@pytest.fixture
def server(clean_database, setup_server):
    return setup_server


@pytest.fixture(scope="module")
def setup_server(tmp_database):
    process = multiprocessing.Process(
        target=_setup_server,
        args=("app.main:app",),
        kwargs={"host": "localhost", "port": 5001, "database_url": tmp_database},
        daemon=True,
    )
    process.start()
    for i in range(50):  # 5-second timeout
        sleep(0.1)
        try:
            requests.get("http://localhost:5001")
        except requests.ConnectionError:
            continue
        else:
            break
    else:
        raise TimeoutError("Server did not start in time")

    yield process

    process.terminate()
```

The `setup_server` fixture starts the dashboard server in a separate process.
The `_setup_server` function is executed in the new process and sets the necessary environment variable for the database URL.
After the process is started, we wait for the server to respond to a request with a timeout of five seconds.
This ensures that the server is actually running before we continue with the tests.
As before, we use the `clean_database` fixture to set up the database for the server.
Instead of restarting the server for each test, we only start it once per test module.
The `clean_database` fixture cleans up the database while the server keeps running.
This saves valuable test time and still provides an independent environment for each test.
This only works as long as the dashboard itself is stateless.
If the dashboard maintains any kind of global state, we would need to restart the server for each test.

The following test checks if we receive an answer from the LLM when we ask a question:

```python
@pytest.mark.e2e
def test_ask_question(server, page):
    page.goto("http://localhost:5001")
    page.get_by_label("Question").type("How are you doing?")
    page.get_by_text("Ask").click()
    page.wait_for_selector("input[name='answer']")

    assert page.get_by_label("Question").input_value() == "How are you doing?"
    assert page.get_by_label("Answer").input_value()
    assert page.get_by_text("Reset").is_visible()
```

The `page` fixture is provided by Playwright and represents the browser page.
We use it to navigate to our dashboard and interact with it as a user would.
We locate the question input field by its label, type a question, and click the button labeled "Ask."
We then wait for the answer field to appear.
Finally, we check if the question field still contains the question, the answer field is filled, and the reset button is visible.

This test is very brittle.
It relies on the labels of the HTML elements to locate them and will time out if the LLM takes too long to respond.
This is typical for end-to-end tests, which is why they may need to be updated more frequently.
If the labels change, the test will fail, even though the functionality is still working.
This brittleness is why end-to-end tests are the fewest in your test suite and are only run after the unit and integration tests pass.

On the other hand, these tests most closely align with the user experience.
If the test fails due to a label change, the user would also be confused, and may need to be notified of this change.
If the test times out due to the LLM, the user may also find the delay too long.
This way, end-to-end tests provide a distinct kind of feedback, compared to the other test types.

You can find the `test_app` module with all end-to-end tests [here](https://github.com/tilman151/testing-fasthtml/blob/main/tests/test_app.py).
They are marked with the `e2e` marker, so you can run them with the following command:

```bash
poe setup_e2e  # installs chromium, ollama, and pulls the LLM
poe e2e
```

## Conclusion

Automated testing is essential for developing software.
Testing right is hard.
Nevertheless, I hope this post is helpful for all the anxious data people out there who want to make sure their FastHTML dashboards run well.
Again, you can find the whole code for this post [here](https://github.com/tilman151/testing-fasthtml).

If you learned something, share this post with your peers.
If not, please get back to me on how to improve the outlined testing process.
I'm still on an ongoing testing journey and appreciate feedback, especially on the integration tests.

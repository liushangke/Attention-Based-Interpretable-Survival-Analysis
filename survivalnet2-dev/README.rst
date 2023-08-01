.. image:: https://dl.circleci.com/status-badge/img/gh/PathologyDataScience/survivalnet2/tree/dev.svg?style=svg&circle-token=bdadea364863bd100c6591164f222fe0495e39ae
        :target: https://dl.circleci.com/status-badge/redirect/gh/PathologyDataScience/survivalnet2/tree/dev
        
.. image:: https://codecov.io/gh/PathologyDataScience/survivalnet2/branch/dev/graph/badge.svg?token=YK9H2OA7QO 
 :target: https://codecov.io/gh/PathologyDataScience/survivalnet2

================================================
SurvivalNet
================================================

`SurvivalNet`_ is a Python package for building time-to-event machine learning models.
It provides losses, performance metrics, specialty Keras models, visualization, and other utility functions that can be used with the Keras interface to TensorFlow2.

The /examples folder contains Jupyter notebooks that illustrate how to use various package features.

Developer notes
---------------
SurvivalNet follows a community development model and we welcome engagement with users to enhance SurvivalNet functionality and quality. This section describes our development practices and standards that can help you engage with the team.

Contributing
~~~~~~~~~~~~~~~~~~~~~~~~
In your python environment, install the requirements and pre-commit hooks::

    pip install -r requirements.txt
    pre-commit install

(Optional) Run lints before commiting::

    pre-commit run -a


Issues and pull-requests
~~~~~~~~~~~~~~~~~~~~~~~~
We follow a `git-flow <https://nvie.com/posts/a-successful-git-branching-model/>`_ branching model. New features are branched and merged into a `dev <https://github.com/PathologyDataScience/survivalnet2>`_ and are periodically merged into master through releases.

To implement a new feature:

    1. Create a new Issue with a short but descriptive title.

    2. Assign this issue to yourself. This automatically creates a new branch with the issue # followed by the issue title.

    3. Commit changes to your branch, including implementing tests (see our testing_ guidelines).
    
    4. Submit a pull request and `link the issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`_ there.

Testing
~~~~~~~
We use `pytest <https://docs.pytest.org/en>`_ for package testing.

Testing guidelines and standardization:

    1. We follow the `test outside application code <https://docs.pytest.org/en/6.2.x/goodpractices.html#choosing-a-test-layout-import-rules>`_ organization that places all tests in the `\tests <https://github.com/PathologyDataScience/survivalnet2/tree/dev/tests>`_ folder.

    2. Create one test file per module. Each test case / function pair should have a separate function in the test file to enable quick pinpointing of test failures.  Naming conventions are 'test_module_function_case', where case is a descriptive name. 

            Example: A test of the cox loss to handle NaNs would be in /tests/test_losses.py, with a function name test_cox_loss_accept_nan() or something similar.

    3. For testing float outputs, use `Numpy Test Support <https://numpy.org/doc/stable/reference/routines.testing.html>`_ or analogous `tf.debugging <https://www.tensorflow.org/api_docs/python/tf/debugging>`_ functions. Passing thresholds are set on a function-by-function basis using your best judgment.

    4. Store large testing data in .csv format in `/tests/test_data <https://github.com/PathologyDataScience/survivalnet2/tree/dev/tests/test_data>`_. This improves cross-platform test functionality. Re-use testing data where possible and avoid cut/paste of test data over multiple test functions.

Example: access data within a testing function (for example with our structure below)

.. code-block:: python

    .
    └───tests
    |       └───test_data
    |                   └───test_data_cox_loss.csv
    |       └───test_data.py
    |       └───test_estimators.py
    |       ...

The python code to read data from test_data is shown as below:

.. code-block:: python

    import pandas as pd

    # Absolute path to package root
    ROOT_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    def test_load_csv():
        filepath = os.path.join(ROOT_TEST_DIR, "test_data", "test_data_cox_loss.csv")
        df = pd.read_csv(filepath, header=None)  # Load csv for testing
        
    
Continuous Integration and Code Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SurvivalNet uses `CircleCI <https://circleci.com/>`_ continuous integration. CircleCI checks run for each commit on a pull request and must pass for a pull request to merge. Code coverage of testing is evaluated using `Coverage <https://pypi.org/project/coverage/>`_ with reporting from `CodeCov <https://about.codecov.io/>`_. Passing criteria for coverage are defined in the `CodeCov yaml <https://github.com/PathologyDataScience/survivalnet2/blob/dev/codecov.yml>`_.

Losses
~~~~~~
Loss functions must support masking of missing labels. See cox.py from the losses subpackage for an example.

Bibliography
------------

1. Yousefi S, Song C, Nauata N, Cooper L. Learning genomic representations to predict clinical outcomes in cancer. ICLR Workshop, San Juan, PR, May 2, 2016

2. Yousefi S, Amrollahi F, Amgad M, Dong C, Lewis JE, Song C, Gutman DA, Halani SH, Vega JE, Brat DJ, Cooper LA. Predicting clinical outcomes from large scale cancer genomic profiles with deep survival models. Scientific reports. 2017 Sep 15;7(1):1-1.

3. Halani SH, Yousefi S, Vega JV, Rossi MR, Zhao Z, Amrollahi F, Holder CA, Baxter-Stoltzfus A, Eschbacher J, Griffith B, Olson JJ. Multi-faceted computational assessment of risk and progression in oligodendroglioma implicates NOTCH and PI3K pathways. NPJ precision oncology. 2018 Nov 6;2(1):1-9.

4. Yousefi S, Shaban A, Amgad M, Cooper L. Learning Cancer Outcomes from Heterogeneous Genomic Data Sources: An Adversarial Multi-task Learning Approach, ICML Workshop on Adaptive & Multitask Learning, Long Beach, CA, June 15, 2019

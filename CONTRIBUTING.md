# Contributing to *ghedt*

`ghedt` welcomes and appreciates bug reports, suggestions and contributions from 
everyone. 

This guide describes how to report bugs, suggest new features and contribute 
code to `ghedt`. 

## Reporting bugs

Bugs are reported on the [issue tracker][#issue_tracker].

Follow these steps when submitting a bug report:

1. **Make sure the bug has not been already reported.** Run a quick search
through the [issue tracker][#issue_tracker].
If an open issue is related to your problem, consider adding your input to that
issue. If you find a closed issue related to your problem, open an new issue
and link to the closed issue.
2. **Use a short and descriptive title** when creating a new issue.
3. **Provide detailed steps to reproduce the problem.** Explain, in details,
how to reproduce the problem and what should be the **expected result.** When
possible, include a simple code snippet that isolates and reproduces the
problem.

After submitting a bug report, if you wish to contribute code to fix the
problem, follow the steps outlined in the contribution workflow.

## Contribution workflow

This section outlines the steps for contributing to *pygfunction*.

1. **Open a new [issue][#issue_tracker].**
2. **Use a short and descriptive title.** When proposing an enhancement,
describe in details what the enhancement would entail. If you plan to implement
the enhancement yourself, provide a step-by-step plan for the implementation.
3. **Explain how the enhancement benefits _pygfunction_.**
4. **Create (checkout) a new branch from the master.** The branch name should
follow the naming convention: `issue#_shortDescription`. For example:
issue1_loadAggregation.
5. Implement unit tests for new features. If necessary, update already
implement tests to cover the new features.
6. Before submitting a [pull request][#pull_request], **merge the master to your 
branch.**
7. Once the branch is merged, **delete the branch and close the issue.**

## Managing branches

This section describes various features regarding [git branches][#git_branches]. 

### Create a branch

Prior to making changes to the code, a branch should be created. The following
shows examples for how to create a branch from the command line. 
```angular2html
git checkout -b $branchName &&
git push -u origin $branchName &&
git branch --set-upstream-to=origin/$branchName $branchName &&
```
To list all the branches, run the following command. 
```angular2html
git branch -a
```

## Styleguide

`ghedt` follows the [PEP8 style guide][#pep]. Docstrings are written 
following the [numpydoc format][#numpydoc], see also an example [here][#sphinx].

## References

This contributing outline was originally taken from [pygfunction](https://github.com/MassimoCimmino/pygfunction/blob/master/CONTRIBUTING.md).


[#issue_tracker]: https://github.com/j-c-cook/ghedt/issues
[#pull_request]: https://github.com/j-c-cook/ghedt/pulls
[#pep]: https://www.python.org/dev/peps/pep-0008
[#numpydoc]: https://github.com/numpy/numpy/blob/master/doc/example.py
[#sphinx]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[#git_branches]: https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell
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
issue. If you find a closed issue related to your problem, open a new issue
and link to the closed issue.
2. **Use a short and descriptive title** when creating a new issue.
3. **Provide detailed steps to reproduce the problem.** Explain, in details,
how to reproduce the problem and what should be the **expected result.** When
possible, include a simple code snippet that isolates and reproduces the
problem.

After submitting a bug report, if you wish to contribute code to fix the
problem, follow the steps outlined in the contribution workflow.

## Fork, clone and configure the repository

If you are a developer and are interested in contributing to *ghedt* by 
modifying the code, the first step will likely be to [fork][#fork] the 
repository.

1. [Fork][#fork] the repository to your own Github profile. You will be able to 
   alter any of the code in your own repository but will only be able to modify 
   the code in my repository via pull requests.
2. [Clone][#clone] the forked repository on your personal computer: `git clone https://github.com/YOUR-USERNAME/ghedt`
3. [Configure][#Configure] my repository as a remote fork: `git remote add upstream https://github.com/BETSRG/GHEDTOSU`.
   With my repository configured as a fork you will be able to keep your main 
   branch up to date with mine.

## Contribution workflow

This section outlines the steps for contributing to *ghedt*.

1. **Open a new [issue][#issue_tracker].**
2. **Use a short and descriptive title.** When proposing an enhancement,
describe in details what the enhancement would entail. If you plan to implement
the enhancement yourself, provide a step-by-step plan for the implementation.
3. **Explain how the enhancement benefits _pygfunction_.**
4. **Create (checkout) a new branch from the master.** The branch name should
follow the naming convention: `issue#_shortDescription`. For example:
issue1_loadAggregation. The following is how to checkout, push a branch to your
remote repository, and then set the local branch to an upstream branch. 
```angular2html
git checkout -b $branchName &&
git push -u origin $branchName &&
git branch --set-upstream-to=origin/$branchName $branchName
```
To list all the branches, run the following command. Now you should be able to 
see that you have the new branch you created checked out, and that there is an 
upstream branch as well. 
```angular2html
git branch -a
```
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

## Styleguide

`ghedt` follows the [PEP8 style guide][#pep]. 

## References

This contributing outline was originally taken from [pygfunction](https://github.com/MassimoCimmino/pygfunction/blob/master/CONTRIBUTING.md).


[#issue_tracker]: https://github.com/BETSRG/GHEDTOSU/issues
[#pull_request]: https://github.com/BETSRG/GHEDTOSU/pulls
[#pep]: https://www.python.org/dev/peps/pep-0008
[#numpydoc]: https://github.com/numpy/numpy/blob/master/doc/example.py
[#sphinx]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[#git_branches]: https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell
[#fork]: https://docs.github.com/en/get-started/quickstart/fork-a-repo
[#clone]: https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository
[#Configure]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-for-a-fork
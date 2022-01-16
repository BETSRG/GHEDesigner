# INSTALLATION

The `ghedt` package is tested on Python 3.7 and 3.8. It is recommended that 
either Python 3.7 or 3.8 are used. There are two options for setting up a Python
[virtual environment][#VirtualEnvironments] capable of running `ghedt`:
1. Go to [Python.org](https://www.python.org/downloads/) and download release
   version 3.7.11 for your operating system. 
2. Download the free open-source individual [anaconda][#anaconda] package 
   manager.

Please ensure that whether you install (1) or (2) above that you select the 
"Add to PATH" option any time you see it. Now go into the command prompt and 
type in `python`. The dynamic editor should pop up. It will look something like
the following:
```angular2html
Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
To exit the dynamic environment, now type `quit()` and hit enter. Now ensure 
that the package installer for python is in your path by typing `pip`. You 
should see something like the following:
```angular2html
Usage:   
  pip <command> [options]

Commands:
  install                     Install packages.
  download                    Download packages.
  uninstall                   Uninstall packages.
  freeze                      Output installed packages in requirements format.
  list                        List installed packages.
  show                        Show information about installed packages.
  check                       Verify installed packages have compatible dependencies.
  config                      Manage local and global configuration.
  search                      Search PyPI for packages.
  cache                       Inspect and manage pip's wheel cache.
  index                       Inspect information available from package indexes.
  wheel                       Build wheels from your requirements.
  hash                        Compute hashes of package archives.
  completion                  A helper command used for command completion.
  debug                       Show information useful for debugging.
  help                        Show help for commands.
```

Now to install `ghedt`, simply type `pip install ghedt` and hit enter. You 
should now be able to successfully run the examples. 

[#git]: https://en.wikipedia.org/wiki/Git
[#git-downloads]: https://git-scm.com/downloads
[#git-book]: https://git-scm.com/book/en/v2
[#VirtualEnvironments]: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
[#pyg-branch]: https://github.com/j-c-cook/pygfunction/tree/ghedt
[#anaconda]: https://www.anaconda.com/products/individual
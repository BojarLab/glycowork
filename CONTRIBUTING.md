# How to contribute

## How to get started

Before anything else, please install the git hooks that run automatic scripts during each commit and merge to strip the notebooks of superfluous metadata (and avoid merge conflicts). After cloning the repository, run the following command inside it:
```
nbdev_install_hooks
```

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

### Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch, based on the current dev branch and targeted to merge into the dev branch.
* Ensure that your PR includes a test that fails without your patch, and passes with it.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## PR submission guidelines

* Before submitting a PR, run these commands locally, that will run all tests and return any errors that may have occurred:

```
nbdev_prepare
cd tests
pytest
```

* Please update CHANGELOG.md with any significant changes. Add your changes in the relevant module/submodule section in the following format:

### module_name

#### sub_module_name

##### Added ✨

* Added new feature X (commit-hash)

##### Changed 🔄

* Modified behavior of Y (commit-hash)

##### Deprecated ⚠️

* Removed feature Z (commit-hash)

##### Fixed 🐛

* Fixed bug in W (commit-hash)

* We have a PR template in the .github folder. Consider using it.
* Keep each PR focused. While it's more convenient, do not combine several unrelated fixes together. Create as many branches as needing to keep each PR focused.
* Do not mix style changes/fixes with "functional" changes. It's very difficult to review such PRs and it most likely gets rejected.
* Do not add/remove vertical whitespace. Preserve the original style of the file you edit as much as you can.
* Do not turn an already submitted PR into your development playground. If after you submitted PR, you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
* If, however, you submitted a PR and received a request for changes, you should proceed with commits inside that PR, so that the maintainer can see the incremental fixes and won't need to review the whole PR again. In the exception case where you realize it'll take many many commits to complete the requests, then it's probably best to close the PR, do the work and then submit it again. Use common sense where you'd choose one way over another.
* If you added functionality, please consider adding corresponding unit tests to tests/test_core_functions.py

## Do you want to contribute to the documentation?

* Docs are automatically created from the notebooks in the nbs folder.

## Wishlist for future glycowork updates (last update: 2024-12-17)

### Urgent

* more, and more informative, error messages

### At some point

* any further expansion of our universal input pipeline, to cover more usecases etc.
* split motif_list into ‘core’ motifs (occurring frequently) and ‘extended’ motifs (that are rare or niche) for performance reasons
* characterize_monosaccharide only factors in subsequent sequence context; make it possible (as an option) to also consider upstream sequence context
* implement multiple sequence alignment and other substitution matrices
* parallelize motif matching
* refactor glycan graphs to be directed graphs & treat linkages as edges (would mean a *lot* of downstream fixes/refactoring)
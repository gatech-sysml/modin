README For Branch Management
----------------------------

For the gatech-sysml MODIN repo, we will maintain two branches:

modin_upstream, which is a branch that is up-to-date with the master branch of the actual MODIN repo,

and

local_master, where we will branch off from, and open PRs for.

The general framework of contributing to gatech-sysml's MODIN repo is as follows:

1. Clone the repo.
2. run git checkout local_master.
3. run git checkout -b <issue-#>, where issue-# corresponds to what issue you are going to tackle.
4. Make your commits and write your code, and push up to remote by doing git push origin <issue-#>.
5. Open a PR to merge the contents of your branch into local_master.


We will be updating modin_upstream every week so that we have some frame of reference to start from, in case
things go south, or branches get corrupted.

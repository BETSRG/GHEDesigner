#!/bin/bash

read -p "Enter the name of the new branch (vx.x.x): " branchName

git checkout -b $branchName &&
git push -u origin $branchName &&
git branch --set-upstream-to=origin/$branchName $branchName &&
git branch -a

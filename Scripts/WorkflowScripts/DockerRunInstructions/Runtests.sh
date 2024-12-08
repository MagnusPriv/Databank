#!/bin/bash
git checkout $BRANCH_NAME
git pull
cd Scripts/tests/ 
pytest -vs

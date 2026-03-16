@echo off
echo Generating clean requirements.txt...
pipreqs . --force --ignore .venv --encoding iso-8859-1
echo Finished
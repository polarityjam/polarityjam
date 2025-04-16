# Polarityjam Documentation

Short description of how to locally look at the documentation after you jave done some changes.
Especially those that refer to the notebooks.

## Build and check the documentation

1. install micromamba
2.
3. install an environment - last tested python version is 3.13:

```
micromamba create -y -n sphinx-polarityjam python pip -c conda-forge
micromamba activate sphinx-polarityjam
pip install -r docs/requirements.txt
```

3. Check if tag for "old" version is present.
   Create a tag before of the "old" version of the documentation (e.g. v0.1.0) if not.
   This is important for the readthedocs build process.

```
git tag -a v0.2.2doc -m "documentation for v0.2.2"
git push origin tag v0.2.2doc
```

Stick to the "doc" tag naming convention to differentiate between the code and the documentation version.

4. make sure "pandoc" is installed (on ubuntu: `sudo apt install pandoc`)

5. go to docs folder: `cd docs/`

6. build the documentation locally with `make html`. A folder "\_build" should appear (e.g. docs/\_build).
   This is your local version of the documentation!

7. check for errors in the make process - resolve eventual errors

8. open your browser and drag and drop docs/\_build/html/index.html into your window.
   This will be the view users have on readthedocs.org once you push changes to the docs branch

9. check documentation locally for errors and style. Especially the notebook files! Repeat step 5-8 until satisfied.

10. DO NOT ADD AND THEN PUSH THE "\_build" DIRECTORY! IT SHOULD NOT BE IN THE GITHUB REPOSITORY OF POLARITYJAM!

---
title: AI in the Industry Tutorials (Density Estimation for Anomaly Detection)
author: michele.lombardi2@unibo.it
---

# Density Estimation for Anomaly Detection #

This is the first lecture of the 2021/2022 edition of the "AI in the Industry" course, from [University of Bologna](https://www.unibo.it). Each lecture consists of a tutorial that tackles a simplified industrial problem and tackles it using AI techniques, from Machine Learning to Combinatorial Optimization (and later on their combination).

While the whole course looks like cookbook, the real goal is using examples to teach how industrial problem can be methodically approaches, analyzed, and tackles using a combination of techniques.

This tutorial in particular tackles a simple anomaly detection task via density estimation techniques. The focus is on building an end-to-end solution, where calibrating the detection threshold can be as important as building the density estimation model itself. Being the first lecture, it also contains specific details about the course and the student evaluation methods.

# Accessing the Lecture #

## Using Binder (Easiest) ##

The tutorial is split over multiple (numbered) Jupyter notebooks. _The easiest way to access the tutorial is via Binder_, which enables running the code on a remote server. This is as easy as clicking on the following button:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lompabo/aiiti-01-2021/HEAD)

...And waiting for a while for the environment to load up.

## Local Execution (Preferred) ##

Using Binder is easy, but it may be slow, and most importantly it does not allow one to save progress or modifications done while experimenting. For this reason _students are encouraged to run all lectures locally_. Doing this will require to:

* Install Docker, by following the [online instructions](https://docs.docker.com/get-docker/).
* Install Docker Compose, by following the [online
instructions](https://docs.docker.com/compose/install/)
* Clone the repository with the tutorial, in this case via the command:
```sh
git clone https://github.com/lompabo/aiiti-01-2021.git
```
* Start the container via Docker Compose, from the main directory of the
tutorial:
```sh
docker-compose up
```

On linux systems, you may need to start the docker service first.

No matter which OS your are running, the first execution of this process will be fairly long, since Docker will need to download a base image for the container (think of a virtual machine disk) and then some boilerplate configuration steps will need to be performed (e.g. installing jupyter in the container). Subsequent runs will be much faster.

The process will end with a message such as this one:
```sh
To access the notebook, open this file in a browser:
    file:///home/lompa/.local/share/jupyter/runtime/nbserver-1-open.html
Or copy and paste this URL:
    http://127.0.0.1:39281/?token=0cd92163797c3b3abe67c2b0aea57939867477d6068708a2
```
Copying one of the two addresses in a file browser will provide access to the Jupyter server running in the spawned container. By default, the main lecture folders is shared with the container environment, so any modification you make in the contain will reflect in the host system, and the other way round.

Once you are done, pressing CTRL+C on the terminal will close the Docker container.

For more information about how Docker works (such as the difference between images and containers, or how to get rid of all of them once you are done with the tutorial), you can check the [Docker documentation](https://docs.docker.com/).

## Read-only Access and PDF Notes ##

You can inspect the individual notebooks in by just clicking on any `*.ipynb` file in the `notebooks` directory: github provides a notebook viewer that mostly works, though this access method may occasionally have issue when displaying plots.

The repository contains PDF notes for all the notebooks. They can be used for read-only access (with more consistent results compared to the github notebook viewer), but more importantly they can be useful to add annotations. Just keep in mind that in case of updates, cloning the repository will replace the PDF files.

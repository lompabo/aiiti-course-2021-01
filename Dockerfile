# This is Binder-compatible Dockerfile, which will enable:
# - Online execution via https://mybinder.org/ or a similar environment
# - Local execution via Docker
# Ensuring compatibility requires some care in the Dockerfile definition;
# detailed instructions are available at:
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html
# while some notes are provided directly in this file as commments

# Specify the base image
# NOTE: Binder requires a tag to be specified
FROM python:3.8

# Define a user whose uid is 1000
# NOTE: this is required for Binder compatibility, to ensure that processes
# are not run as root. The "adduser" command is fine for Debian-based images
# (such as python:3.8) and should be replaced when a different distribution
# is used. Keep this as it is unless you know what you are doing
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Install minimal requirements
# NOTE: these are for Binder compatibility. You can change the jupyter and
# jupyterhub versions, but referring a specific version is highly advised
# to ensure reproducibility. Feel free to update the version tags, but it's
# better to keep both packages installed
RUN pip install --no-cache-dir notebook==6.2.0
RUN pip install --no-cache-dir jupyterhub

# CONTAINER SPECIFIC INSTALLATION -------------------------------------------
# Most of the configuration steps specific to your container can fit here

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip

# Install additional Python packages
RUN pip install jupyter pandas sklearn matplotlib ipympl RISE jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --system

# END OF THE CONTAINER SPECIFIC INSTALLATION --------------------------------

# Make sure the contents of our repo are in ${HOME}
# NOTE: this is needed again by Binder, to make the notebook contents
# available to all users. We also need to change the ownership of the home
# directory we previously built. The snippet ends with a user switch
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Specify working directory
WORKDIR ${HOME}

# Use CMD to specify the starting command
# NOTE: Binder will override this by explicitily calling a program (jupyter)
# within the container, and by passing its own list of arguments
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
#      "--ip=0.0.0.0", "--allow-root"]

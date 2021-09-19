# Specify the base image
FROM python:3.8

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip

# Install additional Python packages
RUN pip install jupyter pandas sklearn matplotlib ipympl RISE jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --system

# Make sure the contents of our repo are in /app
COPY . /app

# Specify working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]

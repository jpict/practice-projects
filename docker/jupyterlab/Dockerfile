# Use miniconda base image
FROM continuumio/miniconda3

# Create conda environment
ADD environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml

# Set working directory
WORKDIR /workspace

# Launch into jupyterlab
# Need to use conda run to do this since we can't activate a conda environment with RUN
# An alternative would be to add a "conda activate" to .bashrc and manually edit the path environment variable
CMD ["conda", "run", "--no-capture-output", "-n", "test", "jupyter-lab", "--ip=0.0.0.0", "--allow-root"]

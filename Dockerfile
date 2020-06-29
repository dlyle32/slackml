FROM continuumio/miniconda3

WORKDIR /slackai

# Create the environment:
COPY environment.yml .
RUN conda env create --file environment.yml
#RUN conda install -y numpy
#RUN conda install -y pandas 
#RUN conda install -y matplotlib 

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "slack_ai", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure pandas is installed:"
RUN python -c "import pandas"

# The code to run when container is started:
COPY . .

ENTRYPOINT ["conda", "run", "-n", "slack_ai", "python", "src/NN.py"]

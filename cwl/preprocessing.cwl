cwlVersion: v1.2              # Specifies the version of the Common Workflow Language (CWL) being used
class: CommandLineTool       # Indicates that this is a CWL tool that executes a command-line program

baseCommand: [conda, run, -p, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/use-cases/mnist/.venv-preproc, python, /afs/cern.ch/work/a/azoechba/T6.5-AI-and-ML/use-cases/mnist/mnist-preproc.py]
# The command to be executed by the tool. It runs the 'conda' command with 'run' subcommand, 
# then sets the path to the virtual environment to be used, and finally runs the 'mnist-preproc.py' 
# script using the 'python' interpreter.

# The following requirement does not solve the permission error, but sets the environment variable 
# 'FILE_READ_BUFFER_SIZE' to a value of 10.
requirements:
  EnvVarRequirement:
    envDef:
      FILE_READ_BUFFER_SIZE: "10"

# Defines the inputs to the CWL tool
inputs:
  rawDatasetPath:
    type: string           # Specifies the data type of the input as a string
    inputBinding:          # Defines how the input should be passed to the command line tool
      position: 1          # Specifies the position of the argument in the command line
      prefix: --input      # Specifies the prefix to be used for the argument

  preprocesseDatasetPath:
    type: string           # Specifies the data type of the input as a string
    inputBinding:          # Defines how the input should be passed to the command line tool
      position: 2          # Specifies the position of the argument in the command line
      prefix: --output     # Specifies the prefix to be used for the argument

# Defines the outputs of the CWL tool
outputs:
  preprocessingStdout:
    type: stdout           # Specifies that the output type is the standard output of the command.

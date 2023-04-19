"""
Entry point of the any workflow.
This script gathers the info from the workflow definition file
and runs a workflow step by step, passing the info loaded from
the workflow definition file.

Example:

>>> python run-workflow.py -f ./use-cases/mnist/training-workflow.yml

"""

import os
import subprocess
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simple DT workflow.')
    parser.add_argument(
        '-f', '--workflow-file',
        type=str,
        help='Path to file describing DT a workflow.',
        required=True
    )
    args = parser.parse_args()

    # 1. parse workflow file

    with open(args.workflow_file, "r") as stream:
        try:
            workflow = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # 2. Deploy steps (i.e., create conda envs), if not already there

    for step in workflow.get('steps'):
        step_name, step_data = next(iter(step.items()))
        if not os.path.exists(step_data['env']['prefix']):
            print(f'Deploying "{step_name}" step...')
            # Install python env
            subprocess.run(
                (f"mamba env create -p {step_data['env']['prefix']} "
                 f"--file {step_data['env']['file']}"),
                shell=True,
                check=True
            )

    # 3. Run the workflow step-by-step. Like 'conda run ...'
    datasets = workflow.get('datasets')
    for step in workflow.get('steps'):
        step_name, step_data = next(iter(step.items()))
        input_dataset = (
            datasets[step_data['input']]['location']
            if step_data['input'] is not None
            else None
        )
        output_dataset = (
            datasets[step_data['output']]['location']
            if step_data['output'] is not None
            else None
        )
        print(f'Running "{step_name}" step...')
        print(
             f"conda run -p {step_data['env']['prefix']} "
             f"{step_data['command']} "
             f"--input {input_dataset} "
             f"--output {output_dataset} \n\n"
             )
        subprocess.run(
            (f"conda run -p {step_data['env']['prefix']} "
             f"{step_data['command']} "
             f"--input {input_dataset} "
             f"--output {output_dataset} "
             ),
            shell=True,
            check=True,
        )

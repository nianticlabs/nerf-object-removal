import os
import subprocess
import kfp
from kfp.components import func_to_container_op
import argparse
import kubernetes as k8s # should pip install kubernetes first

# This is the URL of the pipelines and should be copied to your code
PIPELINES_URL = 'https://5a49b7b435cff2eb-dot-europe-west2.pipelines.googleusercontent.com/'

# This should work when executed within gcloud without extra steps.
# If you're executing from your mac, first run `gcloud auth application-default login`
client = kfp.Client(host=PIPELINES_URL)

# This allows you to mount the shared storage and you should copy it to your code
FILESTORE_CLAIM = kfp.dsl.PipelineVolume("interns-ssd-filestore-claim")

# training the nerf model
def train():

    # Add any imports used in the function within the function scope
    import os
    os.system(f"python bin/train.py -cn lama-rgbd")

# # define all container functions
# train_function = func_to_container_op(train, base_image='eu.gcr.io/res-interns/silvan-object-removal:latest')
# eval_function = func_to_container_op(eval, base_image='eu.gcr.io/res-interns/silvan-object-removal:latest')
# eval_train_function = func_to_container_op(eval_train, base_image='eu.gcr.io/res-interns/silvan-object-removal:latest')

image_sha = os.popen('gcloud container images describe eu.gcr.io/res-interns/lama:latest --format="value(image_summary.digest)"')
image_sha = image_sha.read().split('\n')[0]

train_function = func_to_container_op(train, base_image=f'eu.gcr.io/res-interns/lama@{image_sha}')

@kfp.dsl.pipeline(name='silvanweder-lama')
def train_and_evaluate(machine_type: str = 'nvidia-tesla-a100'):

    train_op = train_function()
    
    # avoid out of shared memory
    volume = kfp.dsl.PipelineVolume(volume=k8s.client.V1Volume(
        name="shm",
        empty_dir=k8s.client.V1EmptyDirVolumeSource(medium='Memory')))

    train_op.add_pvolumes({'/dev/shm': volume})

    # Set up the environment that these operations will be run in
    # Request 8 GPU for training
    # More documentation on how to request GPUs here: https://github.com/kubeflow/pipelines/blob/master/samples/tutorials/gpu/gpu.ipynb
    train_op.set_gpu_limit(4)
    train_op.add_node_selector_constraint('cloud.google.com/gke-accelerator', machine_type) # machine_type is also available
    # Mount the shared storage in /mnt/res_nas
    train_op.add_pvolumes({'/mnt/res_nas': FILESTORE_CLAIM})

def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine_type', type=str, default='nvidia-tesla-a100')
    
    return parser.parse_args()

if __name__ == '__main__':

    # This submits the pipeline to run immediately
    args = arg_parser()

    client.create_run_from_pipeline_func(
        train_and_evaluate,
        arguments={
            'machine_type': args.machine_type
        },
        experiment_name=None, # If you give this a value you can group runs together and compare output metrics
        run_name=f'silvanweder lama'
    )

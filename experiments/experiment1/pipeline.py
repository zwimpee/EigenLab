import os
from tfx.components import CsvExampleGen, TensorBoard
from tfx.components.trainer.component import TrainerComponent
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import trainer_pb2

# Set the paths and directories
_pipeline_name = 'experiment1'
_pipeline_root = os.path.join('pipeline', _pipeline_name)
_data_path = 'data/experiment1.sqlite'
_log_path = 'data/docs/logs/' + _pipeline_name + '.log

# Define the data ingestion component
example_gen = CsvExampleGen(input_base=_data_path)

# Define the trainer component
trainer = TrainerComponent(
    trainer_pb2.TrainArgs(num_steps=1000),
    trainer_pb2.EvalArgs(num_steps=100),
    module_file='./train.py',  # Path to your existing training script
    transformed_examples=example_gen.outputs['examples'])

# Define the TensorBoard component
tensorboard = TensorBoard(
    log_dir=_log_path,
    enable_cache=True)  # Enable caching for TensorBoard

# Define the pipeline
components = [example_gen, trainer]
pipeline_name = 'test_pipeline'
p = pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=_pipeline_root,
    components=components)

# Run the pipeline
context = InteractiveContext()
context.run(p)

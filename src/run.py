from rich import print

from data import get_dataloader_dict
from experiment_builder import ExperimentBuilder
from src.learner import MAMLFewShotClassifier
from utils.parser_utils import get_args

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(
    args=args,
    device=device,
    im_shape=(2, args.image_channels, args.image_height, args.image_width),
)

dataloader_dict = get_dataloader_dict(
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    num_workers=args.num_dataprovider_workers,
    seed=args.seed,
    num_train_episodes=100000,
    num_eval_episodes=600,
    num_classes_per_set=args.num_classes_per_set,
    num_samples_per_class=args.num_samples_per_class,
    num_target_samples=args.num_target_samples,
    data_cache_dir="datasets",
)
maml_system = ExperimentBuilder(
    model=model, dataloader_dict=dataloader_dict, args=args, device=device
)
maml_system.run_experiment()

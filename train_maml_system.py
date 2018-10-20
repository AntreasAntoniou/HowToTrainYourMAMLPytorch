from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from one_shot_learning_network import MAMLFewShotClassifier
from utils.parser_utils import get_args

args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device,
                                      im_shape=(2, args.image_channels,
                                                       args.image_height, args.image_width))
data = MetaLearningSystemDataLoader(args=args)
maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
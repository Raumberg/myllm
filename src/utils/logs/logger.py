import logging
import sys
import datasets
import transformers
from transformers import TrainingArguments

def setup_logging(logger: logging.Logger, training_args: TrainingArguments) -> None:
    """
    Set up logging for the training process.

    This function configures the logging format, sets the logging level based on the 
    provided training arguments, and enables logging for the datasets and transformers libraries.

    Args:
        logger (logging.Logger): The logger instance to configure.
        training_args (TrainingArguments): The training arguments containing logging settings.

    Returns:
        None: This function does not return any value. It configures logging in place.

    Notes:
        - The logging format includes the timestamp, log level, logger name, and message.
        - If `training_args.should_log` is True, the log level is set to INFO by default.
        - The log level is determined by `training_args.get_process_log_level()`.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity(log_level)
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()
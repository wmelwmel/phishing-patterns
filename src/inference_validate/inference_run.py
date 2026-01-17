import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pipelines import InferencePipeline, MessagePreparer, SentenceSplitter

from settings import InferenceSettings, MLflowSettings


def main() -> None:
    logger.info("Start inference pipeline...")

    load_dotenv()

    infer_config = InferenceSettings()
    mlflow_settings = MLflowSettings()

    messages = MessagePreparer(infer_config.messages_to_process).prepare_message_data()
    splitter = SentenceSplitter(
        languages=infer_config.languages, use_gpu=infer_config.use_gpu, use_txt_split=infer_config.use_txt_split
    )
    pipeline = InferencePipeline(
        splitter=splitter, run_id=infer_config.run_id, mlflow_settings=mlflow_settings, use_gpu=infer_config.use_gpu
    )
    results = pipeline.run(messages)

    if infer_config.verbose:
        for msg_n, result in enumerate(results, start=1):
            logger.info(
                "\nMessage {}:\n{}\nContent:\n{}\nPrediction:\n{}\n",
                msg_n,
                result["message_id"],
                np.array(result["sentences"]),
                np.array(result["predicted_features"]),
            )


if __name__ == "__main__":
    main()

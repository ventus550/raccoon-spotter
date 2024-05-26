from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, radialplot_comparison, sample_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["trained_model", "raccoon_test_features_data_array"],
                outputs="metrics",
            ),
            node(
                func=sample_model,
                inputs=["trained_model", "raccoon_test_features_data_array"],
                outputs="sampled_predictions",
            ),
            node(
                func=radialplot_comparison,
                inputs=[
                    "trained_model",
                    "untrained_model",
                    "raccoon_test_features_data_array",
                ],
                outputs="radialplot_comparison",
            ),
        ]
    )

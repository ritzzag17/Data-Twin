"""Run the whole DataTwin pipeline (or selected steps) in the right order.

    python run_pipeline.py                    # everything
    python run_pipeline.py --only train,predict
    python run_pipeline.py --skip trust,clusters
    python run_pipeline.py --llm              # use the Anthropic API for UNCERTAIN pairs
"""
import argparse
import time
import traceback

from src import (
    clustering, data_quality, duplicate_detection, match_explainer,
    predict_matches, prepare_training_data, prompt_engine, review_manager,
    train_model, trust_score,
)
from src.config import DB_PATH, MATCHES_PATH


def _review_db():
    review_manager.initialize_database(DB_PATH)
    added = review_manager.import_matches(MATCHES_PATH, DB_PATH)
    print(f"Review database ready ({added:,} new pairs).")
    print(review_manager.get_review_statistics(DB_PATH).to_string(index=False))


def build_steps(use_llm):
    return [
        ("quality", data_quality.main),
        ("matching", duplicate_detection.main),
        ("review_db", _review_db),
        ("training_data", prepare_training_data.prepare_training_data),
        ("train", train_model.train_models),
        ("predict", predict_matches.generate_predictions),
        ("explain", match_explainer.generate_explanations),
        ("judge", lambda: prompt_engine.run_prompt_engine(use_llm=use_llm)),
        ("trust", trust_score.main),
        ("clusters", clustering.main),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma separated step names")
    parser.add_argument("--skip", help="comma separated step names to skip")
    parser.add_argument("--llm", action="store_true", help="use LLM judge (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    steps = build_steps(args.llm)
    names = [n for n, _ in steps]
    only = set(args.only.split(",")) if args.only else set(names)
    skip = set(args.skip.split(",")) if args.skip else set()
    unknown = (only | skip) - set(names)
    if unknown:
        parser.error(f"unknown step(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(names)}")

    failures = []
    for name, func in steps:
        if name not in only or name in skip:
            continue
        print(f"\n{'#' * 70}\n# STEP: {name}\n{'#' * 70}")
        start = time.time()
        try:
            func()
            print(f"[{name}] finished in {time.time() - start:.1f}s")
        except Exception:
            failures.append(name)
            traceback.print_exc()
            print(f"[{name}] FAILED - continuing with the next step")
    print("\nPipeline finished." + (f" Failed steps: {', '.join(failures)}" if failures else " All steps OK."))


if __name__ == "__main__":
    main()

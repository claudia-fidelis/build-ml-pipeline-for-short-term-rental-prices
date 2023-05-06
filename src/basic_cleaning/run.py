#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading the data")
    df = pd.read_csv(artifact_local_path)

    # Drop the outlier
    logger.info(f"Dropping the outliers: keeps only values between {args.min_price} and {args.max_price}.")
    df = df[(df['price'] < args.max_price) & (df['price'] > args.min_price)].copy()

    # Convert last_review to datetime
    logger.info(f"Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remove locations outside of boundaries
    logger.info("Removing locations outside of boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Saving to CSV
    logger.info("Saving the clean data")
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
         args.output_artifact,
         type=args.output_type,
         description=args.output_description,
     )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Cleaning dataset",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A brief description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price",
        required=True
    )


    args = parser.parse_args()

    go(args)

# Databricks notebook source
# MAGIC %md
# MAGIC # Geneformer Data Preparation
# MAGIC 
# MAGIC This notebook prepares the data for Geneformer pretraining:
# MAGIC 1. Downloads the Genecorpus-30M dataset from HuggingFace
# MAGIC 2. Downloads the token dictionary
# MAGIC 3. Converts to MDS (Mosaic Data Shard) format for streaming
# MAGIC 
# MAGIC **Run this notebook with a CPU cluster before starting GPU training.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC 
# MAGIC Update these values to match your Databricks environment.

# COMMAND ----------

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================

# Databricks Volume Configuration
CATALOG = "main"
SCHEMA = "guanyu_chen"  
VOLUME_NAME = "sgc"

# Data paths (relative to volume root)
TOKEN_DICT_PATH = "geneformer/data/token_dictionary.pkl"
SOURCE_DATASET_PATH = "geneformer/data/dataset/genecorpus_30M_2048.dataset"
STREAMING_DATASET_PATH = "geneformer/data/dataset/streaming/genecorpus_30M_2048.dataset"

# Data preparation settings
TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

# HuggingFace source URLs
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main"
TOKEN_DICT_URL = f"{HUGGINGFACE_BASE_URL}/token_dictionary.pkl"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup and Path Construction

# COMMAND ----------

import os
import shutil
import subprocess

# Construct volume path
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# Construct full paths
PATHS = {
    'volume': VOLUME_PATH,
    'token_dictionary': f"{VOLUME_PATH}/{TOKEN_DICT_PATH}",
    'source_dataset': f"{VOLUME_PATH}/{SOURCE_DATASET_PATH}",
    'streaming_dataset': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}",
    'train_dir': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}/train",
    'test_dir': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}/test",
}

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Volume Path: {VOLUME_PATH}")
print(f"Token Dictionary: {PATHS['token_dictionary']}")
print(f"Source Dataset: {PATHS['source_dataset']}")
print(f"Streaming Dataset: {PATHS['streaming_dataset']}")
print(f"  - Train: {PATHS['train_dir']}")
print(f"  - Test: {PATHS['test_dir']}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Helper Functions

# COMMAND ----------

def check_mds_valid(path: str) -> bool:
    """Check if MDS dataset directory is valid (has index.json)."""
    index_file = os.path.join(path, 'index.json')
    return os.path.exists(index_file)


def get_dir_size(path: str) -> str:
    """Get human-readable directory size."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def download_file(url: str, dest: str, desc: str = None):
    """Download a file with progress."""
    desc = desc or os.path.basename(dest)
    print(f"Downloading {desc}...")
    
    # Create parent directory
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Use curl for download with progress
    cmd = f'curl -L "{url}" -o "{dest}" --progress-bar'
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0 and os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  ✅ Downloaded: {dest} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ Failed to download: {dest}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check Current Status

# COMMAND ----------

def check_status():
    """Check current status of all data components."""
    print("=" * 60)
    print("CURRENT STATUS")
    print("=" * 60)
    
    status = {}
    
    # Check token dictionary
    token_exists = os.path.exists(PATHS['token_dictionary'])
    status['token_dictionary'] = token_exists
    icon = "✅" if token_exists else "❌"
    print(f"{icon} Token Dictionary: {PATHS['token_dictionary']}")
    
    # Check source dataset
    source_exists = os.path.exists(PATHS['source_dataset'])
    if source_exists:
        # Check for required files
        required = ['dataset.arrow', 'dataset_info.json', 'state.json']
        files = os.listdir(PATHS['source_dataset']) if os.path.isdir(PATHS['source_dataset']) else []
        source_valid = all(f in files for f in required)
    else:
        source_valid = False
    status['source_dataset'] = source_valid
    icon = "✅" if source_valid else "❌"
    print(f"{icon} Source Dataset: {PATHS['source_dataset']}")
    
    # Check train MDS
    train_valid = check_mds_valid(PATHS['train_dir'])
    status['train_mds'] = train_valid
    icon = "✅" if train_valid else "❌"
    size_str = f" ({get_dir_size(PATHS['train_dir'])})" if train_valid else ""
    print(f"{icon} Train MDS: {PATHS['train_dir']}{size_str}")
    
    # Check test MDS
    test_valid = check_mds_valid(PATHS['test_dir'])
    status['test_mds'] = test_valid
    icon = "✅" if test_valid else "❌"
    size_str = f" ({get_dir_size(PATHS['test_dir'])})" if test_valid else ""
    print(f"{icon} Test MDS: {PATHS['test_dir']}{size_str}")
    
    print("=" * 60)
    
    # Summary
    if status['train_mds'] and status['test_mds'] and status['token_dictionary']:
        print("\n🎉 All data is ready for training!")
        print("You can skip to the Summary section for the configuration.")
    else:
        print("\n⚠️  Some data is missing. Run the cells below to prepare it.")
    
    return status

status = check_status()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Download Token Dictionary

# COMMAND ----------

def download_token_dictionary():
    """Download token dictionary from HuggingFace."""
    if os.path.exists(PATHS['token_dictionary']):
        print(f"✅ Token dictionary already exists: {PATHS['token_dictionary']}")
        return True
    
    return download_file(TOKEN_DICT_URL, PATHS['token_dictionary'], "token_dictionary.pkl")

# Run download
download_token_dictionary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Download Source Dataset
# MAGIC 
# MAGIC Download the Genecorpus-30M dataset from HuggingFace.
# MAGIC 
# MAGIC **⚠️ This may take 15-30 minutes depending on your network speed (~15GB).**

# COMMAND ----------

def download_source_dataset():
    """Download source dataset from HuggingFace."""
    source_path = PATHS['source_dataset']
    required_files = ['dataset.arrow', 'dataset_info.json', 'state.json']
    
    # Check if already exists
    if os.path.exists(source_path) and os.path.isdir(source_path):
        existing = os.listdir(source_path)
        if all(f in existing for f in required_files):
            size = get_dir_size(source_path)
            print(f"✅ Source dataset already exists: {source_path} ({size})")
            return True
    
    print("=" * 60)
    print("DOWNLOADING SOURCE DATASET")
    print("=" * 60)
    print("⚠️  This may take 15-30 minutes (dataset is ~15GB)")
    print()
    
    os.makedirs(source_path, exist_ok=True)
    
    dataset_files = [
        ("genecorpus_30M_2048.dataset/dataset.arrow", "dataset.arrow"),
        ("genecorpus_30M_2048.dataset/dataset_info.json", "dataset_info.json"),
        ("genecorpus_30M_2048.dataset/state.json", "state.json"),
    ]
    
    all_success = True
    for remote_file, local_file in dataset_files:
        url = f"{HUGGINGFACE_BASE_URL}/{remote_file}"
        dest = f"{source_path}/{local_file}"
        success = download_file(url, dest, local_file)
        if not success:
            all_success = False
    
    if all_success:
        size = get_dir_size(source_path)
        print(f"\n✅ Source dataset ready: {source_path} ({size})")
    else:
        print(f"\n❌ Some files failed to download")
    
    return all_success

# Run download
download_source_dataset()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Convert to MDS Format
# MAGIC 
# MAGIC Convert the HuggingFace dataset to MDS (Mosaic Data Shard) format for efficient streaming during training.
# MAGIC 
# MAGIC Uses Spark's `dataframe_to_mds` for parallel conversion - **typically ~9 minutes for 30M samples!**

# COMMAND ----------

# Install required packages if not available
# %pip install datasets mosaicml-streaming tqdm

# COMMAND ----------

def get_optimal_partitions():
    """
    Auto-detect optimal number of partitions based on Spark cluster configuration.
    Uses 2-4x the number of executor cores for good parallelism.
    """
    try:
        # Get Spark configuration
        sc = spark.sparkContext
        
        # Calculate total cores and optimal partitions (2-4x cores is usually good)
        total_cores = sc.defaultParallelism
        optimal_partitions = total_cores * 2  # 2x cores for good parallelism
        
        # Clamp to reasonable range
        optimal_partitions = max(8, min(optimal_partitions, 512))
        
        print(f"  Detected: {total_cores} total cores")
        print(f"  Using {optimal_partitions} partitions (2x cores)")
        
        return optimal_partitions
    except Exception as e:
        print(f"  Could not auto-detect cores ({e}), using default 32 partitions")
        return 32


def convert_arrow_to_parquet(dataset, force_recreate: bool = False):
    """
    Convert HuggingFace dataset (Arrow-backed) to Parquet format.
    Writes to local SSD first, then copies to Volume for Spark access.
    
    Args:
        dataset: HuggingFace dataset to convert (full dataset, not split)
        force_recreate: If True, clean existing temp files first
    
    Returns:
        str: volume_parquet_path (Spark-accessible path)
    """
    import pyarrow.parquet as pq
    
    # Temp paths - save in same folder as geneformer dataset
    dataset_base_folder = f"{VOLUME_PATH}/geneformer/data/dataset"
    local_parquet_path = "/local_disk0/temp_parquet_full"
    volume_parquet_path = f"{dataset_base_folder}/temp_parquet_full"
    
    # Check if volume parquet already exists (skip if valid and not forcing recreate)
    if not force_recreate and os.path.exists(volume_parquet_path):
        parquet_files = [f for f in os.listdir(volume_parquet_path) if f.endswith('.parquet')]
        if parquet_files:
            print(f"  ✓ Parquet already exists at: {volume_parquet_path}")
            return volume_parquet_path
    
    print(f"  Converting full dataset: {len(dataset):,} samples to Parquet...")
    
    # Clean temp directories if force_recreate is True
    if force_recreate:
        if os.path.exists(local_parquet_path):
            shutil.rmtree(local_parquet_path)
        if os.path.exists(volume_parquet_path):
            shutil.rmtree(volume_parquet_path)
    os.makedirs(local_parquet_path, exist_ok=True)
    
    # Access the underlying Arrow table directly (no conversion needed)
    arrow_table = dataset.data.table
    print(f"    Arrow table: {arrow_table.num_rows:,} rows, {arrow_table.nbytes / 1e9:.2f} GB")
    
    # Write to Parquet using PyArrow (much faster than HF's to_parquet)
    parquet_file = f"{local_parquet_path}/data.parquet"
    pq.write_table(arrow_table, parquet_file, compression='snappy')
    print(f"    Saved to local SSD: {parquet_file}")
    
    # Copy from local disk to Volume (Spark can't read local_disk0 directly)
    print(f"    Copying to Volume...")
    dbutils.fs.cp(
        f"file:{local_parquet_path}/",
        f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/geneformer/data/dataset/temp_parquet_full",
        recurse=True
    )
    print(f"    Copied to: {volume_parquet_path}")
    
    # Cleanup local temp (local SSD is ephemeral anyway)
    if os.path.exists(local_parquet_path):
        shutil.rmtree(local_parquet_path)
        print(f"    Cleaned up local temp: {local_parquet_path}")
    
    return volume_parquet_path


def convert_sparkdf_to_mds(spark_df, output_path: str, name: str = "", num_partitions: int = None):
    """
    Convert Spark DataFrame to MDS format using Spark's dataframe_to_mds.
    
    Args:
        spark_df: Spark DataFrame to convert
        output_path: Output path for MDS files
        name: Name for logging
        num_partitions: Number of partitions (auto-detected if None)
    """
    from streaming.base.converters import dataframe_to_mds
    from pyspark.sql.functions import col
    from pyspark.sql.types import LongType
    
    # Auto-detect optimal partitions if not specified
    if num_partitions is None:
        num_partitions = get_optimal_partitions()
    
    row_count = spark_df.count()
    print(f"    {name}: {row_count:,} rows")
    
    # Cast length to LongType to match expected 'int64'
    spark_df = spark_df.withColumn("length", col("length").cast(LongType()))
    
    # Repartition for parallel writes
    spark_df = spark_df.repartition(num_partitions)
    print(f"    Repartitioned to {num_partitions} partitions")
    
    # Clean output directory if exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # MDS kwargs - dataframe_to_mds requires explicit types
    # Output types compatible with train.py expectations:
    # - input_ids: ndarray:int64 (train.py converts to int64 anyway)
    # - length: int64 (compatible with Python int)
    mds_kwargs = {
        'out': output_path,
        'columns': {'input_ids': 'ndarray:int64', 'length': 'int64'},
        'compression': 'zstd',
        'size_limit': '256mb'
    }
    
    print(f"    Writing {name} MDS with dataframe_to_mds ({num_partitions} parallel writers)...")
    dataframe_to_mds(spark_df, merge_index=True, mds_kwargs=mds_kwargs)
    
    print(f"    ✓ {name} MDS complete!")


def prepare_mds_dataset(force_recreate: bool = False, num_partitions: int = None):
    """
    Convert HuggingFace dataset to MDS format using Spark's dataframe_to_mds.
    This uses parallel processing for fast conversion (~9 mins for 30M samples).
    
    Optimized pipeline:
    1. Load source dataset ONCE
    2. Convert FULL dataset to Parquet (Arrow → Parquet) - single conversion
    3. Load Parquet into Spark DataFrame
    4. Split in Spark (randomSplit) - proper train/test split
    5. Convert each split to MDS
    
    Args:
        force_recreate: If True, recreate even if MDS already exists
        num_partitions: Number of Spark partitions (auto-detected if None)
    """
    from datasets import load_from_disk
    import time
    
    # Check if already exists (skip if valid and not forcing recreate)
    mds_exists = check_mds_valid(PATHS['train_dir']) and check_mds_valid(PATHS['test_dir'])
    
    if mds_exists and not force_recreate:
        train_size = get_dir_size(PATHS['train_dir'])
        test_size = get_dir_size(PATHS['test_dir'])
        print(f"✅ MDS dataset already exists:")
        print(f"   Train: {PATHS['train_dir']} ({train_size})")
        print(f"   Test: {PATHS['test_dir']} ({test_size})")
        return True
    
    print("=" * 60)
    print("CONVERTING TO MDS FORMAT (Spark Parallel)")
    print("=" * 60)
    if mds_exists:
        print("⚠️  force_recreate=True, will overwrite existing data")
    print()
    
    start_time = time.time()
    
    # Check source dataset exists
    if not os.path.exists(PATHS['source_dataset']):
        print(f"❌ Source dataset not found: {PATHS['source_dataset']}")
        print("   Please run the download cell first.")
        return False
    
    # =========================================
    # STEP 1: Load source dataset ONCE
    # =========================================
    print(f"Step 1: Loading source dataset: {PATHS['source_dataset']}")
    dataset = load_from_disk(PATHS['source_dataset'])
    print(f"  Dataset loaded: {dataset}")
    print(f"  Total samples: {len(dataset):,}")
    
    # =========================================
    # STEP 2: Convert FULL dataset to Parquet
    # (Single Arrow → Parquet conversion)
    # =========================================
    print(f"\nStep 2: Converting full Arrow dataset to Parquet...")
    volume_parquet_path = convert_arrow_to_parquet(dataset, force_recreate=force_recreate)
    print(f"  ✓ Full dataset converted to Parquet: {volume_parquet_path}")
    
    # Free memory - we no longer need the HuggingFace dataset
    del dataset
    
    # =========================================
    # STEP 3: Load Parquet into Spark DataFrame
    # =========================================
    print(f"\nStep 3: Loading Parquet into Spark DataFrame...")
    spark_df = spark.read.parquet(volume_parquet_path)
    total_count = spark_df.count()
    print(f"  Loaded {total_count:,} rows into Spark DataFrame")
    
    # =========================================
    # STEP 4: Split in Spark
    # =========================================
    print(f"\nStep 4: Splitting in Spark (test_ratio={TEST_SPLIT_RATIO}, seed={RANDOM_SEED})...")
    train_ratio = 1.0 - TEST_SPLIT_RATIO
    train_df, test_df = spark_df.randomSplit([train_ratio, TEST_SPLIT_RATIO], seed=RANDOM_SEED)
    print(f"  Split ratios: train={train_ratio}, test={TEST_SPLIT_RATIO}")
    
    # =========================================
    # STEP 5: Convert each split to MDS
    # =========================================
    print(f"\nStep 5: Converting splits to MDS...")
    
    print(f"\n  [Train] {PATHS['train_dir']}")
    convert_sparkdf_to_mds(train_df, PATHS['train_dir'], name="Train", num_partitions=num_partitions)
    train_size = get_dir_size(PATHS['train_dir'])
    print(f"  ✓ Train MDS complete: {train_size}")
    
    print(f"\n  [Test] {PATHS['test_dir']}")
    convert_sparkdf_to_mds(test_df, PATHS['test_dir'], name="Test", num_partitions=num_partitions)
    test_size = get_dir_size(PATHS['test_dir'])
    print(f"  ✓ Test MDS complete: {test_size}")
    
    # Note: Keeping temp Parquet in volume for potential reuse
    print(f"\nParquet file kept at: {volume_parquet_path}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ MDS CONVERSION COMPLETE! (Total time: {elapsed/60:.1f} minutes)")
    print("=" * 60)
    return True

# Run MDS conversion
# Set force_recreate=True to recreate even if exists
# num_partitions=None auto-detects based on cluster cores
prepare_mds_dataset(force_recreate=True, num_partitions=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Final Verification & Training Configuration

# COMMAND ----------

def print_summary():
    """Print final summary and training configuration."""
    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    
    all_ready = True
    
    # Check all components
    checks = [
        ("Token Dictionary", PATHS['token_dictionary'], os.path.exists(PATHS['token_dictionary'])),
        ("Train MDS", PATHS['train_dir'], check_mds_valid(PATHS['train_dir'])),
        ("Test MDS", PATHS['test_dir'], check_mds_valid(PATHS['test_dir'])),
    ]
    
    for name, path, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {path}")
        if not status:
            all_ready = False
    
    print("=" * 60)
    
    if all_ready:
        print("\n🎉 ALL DATA IS READY FOR TRAINING!")
        print("\n" + "=" * 60)
        print("COPY THIS TO parameters.yaml:")
        print("=" * 60)
        print(f"""
# ============================================
# Databricks Volume Configuration
# ============================================
volume:
  catalog: {CATALOG}
  schema: {SCHEMA}
  volume_name: {VOLUME_NAME}

# Data paths relative to the volume root
data:
  source_dataset: {SOURCE_DATASET_PATH}
  streaming_dataset: {STREAMING_DATASET_PATH}
  token_dictionary: {TOKEN_DICT_PATH}
  test_split_ratio: {TEST_SPLIT_RATIO}

# Checkpoint path relative to volume root
checkpoints:
  folder: geneformer/checkpoints
""")
        print("=" * 60)
        print("\nNow you can run the training with:")
        print("  composer train.py parameters.yaml")
    else:
        print("\n❌ Some data is missing. Run the cells above to prepare it.")

print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Optional) Cleanup Source Dataset
# MAGIC 
# MAGIC After MDS conversion, you can optionally delete the source HuggingFace dataset to save space.
# MAGIC **Only run this if MDS conversion was successful!**

# COMMAND ----------

def cleanup_source_dataset():
    """Remove source dataset to save space (optional)."""
    if not (check_mds_valid(PATHS['train_dir']) and check_mds_valid(PATHS['test_dir'])):
        print("❌ MDS dataset is not valid. Cannot cleanup source dataset.")
        return
    
    if not os.path.exists(PATHS['source_dataset']):
        print("Source dataset already removed.")
        return
    
    size = get_dir_size(PATHS['source_dataset'])
    print(f"Removing source dataset: {PATHS['source_dataset']} ({size})")
    shutil.rmtree(PATHS['source_dataset'])
    print("✅ Source dataset removed.")

# Uncomment to run cleanup
# cleanup_source_dataset()

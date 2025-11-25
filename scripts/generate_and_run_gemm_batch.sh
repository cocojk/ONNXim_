#!/usr/bin/env bash
# generate_and_run_gemm_batch.sh
#
# Generate models_list JSON files with different batch sizes and provide commands
# to run the simulator with both sequential (simple) and concurrent (time_multiplex) schedulers.
# Enhanced version: supports M/K/N sweeps, multiple configs, and parallel execution.
#
# Usage:
#   ./scripts/generate_and_run_gemm_batch.sh --M "1,256,1024" --K "1,256,1024" --N "1,256,1024" --configs "pacHBM.json pacHBM_HC_tm.json" --run
#   ./scripts/generate_and_run_gemm_batch.sh --model_name matmul_1024_49152_12288 --batch_sizes "1 2 4 8"
#
# Options:
#   --model_name    : Name of the model (default: matmul_1024_49152_12288)
#   --batch_sizes   : Space-separated list of batch sizes (default: "1 2 4 8")
#   --M             : Comma-separated list of M dimensions (e.g., "1,256,1024,4096,16384")
#   --K             : Comma-separated list of K dimensions (e.g., "1,256,1024,4096,16384")
#   --N             : Comma-separated list of N dimensions (e.g., "1,256,1024,4096,16384")
#   --request_time  : Request time for each entry (default: 0)
#   --output_dir    : Output directory for models_list files (default: model_lists/)
#   --run           : Actually run the simulator commands (default: just print commands)
#   --scheduler     : Scheduler type: "simple", "time_multiplex", or "both" (default: "both")
#   --config        : Single config file path (default: configs/isca_configs/base/pacHBM.json)
#   --configs       : Space-separated list of config files (overrides --config)
#   --log_dir       : Log directory (default: configs/isca_configs/log/)
#   --parallel      : Number of parallel jobs (default: 10)
#   --debug                Enable debug logging (sets log_level to debug)

set -uo pipefail

# Defaults
MODEL_NAME=""
BATCH_SIZES="1 2 4 8"
M_VALS=""
K_VALS=""
N_VALS=""
REQUEST_TIME=0
OUTPUT_DIR="model_lists"
RUN_SIMULATOR=false
SCHEDULER_TYPE="both"
CONFIG_SIMPLE="configs/isca_configs/base/pacHBM.json"
CONFIG_TM="configs/isca_configs/base/pacHBM_time_multiplex.json"
CONFIGS=""
LOG_DIR="configs/isca_configs/log"
SIMULATOR="./build/bin/Simulator"
MAX_PARALLEL=10
LOG_LEVEL="info"
LOG_ADDON=""


# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --batch_sizes)
      # Normalize batch sizes: convert commas to spaces if present
      BATCH_SIZES=$(echo "$2" | tr ',' ' ')
      shift 2
      ;;
    --M)
      M_VALS="$2"
      shift 2
      ;;
    --K)
      K_VALS="$2"
      shift 2
      ;;
    --N)
      N_VALS="$2"
      shift 2
      ;;
    --request_time)
      REQUEST_TIME="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run)
      RUN_SIMULATOR=true
      shift
      ;;
    --scheduler)
      SCHEDULER_TYPE="$2"
      shift 2
      ;;
    --config)
      CONFIG_SIMPLE="$2"
      shift 2
      ;;
    --configs)
      CONFIGS="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log_addon)
      LOG_ADDON="$2"
      shift 2
      ;;
    --parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --debug)
      LOG_LEVEL="debug"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

Generate models_list JSON files with different batch sizes and run simulator commands.

Options:
  --model_name NAME      Model name (default: matmul_1024_49152_12288)
  --batch_sizes "LIST"   Space or comma-separated batch sizes (default: "1 2 4 8")
  --M "LIST"             Comma-separated M dimensions (e.g., "1,256,1024,4096,16384")
  --K "LIST"             Comma-separated K dimensions (e.g., "1,256,1024,4096,16384")
  --N "LIST"             Comma-separated N dimensions (e.g., "1,256,1024,4096,16384")
  --request_time TIME    Request time for each entry (default: 0)
  --output_dir DIR       Output directory for models_list files (default: model_lists/)
  --run                  Actually run the simulator (default: just print commands)
  --scheduler TYPE       "simple", "time_multiplex", or "both" (default: "both")
  --config PATH          Single config file for simple scheduler (default: pacHBM.json)
  --configs "LIST"       Space-separated list of config files (overrides --config)
  --log_dir DIR          Log directory (default: configs/isca_configs/log/)
  --parallel N           Number of parallel jobs (default: 10)

Examples:
  # Single model with batch sizes
  $0 --model_name matmul_1024_49152_12288 --batch_sizes "1 2 4 8"
  
  # M/K/N sweep with multiple configs
  $0 --M "1,256,1024" --K "1,256,1024" --N "1,256,1024" \\
     --configs "configs/isca_configs/base/pacHBM.json configs/isca_configs/base/pacHBM_HC_tm.json" --run
  
  # M/K/N sweep with parallel execution
  $0 --M "1,256,1024,4096,16384" --K "1,256,1024,4096,16384" --N "1,256,1024,4096,16384" \\
     --configs "configs/isca_configs/base/pacHBM.json configs/isca_configs/base/pacHBM_HC_tm.json" \\
     --parallel 10 --run
  
  # M/K/N sweep with specific batch sizes
  $0 --M "1,256,1024" --K "1,256,1024" --N "1,256,1024" \\
     --batch_sizes "1 2 4" \\
     --configs "configs/isca_configs/base/pacHBM.json" --run
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Determine model base path
MODEL_BASE_PATH="${ONNXIM_HOME:-.}"
if [[ -z "${ONNXIM_HOME:-}" ]]; then
  # Try to detect ONNXIM_HOME from script location
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  MODEL_BASE_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Check if simulator exists
if [[ ! -x "$SIMULATOR" ]]; then
  echo "Warning: Simulator not found at $SIMULATOR" >&2
  if [[ "$RUN_SIMULATOR" == true ]]; then
    echo "Cannot run simulator. Exiting." >&2
    exit 1
  fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Function to check and generate a model
check_and_generate_model() {
  local m_val=$1
  local k_val=$2
  local n_val=$3
  local model_name="matmul_${m_val}_${k_val}_${n_val}"
  local model_dir="${MODEL_BASE_PATH}/models/${model_name}"
  local onnx_file="${model_dir}/${model_name}.onnx"
  
  if [[ -f "$onnx_file" ]]; then
    return 0  # Model exists
  fi
  
  echo "Generating model: $model_name (M=$m_val, K=$k_val, N=$n_val)"
  
  GENERATE_SCRIPT="${MODEL_BASE_PATH}/scripts/generate_gemm_onnx.sh"
  if [[ ! -f "$GENERATE_SCRIPT" ]]; then
    echo "Error: Generate script not found: $GENERATE_SCRIPT" >&2
    return 1
  fi
  
  if bash "$GENERATE_SCRIPT" --M "$m_val" --K "$k_val" --N "$n_val" --onnxim_home "$MODEL_BASE_PATH" >/dev/null 2>&1; then
    return 0
  else
    echo "Error: Failed to generate model $model_name" >&2
    return 1
  fi
}

# Function to generate models_list JSON file
generate_models_list() {
  local model_name=$1
  local batch_size=$2
  local output_file=""
  
  # Validate batch_size is a positive integer
  if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || [[ $batch_size -lt 1 ]]; then
    echo "Error: Invalid batch size: $batch_size (must be a positive integer)" >&2
    return 1
  fi
  
  output_file="${OUTPUT_DIR}/${model_name}_batch${batch_size}.json"

  # Generate JSON content
  echo "{" > "$output_file"
  echo "  \"models\": [" >> "$output_file"
  
  for ((i=0; i<batch_size; i++)); do
    echo "    {" >> "$output_file"
    echo "      \"name\": \"$model_name\"," >> "$output_file"
    echo "      \"request_time\": $REQUEST_TIME" >> "$output_file"
    if [[ $i -lt $((batch_size - 1)) ]]; then
      echo "    }," >> "$output_file"
    else
      echo "    }" >> "$output_file"
    fi
  done
  
  echo "  ]" >> "$output_file"
  echo "}" >> "$output_file"
  
  echo "$output_file"
}

# Function to get config name for log file
get_config_name() {
  local config_path=$1
  local basename=$(basename "$config_path" .json)
  echo "$basename"
}

# Function to run a single simulation
run_simulation() {
  local config_path=$1
  local models_list=$2
  local model_name=$3
  local batch_size=$4
  local config_name=$(get_config_name "$config_path")
  
  # Determine log file name
  local log_file="${LOG_DIR}/gemm_${config_name}_${model_name}${LOG_ADDON}"
  if [[ $batch_size -gt 1 ]]; then
    log_file="${log_file}_batch${batch_size}"
  fi
  log_file="${log_file}.log"
  
  local cmd="$SIMULATOR --config $config_path --models_list $models_list --log_level  $LOG_LEVEL > $log_file 2>&1"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $model_name (batch=$batch_size, config=$config_name)"
  
  # Run command (may exit with non-zero due to ASAN memory leaks, but simulation may still succeed)
  eval "$cmd" || true
  
  # Check if simulation actually completed by looking for "Simulation time:" in log
  if [[ -f "$log_file" ]]; then
    if grep -q "Simulation time:" "$log_file" 2>/dev/null; then
      local sim_time=$(grep "Simulation time:" "$log_file" | tail -1 | grep -oP 'time: \K[0-9.]+' || echo "")
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed: $model_name (batch=$batch_size, config=$config_name) - ${sim_time}s"
      return 0
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ Warning: $model_name (batch=$batch_size, config=$config_name) may not have completed" >&2
      return 1
    fi
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ Warning: Log file not found for $model_name (batch=$batch_size, config=$config_name)" >&2
    return 1
  fi
}

# Build list of models to process
MODELS_TO_PROCESS=()

if [[ -n "$M_VALS" && -n "$K_VALS" && -n "$N_VALS" ]]; then
  # M/K/N sweep mode
  echo "=========================================="
  echo "M/K/N Sweep Mode"
  echo "=========================================="
  echo "M values: $M_VALS"
  echo "K values: $K_VALS"
  echo "N values: $N_VALS"
  echo ""
  
  # Convert comma-separated to arrays
  IFS=',' read -ra M_ARRAY <<< "$M_VALS"
  IFS=',' read -ra K_ARRAY <<< "$K_VALS"
  IFS=',' read -ra N_ARRAY <<< "$N_VALS"
  
  # Generate all combinations
  for m_val in "${M_ARRAY[@]}"; do
    for k_val in "${K_ARRAY[@]}"; do
      for n_val in "${N_ARRAY[@]}"; do
        MODELS_TO_PROCESS+=("matmul_${m_val}_${k_val}_${n_val}")
      done
    done
  done
  
  # For M/K/N sweeps, default to batch size 1 if not explicitly specified
  # Check if BATCH_SIZES is still the default value (user didn't specify --batch_sizes)
  if [[ "$BATCH_SIZES" == "1 2 4 8" ]]; then
    BATCH_SIZES="1"  # Default to batch size 1 for M/K/N sweeps to avoid too many combinations
  fi
elif [[ -n "$MODEL_NAME" ]]; then
  # Single model mode
  MODELS_TO_PROCESS=("$MODEL_NAME")
else
  echo "Error: Must specify either --model_name or --M/--K/--N" >&2
  exit 1
fi

echo "Total models to process: ${#MODELS_TO_PROCESS[@]}"
echo "Batch sizes: $BATCH_SIZES"
echo ""

# Step 1: Check and generate all required models
echo "=========================================="
echo "Step 1: Checking and generating models"
echo "=========================================="

MISSING_MODELS=()
for model_name in "${MODELS_TO_PROCESS[@]}"; do
  if [[ "$model_name" =~ ^matmul_([0-9]+)_([0-9]+)_([0-9]+)$ ]]; then
    M_VAL="${BASH_REMATCH[1]}"
    K_VAL="${BASH_REMATCH[2]}"
    N_VAL="${BASH_REMATCH[3]}"
    
    if ! check_and_generate_model "$M_VAL" "$K_VAL" "$N_VAL"; then
      MISSING_MODELS+=("$model_name")
    fi
  else
    # Check if model exists
    MODEL_DIR="${MODEL_BASE_PATH}/models/${model_name}"
    ONNX_FILE="${MODEL_DIR}/${model_name}.onnx"
    if [[ ! -f "$ONNX_FILE" ]]; then
      echo "Error: Model $model_name not found and cannot auto-generate (must follow format: matmul_M_K_N)" >&2
      MISSING_MODELS+=("$model_name")
    fi
  fi
done

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
  echo "Error: Failed to generate ${#MISSING_MODELS[@]} model(s):" >&2
  printf '  %s\n' "${MISSING_MODELS[@]}" >&2
  exit 1
fi

echo "✓ All models are available"
echo ""

# Step 2: Generate models_list files
echo "=========================================="
echo "Step 2: Generating models_list files"
echo "=========================================="

GENERATED_FILES=()
for model_name in "${MODELS_TO_PROCESS[@]}"; do
  for batch_size in $BATCH_SIZES; do
    if [[ $batch_size -lt 1 ]]; then
      continue
    fi
    output_file=$(generate_models_list "$model_name" "$batch_size")
    GENERATED_FILES+=("$output_file")
  done
done

echo "Generated ${#GENERATED_FILES[@]} models_list file(s)"
echo ""

# Step 3: Build command queue
echo "=========================================="
echo "Step 3: Building command queue"
echo "=========================================="

COMMAND_QUEUE=()

# Determine which configs to use
if [[ -n "$CONFIGS" ]]; then
  # Use provided configs
  IFS=' ' read -ra CONFIG_ARRAY <<< "$CONFIGS"
else
  # Use default configs based on scheduler type
  CONFIG_ARRAY=()
  if [[ "$SCHEDULER_TYPE" == "simple" || "$SCHEDULER_TYPE" == "both" ]]; then
    CONFIG_ARRAY+=("$CONFIG_SIMPLE")
  fi
  if [[ "$SCHEDULER_TYPE" == "time_multiplex" || "$SCHEDULER_TYPE" == "both" ]]; then
    # Try to find time_multiplex version of config
    if [[ -f "$CONFIG_TM" ]]; then
      CONFIG_ARRAY+=("$CONFIG_TM")
    else
      # Try to infer from CONFIG_SIMPLE
      CONFIG_TM_INFERRED="${CONFIG_SIMPLE%.json}_tm.json"
      if [[ -f "$CONFIG_TM_INFERRED" ]]; then
        CONFIG_ARRAY+=("$CONFIG_TM_INFERRED")
      else
        echo "Warning: Time multiplex config not found, skipping" >&2
      fi
    fi
  fi
fi

# Build command queue
for model_name in "${MODELS_TO_PROCESS[@]}"; do
  for batch_size in $BATCH_SIZES; do
    if [[ $batch_size -lt 1 ]]; then
      continue
    fi
    
    # Determine models_list filename
    if [[ $batch_size -eq 1 ]]; then
      models_list="${OUTPUT_DIR}/${model_name}.json"
    else
      models_list="${OUTPUT_DIR}/${model_name}_batch${batch_size}.json"
    fi
    
    # Add command for each config
    for config_path in "${CONFIG_ARRAY[@]}"; do
      if [[ ! -f "$config_path" ]]; then
        echo "Warning: Config file not found: $config_path" >&2
        continue
      fi
      COMMAND_QUEUE+=("$config_path|$models_list|$model_name|$batch_size")
    done
  done
done

echo "Total commands in queue: ${#COMMAND_QUEUE[@]}"
echo ""

# Step 4: Print or run commands
if [[ "$RUN_SIMULATOR" == false ]]; then
  echo "=========================================="
  echo "Simulator Commands (not running)"
  echo "=========================================="
  echo ""
  
  for cmd_entry in "${COMMAND_QUEUE[@]}"; do
    IFS='|' read -r config_path models_list model_name batch_size <<< "$cmd_entry"
    config_name=$(get_config_name "$config_path")
    log_file="${LOG_DIR}/gemm_${config_name}_${model_name}${LOG_ADDON}"
    if [[ $batch_size -gt 1 ]]; then
      log_file="${log_file}_batch${batch_size}"
    fi
    log_file="${log_file}.log"
    
    echo "# $model_name (batch=$batch_size, config=$config_name)"
    echo "$SIMULATOR --config $config_path --models_list $models_list --log_level $LOG_LEVEL > $log_file 2>&1"
    echo ""
  done
  
  echo "=========================================="
  echo "To run these commands, use: --run"
  echo "To run in parallel (10 jobs), use: --run --parallel 10"
  echo "=========================================="
else
  echo "=========================================="
  echo "Step 4: Running Simulator Commands"
  echo "=========================================="
  echo "Running ${#COMMAND_QUEUE[@]} simulation(s) with max $MAX_PARALLEL parallel jobs"
  echo ""
  
  # Parallel execution with queue
  running=0
  pids=()
  completed=0
  failed=0
  
  # Function to wait for a job to complete
  wait_for_job() {
    if wait -n 2>/dev/null; then
      ((completed++))
    else
      ((failed++))
    fi
    ((running--))
    # Clean up finished PIDs
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    pids=("${new_pids[@]}")
  }
  
  # Process queue
  for cmd_entry in "${COMMAND_QUEUE[@]}"; do
    IFS='|' read -r config_path models_list model_name batch_size <<< "$cmd_entry"
    
    # Wait if we're at max parallel
    while [[ $running -ge $MAX_PARALLEL ]]; do
      wait_for_job
    done
    
    # Run simulation in background
    (run_simulation "$config_path" "$models_list" "$model_name" "$batch_size") &
    pid=$!
    pids+=("$pid")
    ((running++))
  done
  
  # Wait for remaining jobs
  while [[ $running -gt 0 ]]; do
    wait_for_job
  done
  
  echo ""
  echo "=========================================="
  echo "All commands completed!"
  echo "  Completed: $completed"
  echo "  Failed: $failed"
  echo "  Total: ${#COMMAND_QUEUE[@]}"
  echo "=========================================="
fi

###############################################################################
# Find the directory of the current script ####################################
###############################################################################
# https://stackoverflow.com/a/246128/5766934
# https://www.delftstack.com/howto/linux/get-script-directory-in-bash/#:~:text=The%20dirname%20command%20is%20a,or%20directly%20executed%20bash%20script.

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
###############################################################################

###############################################################################
# Set up the environment ######################################################
###############################################################################

if [ -z "$KALDI_ROOT" ] && [ -d "/mm1/boeddeker/deploy/kaldi" ]; then
  export KALDI_ROOT=/mm1/boeddeker/deploy/kaldi
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ -d "$SCRIPT_DIR/conda/bin" ]; then
  export PATH="$SCRIPT_DIR/conda/bin:$PATH"
  export PYTHONUSERBASE="$SCRIPT_DIR/pythonuserbase"
fi
export PATH="$SCRIPT_DIR/../bin:$PATH"

###############################################################################

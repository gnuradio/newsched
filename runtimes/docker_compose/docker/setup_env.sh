# Store a variable with the path to the directory in which this file resides
DIR=$PREFIX #e.g. /opt/newsched_prefix/ or $HOME/newsched_prefix
# If sh is bash we can obtain this automatically
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
export PATH="$DIR/bin:$PATH"
export PYTHONPATH="$DIR/lib/python3/site-packages:$DIR/lib/python3/dist-packages:$DIR/lib/python3.8/site-packages:$DIR/lib/python3.8/dist-packages:$DIR/lib64/python3/site-packages:$DIR/lib64/python3/dist-packages:$DIR/lib64/python3.8/site-packages:$DIR/lib64/python3.8/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$DIR/lib:$DIR/lib64/:$DIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$DIR/lib:$DIR/lib64/:$DIR/lib/x86_64-linux-gnu:$LIBRARY_PATH"
export PKG_CONFIG_PATH="$DIR/lib/pkgconfig:$DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
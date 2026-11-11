# Sourced by the cluster/*.sh scripts. Run them on Dev-1 (10.56.0.11).
# Node k (Dev-k) = PP stage k-1, IPoIB 10.56.0.1k. Dev-1 can ssh to all four.
NODES="10.56.0.11 10.56.0.12 10.56.0.13 10.56.0.14"
SRC=$HOME/v41/src
rank_ip() { echo 10.56.0.$((11 + $1)); }
